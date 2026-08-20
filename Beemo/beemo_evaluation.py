"""Leakage-safe Beemo evaluation with record-clustered uncertainty."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from llm_detection.data import data_row_key, stable_int
from llm_detection.evaluation import (
    ALL_METHODS,
    actual_fpr,
    auroc,
    calibration_threshold,
    compact_evaluation_row,
    detector_local_values,
    orientation,
    oriented_score,
    partial_auroc,
    percentile_interval,
    precompute_evaluation_scores,
    tpr,
    write_tidy_csv,
)
from llm_detection.io import atomic_write_json, iter_jsonl


CONDITION_MEMBERS = {
    "original": ("original",),
    "expert": ("expert",),
    "llama_p1": ("llama_p1",),
    "llama_p2": ("llama_p2",),
    "llama_p3": ("llama_p3",),
    "gpt_p1": ("gpt_p1",),
    "gpt_p2": ("gpt_p2",),
    "gpt_p3": ("gpt_p3",),
    "llama_all": ("llama_p1", "llama_p2", "llama_p3"),
    "gpt_all": ("gpt_p1", "gpt_p2", "gpt_p3"),
}


def _select(
    rows: Sequence[dict[str, Any]],
    split: str,
    variants: Iterable[str],
) -> list[dict[str, Any]]:
    allowed = set(variants)
    return [
        row
        for row in rows
        if row["split"] == split and row["beemo_variant"] in allowed
    ]


def _candidate_specs(
    detector: str,
    direction: int,
    base_rows: Sequence[dict[str, Any]],
    quantiles: Sequence[float],
) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = [{}]
    if detector == "lrr":
        nll = np.concatenate(
            [-np.asarray(row["token_features"]["logp"], dtype=float) for row in base_rows]
        )
        log_rank = np.concatenate(
            [np.asarray(row["token_features"]["log_rank"], dtype=float) for row in base_rows]
        )
        for nll_quantile in quantiles:
            for rank_quantile in quantiles:
                candidates.append(
                    {
                        "nll_upper": float(np.quantile(nll, nll_quantile)),
                        "log_rank_upper": float(np.quantile(log_rank, rank_quantile)),
                    }
                )
    else:
        values = np.concatenate(
            [direction * detector_local_values(row, detector) for row in base_rows]
        )
        for quantile in quantiles:
            candidates.append(
                {"lower": float(np.quantile(values, 1.0 - float(quantile)))}
            )
    return candidates


def tune_balanced_clipping_spec(
    rows: Sequence[dict[str, Any]],
    detector: str,
    direction: int,
    quantiles: Sequence[float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Give original, expert, Llama, and GPT families equal objective weight."""
    human = _select(rows, "clipping_tuning", ("human",))
    original = _select(rows, "clipping_tuning", ("original",))
    families = {
        "original": original,
        "expert": _select(rows, "clipping_tuning", ("expert",)),
        "llama": _select(
            rows, "clipping_tuning", ("llama_p1", "llama_p2", "llama_p3")
        ),
        "gpt": _select(rows, "clipping_tuning", ("gpt_p1", "gpt_p2", "gpt_p3")),
    }
    if not human or not original or any(not values for values in families.values()):
        raise ValueError("Beemo clipping tuning split is incomplete")
    best_spec: dict[str, float] = {}
    best_objective = -math.inf
    best_family_aurocs: dict[str, float] = {}
    for spec in _candidate_specs(
        detector, direction, [*human, *original], quantiles
    ):
        human_scores = [oriented_score(row, detector, direction, spec) for row in human]
        family_aurocs = {
            name: auroc(
                human_scores,
                [oriented_score(row, detector, direction, spec) for row in family_rows],
            )
            for name, family_rows in families.items()
        }
        objective = float(np.mean(list(family_aurocs.values())))
        if objective > best_objective + 1e-15:
            best_objective = objective
            best_spec = spec
            best_family_aurocs = family_aurocs
    return best_spec, {
        "balanced_family_mean_auroc": best_objective,
        **{f"{name}_auroc": value for name, value in best_family_aurocs.items()},
    }


def _indices(
    rows: Sequence[dict[str, Any]], split: str, variants: Iterable[str]
) -> np.ndarray:
    allowed = set(variants)
    return np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["split"] == split and row["beemo_variant"] in allowed
        ],
        dtype=np.int64,
    )


def _score_map(
    rows: Sequence[dict[str, Any]],
    scores: np.ndarray,
    row_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for index in row_indices:
        grouped[str(rows[int(index)]["sample_id"])].append(float(scores[int(index)]))
    return {
        sample_id: np.asarray(values, dtype=float)
        for sample_id, values in grouped.items()
    }


def _flatten_sampled(
    grouped: dict[str, np.ndarray], sampled_ids: Sequence[str]
) -> np.ndarray:
    return np.concatenate([grouped[sample_id] for sample_id in sampled_ids])


def _bootstrap_condition(
    human_raw: dict[str, np.ndarray],
    human_clipped: dict[str, np.ndarray],
    condition_raw: dict[str, np.ndarray],
    condition_clipped: dict[str, np.ndarray],
    original_raw: dict[str, np.ndarray],
    original_clipped: dict[str, np.ndarray],
    raw_threshold: float,
    clipped_threshold: float,
    max_fpr: float,
    repetitions: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    source_ids = sorted(human_raw)
    expected = set(source_ids)
    for name, grouped in {
        "human_clipped": human_clipped,
        "condition_raw": condition_raw,
        "condition_clipped": condition_clipped,
        "original_raw": original_raw,
        "original_clipped": original_clipped,
    }.items():
        if set(grouped) != expected:
            raise ValueError(f"Beemo bootstrap source ids disagree for {name}")
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(int(repetitions)):
        sampled = rng.integers(0, len(source_ids), size=len(source_ids))
        sampled_ids = [source_ids[int(index)] for index in sampled]
        raw_h = _flatten_sampled(human_raw, sampled_ids)
        clipped_h = _flatten_sampled(human_clipped, sampled_ids)
        raw_m = _flatten_sampled(condition_raw, sampled_ids)
        clipped_m = _flatten_sampled(condition_clipped, sampled_ids)
        raw_original = _flatten_sampled(original_raw, sampled_ids)
        clipped_original = _flatten_sampled(original_clipped, sampled_ids)
        raw_stats = {
            "actual_fpr": actual_fpr(raw_h, raw_threshold),
            "tpr": tpr(raw_m, raw_threshold),
            "auroc": auroc(raw_h, raw_m),
            "partial_auroc": partial_auroc(raw_h, raw_m, max_fpr),
        }
        clipped_stats = {
            "actual_fpr": actual_fpr(clipped_h, clipped_threshold),
            "tpr": tpr(clipped_m, clipped_threshold),
            "auroc": auroc(clipped_h, clipped_m),
            "partial_auroc": partial_auroc(clipped_h, clipped_m, max_fpr),
        }
        for aggregation, stats in (("raw", raw_stats), ("clipped", clipped_stats)):
            for metric, value in stats.items():
                values[f"{aggregation}_{metric}"].append(value)
        for metric in ("tpr", "auroc", "partial_auroc"):
            values[f"clipped_minus_raw_{metric}"].append(
                clipped_stats[metric] - raw_stats[metric]
            )
        values["raw_condition_minus_original_tpr"].append(
            raw_stats["tpr"] - tpr(raw_original, raw_threshold)
        )
        values["clipped_condition_minus_original_tpr"].append(
            clipped_stats["tpr"] - tpr(clipped_original, clipped_threshold)
        )
        values["raw_condition_minus_original_auroc"].append(
            raw_stats["auroc"] - auroc(raw_h, raw_original)
        )
        values["clipped_condition_minus_original_auroc"].append(
            clipped_stats["auroc"] - auroc(clipped_h, clipped_original)
        )
    return {name: percentile_interval(samples) for name, samples in values.items()}


def _subgroup_rows(
    rows: Sequence[dict[str, Any]],
    precomputed: Any,
    detector: str,
    thresholds: dict[tuple[str, float], float],
    target_fprs: Sequence[float],
    max_fpr: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    test_rows = [row for row in rows if row["split"] == "test"]
    subgroup_specs: list[tuple[str, str, set[str]]] = []
    for field in ("category", "source_generator_model"):
        for value in sorted({str(row[field]) for row in test_rows}):
            subgroup_specs.append(
                (field, value, {str(row["sample_id"]) for row in test_rows if str(row[field]) == value})
            )
    expert_rows = [row for row in test_rows if row["beemo_variant"] == "expert"]
    bins = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0000001))
    for lower, upper in bins:
        ids = {
            str(row["sample_id"])
            for row in expert_rows
            if lower <= float(row["word_edit_ratio_from_original"]) < upper
        }
        if ids:
            subgroup_specs.append(("expert_word_edit_ratio", f"[{lower:.1f},{min(upper, 1.0):.1f})", ids))
    for subgroup_type, subgroup_value, ids in subgroup_specs:
        human_indices = np.asarray(
            [index for index, row in enumerate(rows) if row["split"] == "test" and row["beemo_variant"] == "human" and str(row["sample_id"]) in ids],
            dtype=np.int64,
        )
        for condition in ("original", "expert"):
            condition_indices = np.asarray(
                [index for index, row in enumerate(rows) if row["split"] == "test" and row["beemo_variant"] == condition and str(row["sample_id"]) in ids],
                dtype=np.int64,
            )
            if not len(human_indices) or not len(condition_indices):
                continue
            for aggregation, values in (("raw", precomputed.raw), ("clipped", precomputed.clipped)):
                human_scores = values[human_indices]
                machine_scores = values[condition_indices]
                for target_fpr in target_fprs:
                    threshold = thresholds[(aggregation, float(target_fpr))]
                    output.append(
                        {
                            "detector": detector,
                            "aggregation": aggregation,
                            "target_fpr": target_fpr,
                            "condition": condition,
                            "subgroup_type": subgroup_type,
                            "subgroup_value": subgroup_value,
                            "n_records": len(ids),
                            "actual_fpr": actual_fpr(human_scores, threshold),
                            "tpr": tpr(machine_scores, threshold),
                            "auroc": auroc(human_scores, machine_scores),
                            "partial_auroc_0_5_fpr": partial_auroc(human_scores, machine_scores, max_fpr),
                        }
                    )
    return output


def evaluate_beemo(
    target_score_paths: Mapping[str, str | Path] | str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    binoculars_score_path: str | Path | None = None,
    *,
    bootstrap_repetitions: int | None = None,
) -> dict[str, Any]:
    configured_models = {
        str(model["key"]): model for model in config["scoring_models"]
    }
    if isinstance(target_score_paths, Mapping):
        path_items = list(target_score_paths.items())
    else:
        enabled = [
            model for model in config["scoring_models"] if model.get("enabled", False)
        ]
        if len(enabled) != 1:
            raise ValueError("multiple Beemo scorers require a scorer-to-path mapping")
        path_items = [(str(enabled[0]["key"]), target_score_paths)]
    sources: dict[str, list[dict[str, Any]]] = {}
    source_keys: set[tuple[Any, ...]] | None = None
    for scorer_key, score_path in path_items:
        if scorer_key not in configured_models:
            raise ValueError(f"unknown Beemo scorer key: {scorer_key}")
        rows = [compact_evaluation_row(row) for row in iter_jsonl(score_path)]
        if not rows:
            raise ValueError(f"Beemo target score file is empty for {scorer_key}")
        if {row.get("scorer_key") for row in rows} != {scorer_key}:
            raise ValueError(f"Beemo score rows have the wrong scorer key for {scorer_key}")
        expected_model = configured_models[scorer_key]["id"]
        if {row.get("scoring_model") for row in rows} != {expected_model}:
            raise ValueError(f"Beemo score rows have the wrong model for {scorer_key}")
        keys = {data_row_key(row) for row in rows}
        if source_keys is None:
            source_keys = keys
        elif keys != source_keys:
            raise ValueError("Beemo scorer files do not contain matching row keys")
        sources[scorer_key] = rows
    if binoculars_score_path is not None:
        binoculars_rows = [compact_evaluation_row(row) for row in iter_jsonl(binoculars_score_path)]
        if source_keys != {data_row_key(row) for row in binoculars_rows}:
            raise ValueError("Beemo Binoculars keys do not match target score keys")
        sources["binoculars"] = binoculars_rows
    eval_config = config["evaluation"]
    repetitions = int(bootstrap_repetitions or eval_config["bootstrap_repetitions"])
    target_fprs = [float(value) for value in eval_config["target_fprs"]]
    max_fpr = float(eval_config["partial_auroc_max_fpr"])
    conditions = list(eval_config["reported_conditions"])
    if any(condition not in CONDITION_MEMBERS for condition in conditions):
        raise ValueError("Beemo config contains an unknown reported condition")
    configured = list(config["scoring"]["detectors"])
    available = [
        detector
        for detector in configured
        if detector != "binoculars" or "binoculars" in sources
    ]
    unknown = sorted(set(available) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"unknown Beemo detectors: {unknown}")
    metric_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    frozen_specs: dict[str, Any] = {"scorers": {}}
    configurations = [
        (
            scorer_key,
            str(configured_models[scorer_key]["id"]),
            detector,
            sources[scorer_key],
        )
        for scorer_key, _ in path_items
        for detector in available
        if detector != "binoculars"
    ]
    if "binoculars" in sources:
        configurations.append(
            (
                "binoculars",
                f"{config['binoculars']['observer']}::{config['binoculars']['performer']}",
                "binoculars",
                sources["binoculars"],
            )
        )
    for scorer_key, scoring_model, detector, rows in configurations:
        tune_human = _select(rows, "clipping_tuning", ("human",))
        tune_original = _select(rows, "clipping_tuning", ("original",))
        direction = orientation(tune_human, tune_original, detector)
        clip_spec, tuning_diagnostics = tune_balanced_clipping_spec(
            rows, detector, direction, eval_config["clipping_quantiles"]
        )
        precomputed = precompute_evaluation_scores(
            rows, detector, direction, clip_spec, "beemo_native_balanced_tuning"
        )
        calibration_indices = _indices(rows, "calibration", ("human",))
        test_human_indices = _indices(rows, "test", ("human",))
        original_indices = _indices(rows, "test", ("original",))
        thresholds: dict[tuple[str, float], float] = {}
        for aggregation, values in (("raw", precomputed.raw), ("clipped", precomputed.clipped)):
            calibration_scores = values[calibration_indices]
            for target_fpr in target_fprs:
                thresholds[(aggregation, target_fpr)] = calibration_threshold(calibration_scores, target_fpr)
        scorer_specs = frozen_specs["scorers"].setdefault(
            scorer_key,
            {"scoring_model": scoring_model, "detectors": {}},
        )
        scorer_specs["detectors"][detector] = {
            "direction": direction,
            "clipping_specification": clip_spec,
            "tuning_diagnostics": tuning_diagnostics,
            "thresholds": {
                aggregation: {str(target_fpr): thresholds[(aggregation, target_fpr)] for target_fpr in target_fprs}
                for aggregation in ("raw", "clipped")
            },
        }
        human_maps = {
            aggregation: _score_map(rows, values, test_human_indices)
            for aggregation, values in (("raw", precomputed.raw), ("clipped", precomputed.clipped))
        }
        original_maps = {
            aggregation: _score_map(rows, values, original_indices)
            for aggregation, values in (("raw", precomputed.raw), ("clipped", precomputed.clipped))
        }
        for condition in conditions:
            condition_indices = _indices(rows, "test", CONDITION_MEMBERS[condition])
            condition_maps = {
                aggregation: _score_map(rows, values, condition_indices)
                for aggregation, values in (("raw", precomputed.raw), ("clipped", precomputed.clipped))
            }
            for target_fpr in target_fprs:
                intervals = _bootstrap_condition(
                    human_maps["raw"],
                    human_maps["clipped"],
                    condition_maps["raw"],
                    condition_maps["clipped"],
                    original_maps["raw"],
                    original_maps["clipped"],
                    thresholds[("raw", target_fpr)],
                    thresholds[("clipped", target_fpr)],
                    max_fpr,
                    repetitions,
                    int(eval_config["bootstrap_seed"])
                    + stable_int(scorer_key, detector, condition, bits=20),
                )
                point: dict[str, dict[str, float]] = {}
                for aggregation, values in (("raw", precomputed.raw), ("clipped", precomputed.clipped)):
                    human_scores = values[test_human_indices]
                    machine_scores = values[condition_indices]
                    original_scores = values[original_indices]
                    threshold = thresholds[(aggregation, target_fpr)]
                    point[aggregation] = {
                        "actual_fpr": actual_fpr(human_scores, threshold),
                        "tpr": tpr(machine_scores, threshold),
                        "auroc": auroc(human_scores, machine_scores),
                        "partial_auroc": partial_auroc(human_scores, machine_scores, max_fpr),
                        "condition_minus_original_tpr": tpr(machine_scores, threshold) - tpr(original_scores, threshold),
                        "condition_minus_original_auroc": auroc(human_scores, machine_scores) - auroc(human_scores, original_scores),
                    }
                for aggregation in ("raw", "clipped"):
                    clipped_difference = {
                        metric: point["clipped"][metric] - point["raw"][metric]
                        for metric in ("tpr", "auroc", "partial_auroc")
                    }
                    row = {
                        "run_protocol": config["protocol_name"],
                        "result_label": config["result_label"],
                        "dataset": "beemo",
                        "scorer_key": scorer_key,
                        "scoring_model": scoring_model,
                        "detector": detector,
                        "score_source": "falcon_binoculars_pair"
                        if detector == "binoculars"
                        else "reference_single_pass_output_only",
                        "analysis": "beemo_native_balanced_tuning",
                        "condition": condition,
                        "condition_members": json.dumps(CONDITION_MEMBERS[condition]),
                        "aggregation": aggregation,
                        "direction": direction,
                        "clipping_specification": json.dumps(clip_spec if aggregation == "clipped" else {}, sort_keys=True),
                        "target_fpr": target_fpr,
                        "calibration_threshold": thresholds[(aggregation, target_fpr)],
                        "actual_fpr": point[aggregation]["actual_fpr"],
                        "actual_fpr_ci_low": intervals[f"{aggregation}_actual_fpr"][0],
                        "actual_fpr_ci_high": intervals[f"{aggregation}_actual_fpr"][1],
                        "tpr": point[aggregation]["tpr"],
                        "tpr_ci_low": intervals[f"{aggregation}_tpr"][0],
                        "tpr_ci_high": intervals[f"{aggregation}_tpr"][1],
                        "auroc": point[aggregation]["auroc"],
                        "auroc_ci_low": intervals[f"{aggregation}_auroc"][0],
                        "auroc_ci_high": intervals[f"{aggregation}_auroc"][1],
                        "partial_auroc_0_5_fpr": point[aggregation]["partial_auroc"],
                        "partial_auroc_ci_low": intervals[f"{aggregation}_partial_auroc"][0],
                        "partial_auroc_ci_high": intervals[f"{aggregation}_partial_auroc"][1],
                        "clipped_minus_raw_tpr": clipped_difference["tpr"] if aggregation == "clipped" else 0.0,
                        "clipped_minus_raw_tpr_ci_low": intervals["clipped_minus_raw_tpr"][0] if aggregation == "clipped" else 0.0,
                        "clipped_minus_raw_tpr_ci_high": intervals["clipped_minus_raw_tpr"][1] if aggregation == "clipped" else 0.0,
                        "clipped_minus_raw_auroc": clipped_difference["auroc"] if aggregation == "clipped" else 0.0,
                        "clipped_minus_raw_auroc_ci_low": intervals["clipped_minus_raw_auroc"][0] if aggregation == "clipped" else 0.0,
                        "clipped_minus_raw_auroc_ci_high": intervals["clipped_minus_raw_auroc"][1] if aggregation == "clipped" else 0.0,
                        "clipped_minus_raw_partial_auroc": clipped_difference["partial_auroc"] if aggregation == "clipped" else 0.0,
                        "clipped_minus_raw_partial_auroc_ci_low": intervals["clipped_minus_raw_partial_auroc"][0] if aggregation == "clipped" else 0.0,
                        "clipped_minus_raw_partial_auroc_ci_high": intervals["clipped_minus_raw_partial_auroc"][1] if aggregation == "clipped" else 0.0,
                        "condition_minus_original_tpr": point[aggregation]["condition_minus_original_tpr"],
                        "condition_minus_original_tpr_ci_low": intervals[f"{aggregation}_condition_minus_original_tpr"][0],
                        "condition_minus_original_tpr_ci_high": intervals[f"{aggregation}_condition_minus_original_tpr"][1],
                        "condition_minus_original_auroc": point[aggregation]["condition_minus_original_auroc"],
                        "condition_minus_original_auroc_ci_low": intervals[f"{aggregation}_condition_minus_original_auroc"][0],
                        "condition_minus_original_auroc_ci_high": intervals[f"{aggregation}_condition_minus_original_auroc"][1],
                        "n_clipping_tuning_records": len(tune_human),
                        "n_calibration_human": len(calibration_indices),
                        "n_test_human": len(test_human_indices),
                        "n_test_positive_texts": len(condition_indices),
                        "n_test_record_clusters": len(human_maps[aggregation]),
                        "bootstrap_repetitions": repetitions,
                        "bootstrap_seed": eval_config["bootstrap_seed"],
                    }
                    metric_rows.append(row)
        configuration_subgroups = _subgroup_rows(
            rows, precomputed, detector, thresholds, target_fprs, max_fpr
        )
        for subgroup_row in configuration_subgroups:
            subgroup_row["scorer_key"] = scorer_key
            subgroup_row["scoring_model"] = scoring_model
        subgroup_rows.extend(configuration_subgroups)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_tidy_csv(output / "metrics.csv", metric_rows)
    if subgroup_rows:
        write_tidy_csv(output / "subgroup_metrics.csv", subgroup_rows)
    atomic_write_json(output / "frozen_specs.json", frozen_specs)
    expected_rows = len(configurations) * len(conditions) * 2 * len(target_fprs)
    if len(metric_rows) != expected_rows:
        raise ValueError(f"Beemo evaluation wrote {len(metric_rows)} rows; expected {expected_rows}")
    summary = {
        "evaluation_status": "complete",
        "protocol": config["protocol_name"],
        "primary_condition": eval_config["primary_condition"],
        "detectors": available,
        "scorers": [scorer_key for scorer_key, _ in path_items],
        "detector_configurations": [
            {
                "scorer_key": scorer_key,
                "scoring_model": scoring_model,
                "detector": detector,
            }
            for scorer_key, scoring_model, detector, _ in configurations
        ],
        "conditions": conditions,
        "target_fprs": target_fprs,
        "metrics_rows": len(metric_rows),
        "subgroup_rows": len(subgroup_rows),
        "bootstrap_repetitions": repetitions,
        "interpretation": "Expert and LLM edits remain machine-origin positives; human_output is the negative class.",
    }
    atomic_write_json(output / "summary.json", summary)
    return summary
