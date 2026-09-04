"""Leakage-safe clipping, calibration, final metrics, and clustered bootstrap."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .data import data_row_key, stable_int
from .io import iter_jsonl
from .scoring import EPS
from .detector_revision import (REVISION, PAIR_METHODS, ORIGIN_NLL,
    origin_components, origin_score, constant_clean_scores, structurally_constant_candidate)


SINGLE_METHODS = [
    "log_likelihood",
    "rank",
    "log_rank",
    "lrr",
    "entropy",
    "entropy_gap",
]
ALL_METHODS = SINGLE_METHODS + ["binoculars"]
REVISED_METHODS = SINGLE_METHODS + ["binocular_gap", "binocular_origin"]
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
GENERIC_FEATURE = {
    "log_likelihood": "logp",
    "rank": "rank",
    "log_rank": "log_rank",
    "entropy": "entropy",
}
EVALUATION_TOKEN_FEATURES = {
    ORIGIN_NLL,
    "logp",
    "rank",
    "log_rank",
    "entropy",
    "performer_nll",
    "observer_to_performer_cross_entropy",
}
EVALUATION_UNUSED_BULK_FIELDS = {
    "text",
    "prompt",
    "document_features",
}


@dataclass(frozen=True)
class PrecomputedEvaluationScores:
    """Compact per-analysis document scores addressed by stable row index."""

    detector: str
    analysis: str
    raw: np.ndarray
    clipped: np.ndarray
    source_ids: tuple[str, ...]
    source_row_indices: dict[str, np.ndarray]
    elapsed_seconds: float

    def values(self, aggregation: str, row_indices: Sequence[int]) -> np.ndarray:
        if aggregation not in {"raw", "clipped"}:
            raise ValueError(f"unknown aggregation {aggregation!r}")
        scores = self.raw if aggregation == "raw" else self.clipped
        return scores[np.asarray(row_indices, dtype=np.int64)]


def _report_precompute_timing(event: dict[str, Any]) -> None:
    """Emit machine-readable timing without changing scientific output rows."""
    print(
        "evaluation_precompute_timing="
        + json.dumps(event, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def compact_evaluation_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep saved score files rich while minimizing evaluator resident memory."""
    compact = {
        key: value
        for key, value in row.items()
        if key not in EVALUATION_UNUSED_BULK_FIELDS and key != "token_features"
    }
    compact["token_features"] = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in row["token_features"].items()
        if name in EVALUATION_TOKEN_FEATURES
    }
    return compact


def detector_raw_score(row: dict[str, Any], detector: str) -> float:
    if detector == "binocular_origin":
        return origin_score(row)
    if detector == "binocular_gap":
        detector = "binoculars"
    return float(row["doc_scores"][detector])


def detector_local_values(row: dict[str, Any], detector: str) -> np.ndarray:
    features = row["token_features"]
    if detector in GENERIC_FEATURE:
        return np.asarray(features[GENERIC_FEATURE[detector]], dtype=float)
    if detector == "entropy_gap":
        return -np.asarray(features["logp"], dtype=float) - np.asarray(
            features["entropy"], dtype=float
        )
    if detector in {"binoculars", "binocular_gap"}:
        return np.asarray(features["performer_nll"], dtype=float) - np.asarray(
            features["observer_to_performer_cross_entropy"], dtype=float
        )
    raise KeyError(f"{detector} does not have a single local contribution")


def orientation(tuning_human: Sequence[dict[str, Any]], tuning_llm: Sequence[dict[str, Any]], detector: str, revised: bool = False) -> int:
    if detector == "binocular_origin":
        return -1
    if revised and detector == "lrr":
        return 1
    human = np.asarray([detector_raw_score(row, detector) for row in tuning_human])
    llm = np.asarray([detector_raw_score(row, detector) for row in tuning_llm])
    if not len(human) or not len(llm):
        raise ValueError(f"cannot orient {detector} without tuning human and LLM rows")
    return 1 if float(llm.mean()) >= float(human.mean()) else -1


def oriented_score(
    row: dict[str, Any],
    detector: str,
    direction: int,
    clip_spec: dict[str, float] | None = None,
) -> float:
    spec = clip_spec or {}
    if not spec:
        return direction * detector_raw_score(row, detector)
    if detector == "binocular_origin":
        return direction * origin_score(row, spec)
    if detector == "lrr":
        nll = -np.asarray(row["token_features"]["logp"], dtype=float)
        log_rank = np.asarray(row["token_features"]["log_rank"], dtype=float)
        nll = np.minimum(nll, spec["nll_upper"])
        log_rank = np.minimum(log_rank, spec["log_rank_upper"])
        return direction * float(nll.mean() / (log_rank.mean() + EPS))
    local = direction * detector_local_values(row, detector)
    clipped = np.maximum(local, spec["lower"])
    transformed = float(clipped.mean())
    if detector in {"binoculars", "binocular_gap"}:
        # Raw Binoculars is exp(mean local gap); exp is monotone and preserves
        # the calibrated orientation while retaining the documented formula.
        return float(math.exp(direction * transformed)) * direction
    return transformed


def precompute_evaluation_scores(
    rows: Sequence[dict[str, Any]],
    detector: str,
    direction: int,
    clip_spec: dict[str, float],
    analysis: str,
    timing_callback: Callable[[dict[str, Any]], None] | None = None,
) -> PrecomputedEvaluationScores:
    """Calculate each raw and frozen-spec score exactly once for an analysis.

    This function must be called only after orientation and clipping selection
    have finished. Bootstrap code consumes the resulting float arrays by row
    and clustered source index; it never reopens token-level feature arrays.
    """
    started = time.perf_counter()
    raw = np.empty(len(rows), dtype=np.float64)
    clipped = np.empty(len(rows), dtype=np.float64)
    source_ids: list[str] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        raw[row_index] = oriented_score(row, detector, direction)
        clipped[row_index] = oriented_score(
            row, detector, direction, clip_spec
        )
        source_id = str(row["sample_id"])
        source_ids.append(source_id)
        grouped[source_id].append(row_index)
    elapsed = time.perf_counter() - started
    source_row_indices = {
        source_id: np.asarray(indices, dtype=np.int64)
        for source_id, indices in grouped.items()
    }
    result = PrecomputedEvaluationScores(
        detector=detector,
        analysis=analysis,
        raw=raw,
        clipped=clipped,
        source_ids=tuple(source_ids),
        source_row_indices=source_row_indices,
        elapsed_seconds=elapsed,
    )
    if timing_callback is not None:
        timing_callback(
            {
                "analysis": analysis,
                "detector": detector,
                "elapsed_seconds": elapsed,
                "row_count": len(rows),
                "raw_score_count": len(rows),
                "clipped_score_count": len(rows),
            }
        )
    return result


def auroc(human_scores: Sequence[float], llm_scores: Sequence[float]) -> float:
    """Mann-Whitney AUROC with half credit for ties."""
    human = np.asarray(human_scores, dtype=float)
    llm = np.asarray(llm_scores, dtype=float)
    if not len(human) or not len(llm):
        return float("nan")
    ordered_human = np.sort(human)
    strictly_lower = np.searchsorted(ordered_human, llm, side="left")
    lower_or_equal = np.searchsorted(ordered_human, llm, side="right")
    ties = lower_or_equal - strictly_lower
    favorable = strictly_lower.astype(float) + 0.5 * ties
    return float(favorable.sum() / (len(human) * len(llm)))


def roc_points(human_scores: Sequence[float], llm_scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    human = np.asarray(human_scores, dtype=float)
    llm = np.asarray(llm_scores, dtype=float)
    if not len(human) or not len(llm):
        return np.array([]), np.array([])
    scores = np.concatenate([human, llm])
    labels = np.concatenate([np.zeros(len(human), dtype=int), np.ones(len(llm), dtype=int)])
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    labels = labels[order]
    boundaries = np.r_[np.where(np.diff(scores) != 0)[0], len(scores) - 1]
    true_positive = np.cumsum(labels)[boundaries]
    false_positive = (boundaries + 1) - true_positive
    fpr = np.r_[0.0, false_positive / len(human), 1.0]
    tpr = np.r_[0.0, true_positive / len(llm), 1.0]
    return fpr, tpr


def partial_auroc(
    human_scores: Sequence[float],
    llm_scores: Sequence[float],
    max_fpr: float,
) -> float:
    fpr, tpr = roc_points(human_scores, llm_scores)
    if not len(fpr):
        return float("nan")
    below = fpr < max_fpr
    x = list(fpr[below])
    y = list(tpr[below])
    y_at_max = float(np.interp(max_fpr, fpr, tpr))
    x.append(max_fpr)
    y.append(y_at_max)
    return float(_trapezoid(np.asarray(y), np.asarray(x)) / max_fpr)


def calibration_threshold(human_scores: Sequence[float], target_fpr: float) -> float:
    """Conservative upper order-statistic threshold from calibration humans only."""
    values = np.sort(np.asarray(human_scores, dtype=float))
    if not len(values):
        raise ValueError("calibration requires human scores")
    # Strictly exceed at most floor(alpha*n) calibration observations.
    allowed = int(math.floor(target_fpr * len(values)))
    index = max(0, len(values) - allowed - 1)
    return float(np.nextafter(values[index], math.inf))


def actual_fpr(human_scores: Sequence[float], threshold: float) -> float:
    values = np.asarray(human_scores, dtype=float)
    return float(np.mean(values >= threshold)) if len(values) else float("nan")


def tpr(llm_scores: Sequence[float], threshold: float) -> float:
    values = np.asarray(llm_scores, dtype=float)
    return float(np.mean(values >= threshold)) if len(values) else float("nan")


def _candidate_specs(
    detector: str,
    direction: int,
    base_rows: Sequence[dict[str, Any]],
    quantiles: Sequence[float],
) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = [{}]
    if detector == "binocular_origin":
        nll = np.concatenate([origin_components(row)[0] for row in base_rows])
        return candidates + [{"nll_upper": float(np.quantile(nll, q))} for q in quantiles]
    if detector == "lrr":
        nll = np.concatenate(
            [-np.asarray(row["token_features"]["logp"], dtype=float) for row in base_rows]
        )
        log_rank = np.concatenate(
            [np.asarray(row["token_features"]["log_rank"], dtype=float) for row in base_rows]
        )
        for first in quantiles:
            for second in quantiles:
                candidates.append(
                    {
                        "nll_upper": float(np.quantile(nll, first)),
                        "log_rank_upper": float(np.quantile(log_rank, second)),
                    }
                )
    else:
        values = np.concatenate(
            [direction * detector_local_values(row, detector) for row in base_rows]
        )
        for quantile in quantiles:
            candidates.append({"lower": float(np.quantile(values, 1.0 - quantile))})
    return candidates


def tune_clipping_spec(
    detector: str,
    direction: int,
    tuning_human: Sequence[dict[str, Any]],
    tuning_clean_llm: Sequence[dict[str, Any]],
    tuning_mixture: Sequence[dict[str, Any]],
    quantiles: Sequence[float],
    reject_constant: bool = False,
    diagnostics: list | None = None,
) -> dict[str, float]:
    """Tune one frozen spec on the predefined random+tail mixture only."""
    if not tuning_mixture:
        raise ValueError("clipping tuning mixture is empty")
    base_rows = list(tuning_human) + list(tuning_clean_llm)
    best: dict[str, float] = {}
    best_objective = -math.inf
    for spec in _candidate_specs(detector, direction, base_rows, quantiles):
        human = [oriented_score(row, detector, direction, spec) for row in tuning_human]
        clean = [oriented_score(row, detector, direction, spec) for row in tuning_clean_llm]
        if reject_constant and spec and (constant_clean_scores(human, clean) or
                structurally_constant_candidate(base_rows, detector, direction, spec)):
            if diagnostics is not None:
                diagnostics.append({"detector": detector, "specification": spec,
                    "eligible": False, "reason": "constant_clean_tuning_scores"})
            continue
        mixture = [oriented_score(row, detector, direction, spec) for row in tuning_mixture]
        objective = 0.8 * auroc(human, mixture) + 0.2 * auroc(human, clean)
        if objective > best_objective + 1e-15:
            best_objective = objective
            best = spec
    return best


def _clean(rows: Sequence[dict[str, Any]], split: str, label: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["split"] == split
        and row["label"] == label
        and row["contamination_mode"] == "none"
    ]


def _attacked(
    rows: Sequence[dict[str, Any]],
    split: str,
    mode: str,
    ratio: float,
) -> list[dict[str, Any]]:
    if ratio == 0:
        return _clean(rows, split, "llm")
    return [
        row
        for row in rows
        if row["split"] == split
        and row["label"] == "llm"
        and row["contamination_mode"] == mode
        and abs(float(row["requested_contamination_ratio"]) - ratio) < 1e-12
    ]


def _clean_indices(
    rows: Sequence[dict[str, Any]], split: str, label: str
) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["split"] == split
            and row["label"] == label
            and row["contamination_mode"] == "none"
        ],
        dtype=np.int64,
    )


def _attacked_indices(
    rows: Sequence[dict[str, Any]],
    split: str,
    mode: str,
    ratio: float,
) -> np.ndarray:
    if ratio == 0:
        return _clean_indices(rows, split, "llm")
    return np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["split"] == split
            and row["label"] == "llm"
            and row["contamination_mode"] == mode
            and abs(float(row["requested_contamination_ratio"]) - ratio) < 1e-12
        ],
        dtype=np.int64,
    )


def _selected_source_indices(
    precomputed: PrecomputedEvaluationScores,
    row_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    selected = np.zeros(len(precomputed.raw), dtype=bool)
    selected[np.asarray(row_indices, dtype=np.int64)] = True
    return {
        source_id: source_indices[selected[source_indices]]
        for source_id, source_indices in precomputed.source_row_indices.items()
        if np.any(selected[source_indices])
    }


def _bootstrap_index_values(
    scores: np.ndarray,
    grouped_indices: dict[str, np.ndarray],
    sampled_ids: Sequence[str],
) -> np.ndarray:
    chunks = [
        scores[grouped_indices[str(sample_id)]]
        for sample_id in sampled_ids
        if str(sample_id) in grouped_indices
    ]
    if not chunks:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(chunks)


def percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def _metric_bootstrap(
    precomputed: PrecomputedEvaluationScores,
    human_indices: Sequence[int],
    llm_indices: Sequence[int],
    raw_threshold: float,
    clipped_threshold: float,
    max_fpr: float,
    repetitions: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    human_grouped = _selected_source_indices(precomputed, human_indices)
    llm_grouped = _selected_source_indices(precomputed, llm_indices)
    ids = sorted(set(human_grouped) | set(llm_grouped))
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(repetitions):
        sampled = [rng.choice(ids) for _ in ids]
        raw_h = _bootstrap_index_values(
            precomputed.raw, human_grouped, sampled
        )
        raw_m = _bootstrap_index_values(
            precomputed.raw, llm_grouped, sampled
        )
        clip_h = _bootstrap_index_values(
            precomputed.clipped, human_grouped, sampled
        )
        clip_m = _bootstrap_index_values(
            precomputed.clipped, llm_grouped, sampled
        )
        raw_tpr = tpr(raw_m, raw_threshold)
        clip_tpr = tpr(clip_m, clipped_threshold)
        samples["raw_actual_fpr"].append(actual_fpr(raw_h, raw_threshold))
        samples["clipped_actual_fpr"].append(actual_fpr(clip_h, clipped_threshold))
        samples["raw_tpr"].append(raw_tpr)
        samples["clipped_tpr"].append(clip_tpr)
        samples["paired_tpr_difference"].append(clip_tpr - raw_tpr)
        samples["raw_auroc"].append(auroc(raw_h, raw_m))
        samples["clipped_auroc"].append(auroc(clip_h, clip_m))
        samples["paired_auroc_difference"].append(
            samples["clipped_auroc"][-1] - samples["raw_auroc"][-1]
        )
        samples["raw_partial_auroc"].append(partial_auroc(raw_h, raw_m, max_fpr))
        samples["clipped_partial_auroc"].append(partial_auroc(clip_h, clip_m, max_fpr))
        samples["paired_partial_auroc_difference"].append(
            samples["clipped_partial_auroc"][-1]
            - samples["raw_partial_auroc"][-1]
        )
    return {name: percentile_interval(values) for name, values in samples.items()}


def robustness_auc(ratios: Sequence[float], tprs: Sequence[float]) -> float:
    x = np.asarray(ratios, dtype=float)
    y = np.asarray(tprs, dtype=float)
    if len(x) < 2 or x[-1] <= x[0]:
        return float("nan")
    return float(_trapezoid(y, x) / (x[-1] - x[0]))


def _robustness_bootstrap(
    precomputed: PrecomputedEvaluationScores,
    human_indices: Sequence[int],
    llm_indices_by_ratio: Sequence[Sequence[int]],
    ratios: Sequence[float],
    aggregation: str,
    threshold: float,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    scores = (
        precomputed.raw if aggregation == "raw" else precomputed.clipped
    )
    human_grouped = _selected_source_indices(precomputed, human_indices)
    llm_grouped_by_ratio = [
        _selected_source_indices(precomputed, ratio_indices)
        for ratio_indices in llm_indices_by_ratio
    ]
    ids = sorted(
        set(human_grouped).union(
            *[set(grouped) for grouped in llm_grouped_by_ratio]
        )
    )
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        sampled = [rng.choice(ids) for _ in ids]
        curve = [
            tpr(
                _bootstrap_index_values(scores, grouped, sampled),
                threshold,
            )
            for grouped in llm_grouped_by_ratio
        ]
        values.append(robustness_auc(ratios, curve))
    return percentile_interval(values)


def _merge_score_sources(
    target_rows: Sequence[dict[str, Any]],
    binoculars_rows: Sequence[dict[str, Any]] | None,
) -> dict[str, list[dict[str, Any]]]:
    sources = {"single": list(target_rows)}
    if binoculars_rows is not None:
        target_keys = {data_row_key(row) for row in target_rows}
        binoculars_keys = {data_row_key(row) for row in binoculars_rows}
        if len(target_keys)!=len(target_rows) or len(binoculars_keys)!=len(binoculars_rows):
            raise ValueError("duplicate score-row keys")
        if target_keys != binoculars_keys:
            raise ValueError("Binoculars row keys do not match target-model score keys")
        sources["binoculars"] = list(binoculars_rows)
    return sources


def evaluate(
    target_score_path: str | Path,
    output_csv: str | Path,
    config: dict[str, Any],
    binoculars_score_path: str | Path | None = None,
    timing_callback: Callable[[dict[str, Any]], None] | None = _report_precompute_timing,
) -> list[dict[str, Any]]:
    """Run the full independent-split protocol and write tidy CSV output."""
    target_rows = [
        compact_evaluation_row(row) for row in iter_jsonl(target_score_path)
    ]
    binoculars_rows = (
        [
            compact_evaluation_row(row)
            for row in iter_jsonl(binoculars_score_path)
        ]
        if binoculars_score_path
        else None
    )
    sources = _merge_score_sources(target_rows, binoculars_rows)
    eval_config = config["evaluation"]
    revised = eval_config.get("detector_revision") == REVISION
    rejections: list[dict[str, Any]] = []
    ratios = [float(value) for value in config["contamination"]["ratios"]]
    modes = ["random", "tail"]
    output_rows: list[dict[str, Any]] = []

    configured_detectors = list(config["scoring"].get("detectors", ALL_METHODS))
    unknown = sorted(set(configured_detectors) - set(ALL_METHODS + REVISED_METHODS))
    if unknown:
        raise ValueError(f"unknown configured detectors: {unknown}")
    for detector in configured_detectors:
        if detector in PAIR_METHODS:
            if "binoculars" not in sources:
                raise ValueError(
                    "Binoculars is configured but no Binoculars score file was provided"
                )
            rows = sources["binoculars"]
            score_source = "falcon_binoculars_pair"
        else:
            rows = sources["single"]
            score_source = "target_model_single_pass"

        tune_human = _clean(rows, "clipping_tuning", "human")
        tune_clean_llm = _clean(rows, "clipping_tuning", "llm")
        calibrate_human = _clean(rows, "calibration", "human")
        test_human = _clean(rows, "test", "human")
        direction = (orientation(tune_human, tune_clean_llm, detector, revised=True)
                     if revised else orientation(tune_human, tune_clean_llm, detector))
        tuning_options = {"reject_constant": True, "diagnostics": rejections} if revised else {}
        mixture_spec = eval_config["tuning_mixture"]
        tuning_mixture = [
            row
            for row in rows
            if row["split"] == "clipping_tuning"
            and row["contamination_mode"] in mixture_spec["modes"]
            and any(
                abs(float(row["requested_contamination_ratio"]) - float(ratio)) < 1e-12
                for ratio in mixture_spec["ratios"]
            )
        ]
        primary_spec = tune_clipping_spec(
            detector,
            direction,
            tune_human,
            tune_clean_llm,
            tuning_mixture,
            eval_config["clipping_quantiles"],
            **tuning_options,
        )
        analyses: list[tuple[str, str | None, dict[str, float]]] = [
            ("primary_frozen_mixture", None, primary_spec)
        ]
        if eval_config.get("oracle_mode_specific_clipping"):
            for mode in modes:
                oracle_mixture = [row for row in tuning_mixture if row["contamination_mode"] == mode]
                analyses.append(
                    (
                        "oracle_mode_specific",
                        mode,
                        tune_clipping_spec(
                            detector,
                            direction,
                            tune_human,
                            tune_clean_llm,
                            oracle_mixture,
                            eval_config["clipping_quantiles"],
                            **tuning_options,
                        ),
                    )
                )

        for analysis, oracle_mode, clip_spec in analyses:
            applicable_modes = [oracle_mode] if oracle_mode else modes
            # Orientation and every candidate clipping decision above use only
            # clipping-tuning rows. Only after the specification is frozen do
            # calibration/test document scores enter this compact array cache.
            precomputed = precompute_evaluation_scores(
                rows,
                detector,
                direction,
                clip_spec,
                analysis,
                timing_callback,
            )
            calibrate_human_indices = _clean_indices(
                rows, "calibration", "human"
            )
            test_human_indices = _clean_indices(rows, "test", "human")
            thresholds: dict[tuple[str, float], float] = {}
            for aggregation in ("raw", "clipped"):
                calibration_scores = precomputed.values(
                    aggregation, calibrate_human_indices
                )
                for target_fpr in eval_config["target_fprs"]:
                    thresholds[(aggregation, float(target_fpr))] = calibration_threshold(
                        calibration_scores, float(target_fpr)
                    )

            for mode in applicable_modes:
                curve_values: dict[tuple[str, float], list[float]] = defaultdict(list)
                row_indices: dict[tuple[str, float], list[int]] = defaultdict(list)
                llm_indices_by_ratio: list[np.ndarray] = []
                for ratio in ratios:
                    llm_rows = _attacked(rows, "test", mode, ratio)
                    llm_indices = _attacked_indices(rows, "test", mode, ratio)
                    llm_indices_by_ratio.append(llm_indices)
                    if not llm_rows:
                        raise ValueError(
                            f"no final test rows for {detector}/{mode}/{ratio}"
                        )
                    raw_h = precomputed.values("raw", test_human_indices)
                    raw_m = precomputed.values("raw", llm_indices)
                    clipped_h = precomputed.values(
                        "clipped", test_human_indices
                    )
                    clipped_m = precomputed.values("clipped", llm_indices)
                    for target_fpr in [float(x) for x in eval_config["target_fprs"]]:
                        raw_threshold = thresholds[("raw", target_fpr)]
                        clipped_threshold = thresholds[("clipped", target_fpr)]
                        bootstrap = _metric_bootstrap(
                            precomputed,
                            test_human_indices,
                            llm_indices,
                            raw_threshold,
                            clipped_threshold,
                            float(eval_config["partial_auroc_max_fpr"]),
                            int(eval_config["bootstrap_repetitions"]),
                            int(eval_config["bootstrap_seed"])
                            + stable_int(
                                detector, mode, ratio, target_fpr, bits=20
                            ),
                        )
                        point_tpr = {
                            "raw": tpr(raw_m, raw_threshold),
                            "clipped": tpr(clipped_m, clipped_threshold),
                        }
                        paired = point_tpr["clipped"] - point_tpr["raw"]
                        for aggregation, human_scores, machine_scores in (
                            ("raw", raw_h, raw_m),
                            ("clipped", clipped_h, clipped_m),
                        ):
                            prefix = aggregation
                            index = len(output_rows)
                            row_indices[(aggregation, target_fpr)].append(index)
                            curve_values[(aggregation, target_fpr)].append(
                                point_tpr[aggregation]
                            )
                            ci_fpr = bootstrap[f"{prefix}_actual_fpr"]
                            ci_tpr = bootstrap[f"{prefix}_tpr"]
                            ci_auroc = bootstrap[f"{prefix}_auroc"]
                            ci_partial = bootstrap[f"{prefix}_partial_auroc"]
                            paired_ci = bootstrap["paired_tpr_difference"]
                            paired_auroc_ci = bootstrap[
                                "paired_auroc_difference"
                            ]
                            paired_partial_ci = bootstrap[
                                "paired_partial_auroc_difference"
                            ]
                            raw_auroc_point = auroc(raw_h, raw_m)
                            clipped_auroc_point = auroc(clipped_h, clipped_m)
                            raw_partial_point = partial_auroc(
                                raw_h,
                                raw_m,
                                float(eval_config["partial_auroc_max_fpr"]),
                            )
                            clipped_partial_point = partial_auroc(
                                clipped_h,
                                clipped_m,
                                float(eval_config["partial_auroc_max_fpr"]),
                            )
                            output_rows.append(
                                {
                                    "run_id": config["run_id"],
                                    "result_label": config.get(
                                        "result_label", "SCIENTIFIC_RUN"
                                    ),
                                    "debug_only": bool(
                                        config.get("debug_only", False)
                                    ),
                                    "dataset": config["dataset"],
                                    "dataset_id": config["datasets"][config["dataset"]]["id"],
                                    "dataset_revision": rows[0].get(
                                        "dataset_resolved_revision",
                                        config["datasets"][config["dataset"]].get(
                                            "revision"
                                        ),
                                    ),
                                    "model": config["target_model"],
                                    "model_revision": rows[0].get(
                                        "scoring_model_revision",
                                        rows[0].get(
                                            "target_model_resolved_revision",
                                            config.get("target_model_revision"),
                                        ),
                                    ),
                                    "tokenizer_revision": rows[0].get(
                                        "scoring_tokenizer_revision",
                                        rows[0].get(
                                            "target_tokenizer_resolved_revision",
                                            config.get("target_tokenizer_revision"),
                                        ),
                                    ),
                                    "binoculars_performer_model": (
                                        rows[0].get("binoculars_performer_model")
                                        if detector in PAIR_METHODS
                                        else None
                                    ),
                                    "binoculars_performer_revision": (
                                        rows[0].get(
                                            "binoculars_performer_revision"
                                        )
                                        if detector in PAIR_METHODS
                                        else None
                                    ),
                                    "binoculars_observer_model": (
                                        rows[0].get("binoculars_observer_model")
                                        if detector in PAIR_METHODS
                                        else None
                                    ),
                                    "binoculars_observer_revision": (
                                        rows[0].get(
                                            "binoculars_observer_revision"
                                        )
                                        if detector in PAIR_METHODS
                                        else None
                                    ),
                                    "split": "test",
                                    "generation_seed": "pooled",
                                    "generation_seeds": json.dumps(config["generation"]["seeds"]),
                                    "corruption_seed": config["contamination"]["corruption_seed"],
                                    "corruption_draws": config["contamination"]["random_draws"]
                                    if mode == "random"
                                    else 1,
                                    "corruption_draw_id": (
                                        f"pooled:0-{int(config['contamination']['random_draws']) - 1}"
                                        if mode == "random"
                                        else "0"
                                    ),
                                    "contamination_mode": mode,
                                    "requested_contamination_ratio": ratio,
                                    "realized_contamination_ratio_mean": float(
                                        np.mean(
                                            [
                                                row["realized_contamination_ratio"]
                                                for row in llm_rows
                                            ]
                                        )
                                    ),
                                    "detector": detector,
                                    "score_source": score_source,
                                    "aggregation": aggregation,
                                    "analysis": analysis,
                                    "direction": direction,
                                    "clipping_specification": json.dumps(
                                        clip_spec if aggregation == "clipped" else {},
                                        sort_keys=True,
                                    ),
                                    "target_fpr": target_fpr,
                                    "calibration_threshold": thresholds[(aggregation, target_fpr)],
                                    "actual_fpr": actual_fpr(
                                        human_scores, thresholds[(aggregation, target_fpr)]
                                    ),
                                    "actual_fpr_ci_low": ci_fpr[0],
                                    "actual_fpr_ci_high": ci_fpr[1],
                                    "tpr": point_tpr[aggregation],
                                    "tpr_ci_low": ci_tpr[0],
                                    "tpr_ci_high": ci_tpr[1],
                                    "auroc": auroc(human_scores, machine_scores),
                                    "auroc_ci_low": ci_auroc[0],
                                    "auroc_ci_high": ci_auroc[1],
                                    "partial_auroc_0_5_fpr": partial_auroc(
                                        human_scores,
                                        machine_scores,
                                        float(eval_config["partial_auroc_max_fpr"]),
                                    ),
                                    "partial_auroc_ci_low": ci_partial[0],
                                    "partial_auroc_ci_high": ci_partial[1],
                                    "clipped_minus_raw_tpr": paired
                                    if aggregation == "clipped"
                                    else 0.0,
                                    "clipped_minus_raw_tpr_ci_low": paired_ci[0]
                                    if aggregation == "clipped"
                                    else 0.0,
                                    "clipped_minus_raw_tpr_ci_high": paired_ci[1]
                                    if aggregation == "clipped"
                                    else 0.0,
                                    "clipped_minus_raw_auroc": (
                                        clipped_auroc_point - raw_auroc_point
                                        if aggregation == "clipped"
                                        else 0.0
                                    ),
                                    "clipped_minus_raw_auroc_ci_low": paired_auroc_ci[0]
                                    if aggregation == "clipped"
                                    else 0.0,
                                    "clipped_minus_raw_auroc_ci_high": paired_auroc_ci[1]
                                    if aggregation == "clipped"
                                    else 0.0,
                                    "clipped_minus_raw_partial_auroc": (
                                        clipped_partial_point - raw_partial_point
                                        if aggregation == "clipped"
                                        else 0.0
                                    ),
                                    "clipped_minus_raw_partial_auroc_ci_low": paired_partial_ci[0]
                                    if aggregation == "clipped"
                                    else 0.0,
                                    "clipped_minus_raw_partial_auroc_ci_high": paired_partial_ci[1]
                                    if aggregation == "clipped"
                                    else 0.0,
                                    "robustness_auc_tpr_vs_contamination": None,
                                    "robustness_auc_ci_low": None,
                                    "robustness_auc_ci_high": None,
                                    "n_calibration_human": len(calibrate_human),
                                    "n_test_human": len(test_human),
                                    "n_test_llm": len(llm_rows),
                                    "n_unique_source_sample_ids": len(
                                        {
                                            str(row["sample_id"])
                                            for row in test_human + llm_rows
                                        }
                                    ),
                                    "bootstrap_repetitions": eval_config[
                                        "bootstrap_repetitions"
                                    ],
                                    "bootstrap_seed": eval_config["bootstrap_seed"],
                                }
                            )
                for key, values in curve_values.items():
                    curve_auc = robustness_auc(ratios, values)
                    aggregation, target_fpr = key
                    curve_ci = _robustness_bootstrap(
                        precomputed,
                        test_human_indices,
                        llm_indices_by_ratio,
                        ratios,
                        aggregation,
                        thresholds[(aggregation, target_fpr)],
                        int(eval_config["bootstrap_repetitions"]),
                        int(eval_config["bootstrap_seed"])
                        + stable_int(
                            "robustness",
                            detector,
                            mode,
                            aggregation,
                            target_fpr,
                            bits=20,
                        ),
                    )
                    for index in row_indices[key]:
                        output_rows[index][
                            "robustness_auc_tpr_vs_contamination"
                        ] = curve_auc
                        output_rows[index]["robustness_auc_ci_low"] = curve_ci[0]
                        output_rows[index]["robustness_auc_ci_high"] = curve_ci[1]

    write_tidy_csv(output_csv, output_rows)
    if revised:
        from .io import atomic_write_json
        atomic_write_json(Path(output_csv).with_name("clipping_rejections.json"), rejections)
    return output_rows


def write_tidy_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no evaluation rows to write")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
