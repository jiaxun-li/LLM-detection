#!/usr/bin/env python3
"""Evaluate one-sided token trimming on an existing bounded RAID run.

This is an isolated, point-estimate-only pilot analysis.  It reuses completed
Falcon and Binoculars score packs and never changes the frozen RAID evaluator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_detection.evaluation import auroc, calibration_threshold
from llm_detection.io import atomic_write_json, iter_jsonl
from RAID.raid_evaluation import (
    DETECTORS,
    EPS,
    RATE_ADAPTIVE_BIN_INDICES,
    RATE_ADAPTIVE_CUTPOINTS,
    TARGET_FPR,
    _validate,
    attach_contamination_rates,
    condition,
    contamination_bin,
    domain,
    feat,
    human,
    learn_direction,
    local,
    merge_score_rows,
    oriented_document_score,
    rho,
    source,
    split,
)


TRIM_FRACTIONS = (0.0, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20)
SCOPES = ("full_universal", "eligible_universal")
ALL_RATE_INDICES = (0, *RATE_ADAPTIVE_BIN_INDICES, 5)


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def trimming_rule(detector: str) -> str:
    if detector == "lrr":
        return "paired_leave_one_out_influence_trim"
    return "one_sided_oriented_token_trimmed_mean"


def trimmed_document_score(
    row: Mapping[str, Any], detector: str, direction: int, alpha: float
) -> float:
    """Return an oriented document score after removing adverse tokens.

    Additive detectors drop the lowest oriented local contributions.  LRR is a
    ratio, so it drops paired numerator/denominator tokens whose exact
    leave-one-out removal most increases the oriented document ratio.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"trim fraction must be in [0,1), found {alpha}")
    if alpha == 0.0:
        return oriented_document_score(row, detector, direction)

    if detector == "lrr":
        nll = -feat(row, "logp")
        log_rank = feat(row, "log_rank")
        if len(nll) != len(log_rank):
            raise ValueError("LRR token arrays must have equal length")
        count = len(nll)
        remove = min(int(math.floor(alpha * count)), count - 1)
        if remove == 0:
            return oriented_document_score(row, detector, direction)
        total_nll = float(nll.sum())
        total_log_rank = float(log_rank.sum())
        raw_ratio = total_nll / (total_log_rank + EPS)
        leave_one_out = (total_nll - nll) / (total_log_rank - log_rank + EPS)
        removal_gain = direction * (leave_one_out - raw_ratio)
        # Stable ordering makes tied-token behavior deterministic.
        remove_indices = np.argsort(-removal_gain, kind="stable")[:remove]
        keep = np.ones(count, dtype=bool)
        keep[remove_indices] = False
        return direction * float(nll[keep].mean() / (log_rank[keep].mean() + EPS))

    oriented = direction * local(row, detector)
    count = len(oriented)
    remove = min(int(math.floor(alpha * count)), count - 1)
    if remove == 0:
        return oriented_document_score(row, detector, direction)
    retained_mean = float(np.sort(oriented, kind="stable")[remove:].mean())
    if detector == "binoculars":
        return direction * math.exp(direction * retained_mean)
    return retained_mean


def _score_map(
    rows: Sequence[Mapping[str, Any]], detector: str, direction: int, alpha: float
) -> dict[int, float]:
    return {
        id(row): trimmed_document_score(row, detector, direction, alpha)
        for row in rows
    }


def _thresholds(
    humans: Sequence[Mapping[str, Any]], scores: Mapping[int, float]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in humans:
        grouped[domain(row)].append(scores[id(row)])
    return {
        name: calibration_threshold(values, TARGET_FPR)
        for name, values in sorted(grouped.items())
    }


def _point(
    humans: Sequence[Mapping[str, Any]],
    machines: Sequence[Mapping[str, Any]],
    raw_scores: Mapping[int, float],
    trimmed_scores: Mapping[int, float],
    raw_thresholds: Mapping[str, float],
    trimmed_thresholds: Mapping[str, float],
) -> dict[str, float | int]:
    raw_human = [raw_scores[id(row)] for row in humans]
    trim_human = [trimmed_scores[id(row)] for row in humans]
    raw_machine = [raw_scores[id(row)] for row in machines]
    trim_machine = [trimmed_scores[id(row)] for row in machines]
    raw_tpr = float(
        np.mean([raw_scores[id(row)] >= raw_thresholds[domain(row)] for row in machines])
    )
    trim_tpr = float(
        np.mean(
            [
                trimmed_scores[id(row)] >= trimmed_thresholds[domain(row)]
                for row in machines
            ]
        )
    )
    raw_fpr = float(
        np.mean([raw_scores[id(row)] >= raw_thresholds[domain(row)] for row in humans])
    )
    trim_fpr = float(
        np.mean(
            [
                trimmed_scores[id(row)] >= trimmed_thresholds[domain(row)]
                for row in humans
            ]
        )
    )
    raw_auc = auroc(raw_human, raw_machine)
    trim_auc = auroc(trim_human, trim_machine)
    return {
        "n_test_human": len(humans),
        "n_test_machine": len(machines),
        "raw_tpr": raw_tpr,
        "trimmed_tpr": trim_tpr,
        "paired_tpr_difference": trim_tpr - raw_tpr,
        "raw_auroc": raw_auc,
        "trimmed_auroc": trim_auc,
        "paired_auroc_difference": trim_auc - raw_auc,
        "raw_test_fpr": raw_fpr,
        "trimmed_test_fpr": trim_fpr,
        "fpr_change": trim_fpr - raw_fpr,
    }


def compare_trimmed_mean(
    falcon_rows: Sequence[Mapping[str, Any]],
    binoculars_rows: Sequence[Mapping[str, Any]],
    *,
    expected_attacks: Sequence[str],
    trim_fractions: Sequence[float] = TRIM_FRACTIONS,
) -> dict[str, Any]:
    fractions = tuple(float(value) for value in trim_fractions)
    if not fractions or fractions[0] != 0.0 or tuple(sorted(set(fractions))) != fractions:
        raise ValueError("trim fractions must be unique, increasing, and begin with 0")
    if any(value < 0.0 or value >= 1.0 for value in fractions):
        raise ValueError("trim fractions must lie in [0,1)")

    rows = attach_contamination_rates(
        merge_score_rows(falcon_rows, binoculars_rows)
    )
    attacks, validation = _validate(rows, expected_attacks)
    tuning_humans = [
        row for row in rows if split(row) == "clipping_tuning" and human(row)
    ]
    tuning_clean = [
        row
        for row in rows
        if split(row) == "clipping_tuning"
        and not human(row)
        and condition(row) == "none"
    ]
    tuning_attacked = [
        row
        for row in rows
        if split(row) == "clipping_tuning"
        and not human(row)
        and condition(row) != "none"
    ]
    calibration_humans = [
        row for row in rows if split(row) == "calibration" and human(row)
    ]
    test_humans = [row for row in rows if split(row) == "test" and human(row)]
    test_machines = [
        row for row in rows if split(row) == "test" and not human(row)
    ]
    test_clean = [row for row in test_machines if condition(row) == "none"]
    test_attacked = [row for row in test_machines if condition(row) != "none"]

    fit_rows = [*tuning_humans, *tuning_clean, *tuning_attacked]
    evaluation_rows = [*calibration_humans, *test_humans, *test_machines]
    diagnostics: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    attack_results: list[dict[str, Any]] = []
    rate_results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for detector in DETECTORS:
        direction = learn_direction(detector, tuning_humans, tuning_clean)
        fit_maps = [
            _score_map(fit_rows, detector, direction, alpha) for alpha in fractions
        ]
        evaluation_maps = [
            _score_map(evaluation_rows, detector, direction, alpha)
            for alpha in fractions
        ]
        raw_fit = fit_maps[0]
        raw_clean_auc = auroc(
            [raw_fit[id(row)] for row in tuning_humans],
            [raw_fit[id(row)] for row in tuning_clean],
        )

        for scope in SCOPES:
            eligible = scope == "eligible_universal"
            fitting_attacks = [
                row
                for row in tuning_attacked
                if not eligible or 0.0 < rho(row) <= 0.5
            ]
            groups = {
                attack: [row for row in fitting_attacks if condition(row) == attack]
                for attack in attacks
            }
            groups = {name: group for name, group in groups.items() if group}
            if scope == "full_universal" and len(groups) != len(attacks):
                missing = sorted(set(attacks) - set(groups))
                raise ValueError(f"full-universal tuning is missing attacks: {missing}")
            if not groups:
                raise ValueError(f"{scope} has no eligible tuning attack rows")

            scope_diagnostics: list[dict[str, Any]] = []
            for candidate_index, (alpha, scores) in enumerate(
                zip(fractions, fit_maps)
            ):
                human_scores = [scores[id(row)] for row in tuning_humans]
                clean_auc = auroc(
                    human_scores, [scores[id(row)] for row in tuning_clean]
                )
                attack_aurocs = {
                    name: auroc(human_scores, [scores[id(row)] for row in group])
                    for name, group in sorted(groups.items())
                }
                mean_attack_auc = float(np.mean(list(attack_aurocs.values())))
                objective = 0.8 * mean_attack_auc + 0.2 * clean_auc
                record = {
                    "detector": detector,
                    "scope": scope,
                    "candidate_index": candidate_index,
                    "trim_fraction": alpha,
                    "trimming_rule": trimming_rule(detector),
                    "objective": objective,
                    "mean_attack_auroc": mean_attack_auc,
                    "clean_auroc": clean_auc,
                    "raw_clean_auroc": raw_clean_auc,
                    "clean_auroc_difference": clean_auc - raw_clean_auc,
                    "attack_aurocs": json.dumps(attack_aurocs, sort_keys=True),
                    "objective_group_count": len(attack_aurocs),
                }
                diagnostics.append(record)
                scope_diagnostics.append(record)

            # Candidate order is increasing aggressiveness; exact ties retain raw
            # or the less aggressive trim fraction.
            best = scope_diagnostics[0]
            for candidate in scope_diagnostics[1:]:
                if float(candidate["objective"]) > float(best["objective"]) + 1e-12:
                    best = candidate
            selected.append({**best, "direction": direction})

            candidate_index = int(best["candidate_index"])
            raw_scores = evaluation_maps[0]
            trimmed_scores = evaluation_maps[candidate_index]
            raw_thresholds = _thresholds(calibration_humans, raw_scores)
            trimmed_thresholds = _thresholds(calibration_humans, trimmed_scores)

            method_attack_rows: list[dict[str, Any]] = []
            for attack in ["none", *attacks]:
                machine_rows = [
                    row for row in test_machines if condition(row) == attack
                ]
                result = {
                    "detector": detector,
                    "scope": scope,
                    "condition": attack,
                    "trim_fraction": best["trim_fraction"],
                    "trimming_rule": trimming_rule(detector),
                    **_point(
                        test_humans,
                        machine_rows,
                        raw_scores,
                        trimmed_scores,
                        raw_thresholds,
                        trimmed_thresholds,
                    ),
                }
                attack_results.append(result)
                if attack != "none":
                    method_attack_rows.append(result)

            method_rate_rows: list[dict[str, Any]] = []
            for rate_index in ALL_RATE_INDICES:
                machine_rows = [
                    row
                    for row in test_attacked
                    if contamination_bin(rho(row), RATE_ADAPTIVE_CUTPOINTS)[0]
                    == rate_index
                ]
                if not machine_rows:
                    continue
                _, label = contamination_bin(
                    rho(machine_rows[0]), RATE_ADAPTIVE_CUTPOINTS
                )
                result = {
                    "detector": detector,
                    "scope": scope,
                    "contamination_bin_index": rate_index,
                    "contamination_bin": label,
                    "trim_fraction": best["trim_fraction"],
                    "trimming_rule": trimming_rule(detector),
                    **_point(
                        test_humans,
                        machine_rows,
                        raw_scores,
                        trimmed_scores,
                        raw_thresholds,
                        trimmed_thresholds,
                    ),
                }
                rate_results.append(result)
                method_rate_rows.append(result)

            clean_result = next(
                row
                for row in attack_results
                if row["detector"] == detector
                and row["scope"] == scope
                and row["condition"] == "none"
            )
            eligible_rates = [
                row
                for row in method_rate_rows
                if int(row["contamination_bin_index"])
                in RATE_ADAPTIVE_BIN_INDICES
            ]
            dense = [
                row
                for row in method_rate_rows
                if int(row["contamination_bin_index"]) == 5
            ]
            summaries.append(
                {
                    "detector": detector,
                    "scope": scope,
                    "trim_fraction": best["trim_fraction"],
                    "trimming_rule": trimming_rule(detector),
                    "tuning_objective": best["objective"],
                    "clean_tpr_difference": clean_result["paired_tpr_difference"],
                    "clean_auroc_difference": clean_result[
                        "paired_auroc_difference"
                    ],
                    "eligible_mean_tpr_difference": float(
                        np.mean(
                            [row["paired_tpr_difference"] for row in eligible_rates]
                        )
                    ),
                    "eligible_worst_tpr_difference": float(
                        min(row["paired_tpr_difference"] for row in eligible_rates)
                    ),
                    "eligible_mean_auroc_difference": float(
                        np.mean(
                            [row["paired_auroc_difference"] for row in eligible_rates]
                        )
                    ),
                    "eligible_worst_auroc_difference": float(
                        min(row["paired_auroc_difference"] for row in eligible_rates)
                    ),
                    "attack_mean_tpr_difference": float(
                        np.mean(
                            [row["paired_tpr_difference"] for row in method_attack_rows]
                        )
                    ),
                    "attack_worst_tpr_difference": float(
                        min(row["paired_tpr_difference"] for row in method_attack_rows)
                    ),
                    "attack_mean_auroc_difference": float(
                        np.mean(
                            [row["paired_auroc_difference"] for row in method_attack_rows]
                        )
                    ),
                    "attack_worst_auroc_difference": float(
                        min(row["paired_auroc_difference"] for row in method_attack_rows)
                    ),
                    "dense_rate_tpr_difference": (
                        None if not dense else dense[0]["paired_tpr_difference"]
                    ),
                    "dense_rate_auroc_difference": (
                        None if not dense else dense[0]["paired_auroc_difference"]
                    ),
                    "test_fpr_change": clean_result["fpr_change"],
                }
            )

    expected_selected = len(DETECTORS) * len(SCOPES)
    if len(selected) != expected_selected:
        raise AssertionError(
            f"expected {expected_selected} selected trimming rules, found {len(selected)}"
        )
    return {
        "manifest": {
            "analysis": "raid_trimmed_mean_comparison_v1",
            "result_label": "PILOT_DEBUG_NOT_FOR_FINAL_REPORTING",
            "target_fpr": TARGET_FPR,
            "trim_fractions": list(fractions),
            "scopes": list(SCOPES),
            "selection_objective": "0.8*mean_attack_AUROC+0.2*clean_AUROC",
            "attack_weighting": "equal_across_represented_attack_families",
            "eligible_interval": "0<rho<=0.5",
            "method_count": len(SCOPES),
            "reported_approach_count_including_raw": len(SCOPES) + 1,
            "bootstrap_repetitions": 0,
            "development_source_count": len({source(row) for row in rows}),
            "future_full_run_requirement": (
                "exclude every development_source_id before final splitting"
            ),
            "validation": validation,
        },
        "development_sources": {
            "purpose": "pilot trimmed-aggregation model selection",
            "must_be_excluded_from_future_full_benchmark": True,
            "source_count": len({source(row) for row in rows}),
            "source_ids": sorted({source(row) for row in rows}),
        },
        "selected_trimming": selected,
        "candidate_diagnostics": diagnostics,
        "attack_results": attack_results,
        "contamination_rate_results": rate_results,
        "summary": summaries,
    }


def write_trimmed_mean_artifacts(
    result: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": root / "comparison_manifest.json",
        "development_sources": root / "development_source_ids.json",
        "selected_trimming": root / "selected_trimming.csv",
        "candidate_diagnostics": root / "candidate_diagnostics.csv",
        "attack_results": root / "attack_results.csv",
        "contamination_rate_results": root / "contamination_rate_results.csv",
        "summary": root / "summary.csv",
    }
    atomic_write_json(paths["manifest"], result["manifest"])
    atomic_write_json(paths["development_sources"], result["development_sources"])
    for name, path in paths.items():
        if name not in {"manifest", "development_sources"}:
            _csv(path, result[name])
    atomic_write_json(
        root / "comparison.complete.json",
        {
            "status": "complete",
            "analysis": result["manifest"]["analysis"],
            "method_count": result["manifest"]["method_count"],
            "selected_trimming_rows": len(result["selected_trimming"]),
            "candidate_diagnostic_rows": len(result["candidate_diagnostics"]),
            "artifacts": {name: path.name for name, path in paths.items()},
        },
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--config", default="RAID/config.json")
    parser.add_argument("--output-name", default="trimmed_mean_comparison_v1")
    parser.add_argument(
        "--allow-full-run",
        action="store_true",
        help="Permit an unbounded full run only after an explicit memory review.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = workspace / config["paths"]["runs"] / args.run_id
    results_dir = workspace / config["paths"]["results"] / args.run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RAID run manifest is missing: {manifest_path}")
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("limit_sources") is None and not args.allow_full_run:
        raise ValueError(
            "refusing to load an unbounded full run; use this analysis on the bounded "
            "pilot unless a full-run memory review explicitly allows otherwise"
        )
    falcon_path = run_dir / "falcon_scores.jsonl"
    binoculars_path = run_dir / "binoculars_scores.jsonl"
    if not falcon_path.is_file() or not binoculars_path.is_file():
        raise FileNotFoundError("completed Falcon and Binoculars score packs are required")
    result = compare_trimmed_mean(
        list(iter_jsonl(falcon_path)),
        list(iter_jsonl(binoculars_path)),
        expected_attacks=config["dataset"]["required_attacks"],
    )
    result["manifest"].update(
        {
            "run_id": args.run_id,
            "source_limit": run_manifest.get("limit_sources"),
            "falcon_score_path": str(falcon_path),
            "binoculars_score_path": str(binoculars_path),
        }
    )
    output_dir = results_dir / args.output_name
    paths = write_trimmed_mean_artifacts(result, output_dir)
    print("raid_trimmed_mean_status=complete")
    print(f"raid_trimmed_mean_dir={output_dir}")
    print(f"raid_trimmed_mean_summary={paths['summary']}")


if __name__ == "__main__":
    main()
