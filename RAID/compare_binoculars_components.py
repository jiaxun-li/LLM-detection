#!/usr/bin/env python3
"""Compare gap and component-wise Binoculars clipping on a bounded RAID run.

The analysis reuses completed score packs, writes point estimates only, and
does not modify the frozen RAID evaluator or any completed run artifacts.
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
    QUANTILE_GRID,
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


SCOPES = ("full_universal", "eligible_universal")
FAMILIES = ("gap_clipping", "component_clipping")
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


def componentwise_binoculars_score(
    row: Mapping[str, Any], direction: int, specification: Mapping[str, Any] | None
) -> float:
    """Return the oriented raw or component-wise clipped Binoculars score."""
    specification = specification or {}
    if not specification:
        return oriented_document_score(row, "binoculars", direction)
    performer = feat(row, "performer_nll")
    cross_entropy = feat(row, "cross_entropy")
    if len(performer) != len(cross_entropy):
        raise ValueError("Binoculars component arrays must have equal length")
    performer_contribution = np.maximum(
        direction * performer,
        float(specification["performer_oriented_lower"]),
    )
    cross_contribution = np.maximum(
        -direction * cross_entropy,
        float(specification["cross_oriented_lower"]),
    )
    oriented_log_score = float(
        (performer_contribution + cross_contribution).mean()
    )
    return direction * math.exp(direction * oriented_log_score)


def candidate_specifications(
    family: str,
    direction: int,
    clean_rows: Sequence[Mapping[str, Any]],
    quantiles: Sequence[float] = QUANTILE_GRID,
) -> list[dict[str, float]]:
    """Use one shared quantile index, avoiding a 7-by-7 component search."""
    out: list[dict[str, float]] = [{}]
    if family == "gap_clipping":
        gaps = np.concatenate(
            [direction * local(row, "binoculars") for row in clean_rows]
        )
        out.extend(
            {"gap_oriented_lower": float(np.quantile(gaps, 1.0 - quantile))}
            for quantile in quantiles
        )
        return out
    if family != "component_clipping":
        raise ValueError(f"unknown Binoculars clipping family: {family}")
    performer = np.concatenate(
        [direction * feat(row, "performer_nll") for row in clean_rows]
    )
    cross = np.concatenate(
        [-direction * feat(row, "cross_entropy") for row in clean_rows]
    )
    out.extend(
        {
            "performer_oriented_lower": float(
                np.quantile(performer, 1.0 - quantile)
            ),
            "cross_oriented_lower": float(np.quantile(cross, 1.0 - quantile)),
        }
        for quantile in quantiles
    )
    return out


def binoculars_score(
    row: Mapping[str, Any],
    direction: int,
    family: str,
    specification: Mapping[str, Any],
) -> float:
    if not specification:
        return oriented_document_score(row, "binoculars", direction)
    if family == "gap_clipping":
        return oriented_document_score(
            row,
            "binoculars",
            direction,
            {"lower": specification["gap_oriented_lower"]},
        )
    return componentwise_binoculars_score(row, direction, specification)


def _score_map(
    rows: Sequence[Mapping[str, Any]],
    direction: int,
    family: str,
    specification: Mapping[str, Any],
) -> dict[int, float]:
    return {
        id(row): binoculars_score(row, direction, family, specification)
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
    candidate_scores: Mapping[int, float],
    raw_thresholds: Mapping[str, float],
    candidate_thresholds: Mapping[str, float],
) -> dict[str, float | int]:
    raw_human = [raw_scores[id(row)] for row in humans]
    candidate_human = [candidate_scores[id(row)] for row in humans]
    raw_machine = [raw_scores[id(row)] for row in machines]
    candidate_machine = [candidate_scores[id(row)] for row in machines]
    raw_tpr = float(
        np.mean([raw_scores[id(row)] >= raw_thresholds[domain(row)] for row in machines])
    )
    candidate_tpr = float(
        np.mean(
            [
                candidate_scores[id(row)] >= candidate_thresholds[domain(row)]
                for row in machines
            ]
        )
    )
    raw_fpr = float(
        np.mean([raw_scores[id(row)] >= raw_thresholds[domain(row)] for row in humans])
    )
    candidate_fpr = float(
        np.mean(
            [
                candidate_scores[id(row)] >= candidate_thresholds[domain(row)]
                for row in humans
            ]
        )
    )
    raw_auc = auroc(raw_human, raw_machine)
    candidate_auc = auroc(candidate_human, candidate_machine)
    return {
        "n_test_human": len(humans),
        "n_test_machine": len(machines),
        "raw_tpr": raw_tpr,
        "candidate_tpr": candidate_tpr,
        "paired_tpr_difference": candidate_tpr - raw_tpr,
        "raw_auroc": raw_auc,
        "candidate_auroc": candidate_auc,
        "paired_auroc_difference": candidate_auc - raw_auc,
        "raw_test_fpr": raw_fpr,
        "candidate_test_fpr": candidate_fpr,
        "fpr_change": candidate_fpr - raw_fpr,
    }


def compare_binoculars_components(
    falcon_rows: Sequence[Mapping[str, Any]],
    binoculars_rows: Sequence[Mapping[str, Any]],
    *,
    expected_attacks: Sequence[str],
    quantiles: Sequence[float] = QUANTILE_GRID,
) -> dict[str, Any]:
    quantile_grid = tuple(float(value) for value in quantiles)
    if (
        not quantile_grid
        or tuple(sorted(set(quantile_grid))) != quantile_grid
        or any(value <= 0.0 or value >= 1.0 for value in quantile_grid)
    ):
        raise ValueError("quantiles must be unique, increasing, and in (0,1)")

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
    test_attacked = [row for row in test_machines if condition(row) != "none"]
    clean_fit_rows = [*tuning_humans, *tuning_clean]
    fitting_rows = [*clean_fit_rows, *tuning_attacked]
    evaluation_rows = [*calibration_humans, *test_humans, *test_machines]
    direction = learn_direction("binoculars", tuning_humans, tuning_clean)

    candidate_diagnostics: list[dict[str, Any]] = []
    selected_specs: list[dict[str, Any]] = []
    attack_results: list[dict[str, Any]] = []
    rate_results: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    raw_fit_scores = _score_map(fitting_rows, direction, "gap_clipping", {})
    raw_evaluation_scores = _score_map(
        evaluation_rows, direction, "gap_clipping", {}
    )
    raw_thresholds = _thresholds(calibration_humans, raw_evaluation_scores)
    raw_clean_auroc = auroc(
        [raw_fit_scores[id(row)] for row in tuning_humans],
        [raw_fit_scores[id(row)] for row in tuning_clean],
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

        for family in FAMILIES:
            specifications = candidate_specifications(
                family, direction, clean_fit_rows, quantile_grid
            )
            family_diagnostics: list[dict[str, Any]] = []
            fitting_score_maps = [
                _score_map(fitting_rows, direction, family, specification)
                for specification in specifications
            ]
            for candidate_index, (specification, scores) in enumerate(
                zip(specifications, fitting_score_maps)
            ):
                human_scores = [scores[id(row)] for row in tuning_humans]
                clean_auroc = auroc(
                    human_scores, [scores[id(row)] for row in tuning_clean]
                )
                attack_aurocs = {
                    name: auroc(human_scores, [scores[id(row)] for row in group])
                    for name, group in sorted(groups.items())
                }
                mean_attack_auroc = float(np.mean(list(attack_aurocs.values())))
                objective = 0.8 * mean_attack_auroc + 0.2 * clean_auroc
                record = {
                    "method_id": f"{scope}__{family}",
                    "scope": scope,
                    "family": family,
                    "candidate_index": candidate_index,
                    "quantile": (
                        None if candidate_index == 0 else quantile_grid[candidate_index - 1]
                    ),
                    "specification": json.dumps(specification, sort_keys=True),
                    "objective": objective,
                    "mean_attack_auroc": mean_attack_auroc,
                    "clean_auroc": clean_auroc,
                    "raw_clean_auroc": raw_clean_auroc,
                    "clean_auroc_difference": clean_auroc - raw_clean_auroc,
                    "attack_aurocs": json.dumps(attack_aurocs, sort_keys=True),
                    "objective_group_count": len(attack_aurocs),
                }
                candidate_diagnostics.append(record)
                family_diagnostics.append(record)

            best = family_diagnostics[0]
            for candidate in family_diagnostics[1:]:
                if float(candidate["objective"]) > float(best["objective"]) + 1e-12:
                    best = candidate
                elif (
                    abs(float(candidate["objective"]) - float(best["objective"]))
                    <= 1e-12
                    and best["quantile"] is not None
                    and float(candidate["quantile"]) > float(best["quantile"])
                ):
                    # Larger q clips a smaller tail. Raw still wins any tie
                    # because its quantile is None.
                    best = candidate
            selected_specs.append({**best, "direction": direction})
            best_specification = json.loads(str(best["specification"]))
            candidate_scores = _score_map(
                evaluation_rows, direction, family, best_specification
            )
            candidate_thresholds = _thresholds(
                calibration_humans, candidate_scores
            )

            method_attack_rows: list[dict[str, Any]] = []
            for attack in ["none", *attacks]:
                machine_rows = [
                    row for row in test_machines if condition(row) == attack
                ]
                result = {
                    "method_id": best["method_id"],
                    "scope": scope,
                    "family": family,
                    "condition": attack,
                    "quantile": best["quantile"],
                    "specification": best["specification"],
                    **_point(
                        test_humans,
                        machine_rows,
                        raw_evaluation_scores,
                        candidate_scores,
                        raw_thresholds,
                        candidate_thresholds,
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
                    "method_id": best["method_id"],
                    "scope": scope,
                    "family": family,
                    "contamination_bin_index": rate_index,
                    "contamination_bin": label,
                    "quantile": best["quantile"],
                    "specification": best["specification"],
                    **_point(
                        test_humans,
                        machine_rows,
                        raw_evaluation_scores,
                        candidate_scores,
                        raw_thresholds,
                        candidate_thresholds,
                    ),
                }
                rate_results.append(result)
                method_rate_rows.append(result)

            clean_result = next(
                row for row in attack_results
                if row["method_id"] == best["method_id"]
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
            summary.append(
                {
                    "method_id": best["method_id"],
                    "scope": scope,
                    "family": family,
                    "quantile": best["quantile"],
                    "specification": best["specification"],
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

    expected_selected = len(SCOPES) * len(FAMILIES)
    if len(selected_specs) != expected_selected:
        raise AssertionError(
            f"expected {expected_selected} selected specifications, "
            f"found {len(selected_specs)}"
        )
    return {
        "manifest": {
            "analysis": "raid_binoculars_component_comparison_v1",
            "result_label": "PILOT_DEBUG_NOT_FOR_FINAL_REPORTING",
            "target_fpr": TARGET_FPR,
            "quantile_grid": list(quantile_grid),
            "scopes": list(SCOPES),
            "families": list(FAMILIES),
            "selection_objective": "0.8*mean_attack_AUROC+0.2*clean_AUROC",
            "attack_weighting": "equal_across_represented_attack_families",
            "component_search": "one_shared_quantile_index_not_7x7",
            "eligible_interval": "0<rho<=0.5",
            "method_count": expected_selected,
            "reported_approach_count_including_raw": expected_selected + 1,
            "bootstrap_repetitions": 0,
            "development_source_count": len({source(row) for row in rows}),
            "future_full_run_requirement": (
                "exclude every development_source_id before final splitting"
            ),
            "validation": validation,
        },
        "development_sources": {
            "purpose": "pilot Binoculars robust-aggregation model selection",
            "must_be_excluded_from_future_full_benchmark": True,
            "source_count": len({source(row) for row in rows}),
            "source_ids": sorted({source(row) for row in rows}),
        },
        "selected_specs": selected_specs,
        "candidate_diagnostics": candidate_diagnostics,
        "attack_results": attack_results,
        "contamination_rate_results": rate_results,
        "summary": summary,
    }


def write_component_artifacts(
    result: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": root / "comparison_manifest.json",
        "development_sources": root / "development_source_ids.json",
        "selected_specs": root / "selected_specs.csv",
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
            "selected_spec_rows": len(result["selected_specs"]),
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
    parser.add_argument(
        "--output-name", default="binoculars_component_comparison_v1"
    )
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
            "refusing to load an unbounded full run; use the bounded pilot unless "
            "a full-run memory review explicitly allows otherwise"
        )
    falcon_path = run_dir / "falcon_scores.jsonl"
    binoculars_path = run_dir / "binoculars_scores.jsonl"
    if not falcon_path.is_file() or not binoculars_path.is_file():
        raise FileNotFoundError("completed Falcon and Binoculars score packs are required")
    result = compare_binoculars_components(
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
    paths = write_component_artifacts(result, output_dir)
    print("raid_binoculars_component_status=complete")
    print(f"raid_binoculars_component_dir={output_dir}")
    print(f"raid_binoculars_component_summary={paths['summary']}")


if __name__ == "__main__":
    main()
