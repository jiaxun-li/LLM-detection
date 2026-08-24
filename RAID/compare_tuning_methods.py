#!/usr/bin/env python3
"""Compare exploratory RAID clipping selectors without repeating GPU scoring.

This analysis is deliberately separate from the frozen RAID evaluator.  It
reads an existing bounded run's Falcon and Binoculars score packs and writes a
point-estimate-only tuning comparison below that run's results directory.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_detection.evaluation import actual_fpr, auroc, calibration_threshold
from llm_detection.io import atomic_write_json, iter_jsonl
from RAID.raid_evaluation import (
    DETECTORS,
    RATE_ADAPTIVE_BIN_INDICES,
    RATE_ADAPTIVE_CUTPOINTS,
    TARGET_FPR,
    _validate,
    attach_contamination_rates,
    candidate_specifications,
    condition,
    contamination_bin,
    domain,
    human,
    learn_direction,
    merge_score_rows,
    oriented_document_score,
    rho,
    source,
    split,
)


EXPLORATORY_QUANTILE_GRID = (
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
    0.99,
    0.995,
    0.999,
)
CLEAN_LOSS_BUDGETS = (0.0, 0.01, 0.02, 0.05)
UNIVERSAL_SCOPES = ("full_universal", "eligible_universal")
UNIVERSAL_SELECTORS = (
    "mean_attack_auroc",
    "worst_attack_auroc",
    "rate_balanced_auroc",
    "crossfit_tpr_at_5_fpr",
)
ORACLE_SELECTORS = (
    "mean_attack_auroc",
    "worst_attack_auroc",
    "crossfit_tpr_at_5_fpr",
)
ALL_RATE_INDICES = (0, *RATE_ADAPTIVE_BIN_INDICES, 5)


def method_id(scope: str, selector: str) -> str:
    return f"{scope}__{selector}"


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


def _stable_int(*parts: object) -> int:
    value = 2166136261
    for part in parts:
        for byte in str(part).encode("utf-8"):
            value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def _folds(human_rows: Sequence[Mapping[str, Any]], folds: int, seed: int) -> dict[str, int]:
    by_domain: dict[str, list[str]] = defaultdict(list)
    for row in human_rows:
        by_domain[domain(row)].append(source(row))
    if folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    if any(len(values) < folds for values in by_domain.values()):
        sizes = {name: len(values) for name, values in sorted(by_domain.items())}
        raise ValueError(f"too few tuning humans per domain for {folds} folds: {sizes}")
    out: dict[str, int] = {}
    for name, values in sorted(by_domain.items()):
        ordered = sorted(values, key=lambda value: (_stable_int(seed, name, value), value))
        for index, value in enumerate(ordered):
            out[value] = index % folds
    return out


def _attack_groups(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[condition(row)].append(row)
    return dict(sorted(groups.items()))


def _rate_groups(
    rows: Iterable[Mapping[str, Any]], indices: Sequence[int]
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    wanted = set(indices)
    for row in rows:
        index, label = contamination_bin(rho(row), RATE_ADAPTIVE_CUTPOINTS)
        if index in wanted:
            groups[label].append(row)
    return dict(sorted(groups.items()))


def _scores(
    rows: Sequence[Mapping[str, Any]], detector: str, direction: int, specification: Mapping[str, Any]
) -> dict[int, float]:
    return {
        id(row): oriented_document_score(row, detector, direction, specification)
        for row in rows
    }


def _group_auroc_gains(
    score_map: Mapping[int, float],
    raw_map: Mapping[int, float],
    humans: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    candidate_human = [score_map[id(row)] for row in humans]
    raw_human = [raw_map[id(row)] for row in humans]
    return {
        name: auroc(candidate_human, [score_map[id(row)] for row in rows])
        - auroc(raw_human, [raw_map[id(row)] for row in rows])
        for name, rows in groups.items()
        if rows
    }


def _crossfit_hits(
    score_map: Mapping[int, float],
    humans: Sequence[Mapping[str, Any]],
    machines: Sequence[Mapping[str, Any]],
    fold_by_source: Mapping[str, int],
    folds: int,
) -> dict[int, bool]:
    hits: dict[int, bool] = {}
    for fold in range(folds):
        thresholds: dict[str, float] = {}
        domains = sorted({domain(row) for row in humans})
        for name in domains:
            calibration = [
                score_map[id(row)]
                for row in humans
                if domain(row) == name and fold_by_source[source(row)] != fold
            ]
            if not calibration:
                raise ValueError(f"empty cross-fit calibration fold for domain {name}")
            thresholds[name] = calibration_threshold(calibration, TARGET_FPR)
        for row in machines:
            if fold_by_source[source(row)] == fold:
                hits[id(row)] = score_map[id(row)] >= thresholds[domain(row)]
    if len(hits) != len(machines):
        raise AssertionError("cross-fit decisions do not cover every tuning machine row")
    return hits


def _group_tpr_gains(
    hits: Mapping[int, bool],
    raw_hits: Mapping[int, bool],
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    return {
        name: float(np.mean([hits[id(row)] for row in rows]))
        - float(np.mean([raw_hits[id(row)] for row in rows]))
        for name, rows in groups.items()
        if rows
    }


def _candidate_diagnostics(
    *,
    detector: str,
    scope: str,
    selector: str,
    bin_index: int | None,
    bin_label: str | None,
    candidates: Sequence[Mapping[str, Any]],
    score_maps: Sequence[Mapping[int, float]],
    tuning_humans: Sequence[Mapping[str, Any]],
    clean_machine: Sequence[Mapping[str, Any]],
    objective_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    fold_by_source: Mapping[str, int],
    folds: int,
) -> list[dict[str, Any]]:
    raw_map = score_maps[0]
    raw_clean = auroc(
        [raw_map[id(row)] for row in tuning_humans],
        [raw_map[id(row)] for row in clean_machine],
    )
    raw_hits: dict[int, bool] | None = None
    if selector == "crossfit_tpr_at_5_fpr":
        machines = [row for rows in objective_groups.values() for row in rows]
        raw_hits = _crossfit_hits(
            raw_map, tuning_humans, machines, fold_by_source, folds
        )
    diagnostics: list[dict[str, Any]] = []
    for index, (specification, score_map) in enumerate(zip(candidates, score_maps)):
        clean = auroc(
            [score_map[id(row)] for row in tuning_humans],
            [score_map[id(row)] for row in clean_machine],
        )
        if selector == "crossfit_tpr_at_5_fpr":
            machines = [row for rows in objective_groups.values() for row in rows]
            hits = _crossfit_hits(
                score_map, tuning_humans, machines, fold_by_source, folds
            )
            gains = _group_tpr_gains(hits, raw_hits or {}, objective_groups)
            objective = float(np.mean(list(gains.values())))
        else:
            gains = _group_auroc_gains(
                score_map, raw_map, tuning_humans, objective_groups
            )
            if selector == "worst_attack_auroc":
                objective = float(min(gains.values()))
            else:
                objective = float(np.mean(list(gains.values())))
        diagnostics.append(
            {
                "detector": detector,
                "scope": scope,
                "selector": selector,
                "method_id": method_id(scope, selector),
                "contamination_bin_index": bin_index,
                "contamination_bin": bin_label,
                "candidate_index": index,
                "specification": json.dumps(specification, sort_keys=True),
                "objective": objective,
                "clean_auroc": clean,
                "raw_clean_auroc": raw_clean,
                "clean_auroc_loss": raw_clean - clean,
                "objective_group_count": len(gains),
                "objective_group_gains": json.dumps(gains, sort_keys=True),
            }
        )
    return diagnostics


def _select(
    diagnostics: Sequence[Mapping[str, Any]], clean_loss_budget: float
) -> Mapping[str, Any]:
    feasible = [
        row
        for row in diagnostics
        if float(row["clean_auroc_loss"]) <= clean_loss_budget + 1e-12
    ]
    if not feasible:
        raise AssertionError("the no-clipping candidate must always be feasible")
    # Diagnostics preserve candidate order, so exact ties retain no clipping or
    # the least aggressive earlier candidate.
    best = feasible[0]
    for row in feasible[1:]:
        if float(row["objective"]) > float(best["objective"]) + 1e-12:
            best = row
    return best


def _thresholds(
    calibration_humans: Sequence[Mapping[str, Any]],
    score_map: Mapping[int, float],
) -> dict[str, float]:
    by_domain: dict[str, list[float]] = defaultdict(list)
    for row in calibration_humans:
        by_domain[domain(row)].append(score_map[id(row)])
    return {
        name: calibration_threshold(values, TARGET_FPR)
        for name, values in sorted(by_domain.items())
    }


def _point(
    humans: Sequence[Mapping[str, Any]],
    machines: Sequence[Mapping[str, Any]],
    raw_scores: Mapping[int, float],
    clipped_scores: Mapping[int, float],
    raw_thresholds: Mapping[str, float],
    clipped_thresholds: Mapping[str, float],
) -> dict[str, float | int]:
    raw_human = [raw_scores[id(row)] for row in humans]
    clipped_human = [clipped_scores[id(row)] for row in humans]
    raw_machine = [raw_scores[id(row)] for row in machines]
    clipped_machine = [clipped_scores[id(row)] for row in machines]
    raw_tpr = float(
        np.mean(
            [raw_scores[id(row)] >= raw_thresholds[domain(row)] for row in machines]
        )
    )
    clipped_tpr = float(
        np.mean(
            [
                clipped_scores[id(row)] >= clipped_thresholds[domain(row)]
                for row in machines
            ]
        )
    )
    raw_fpr = float(
        np.mean([raw_scores[id(row)] >= raw_thresholds[domain(row)] for row in humans])
    )
    clipped_fpr = float(
        np.mean(
            [
                clipped_scores[id(row)] >= clipped_thresholds[domain(row)]
                for row in humans
            ]
        )
    )
    raw_auc = auroc(raw_human, raw_machine)
    clipped_auc = auroc(clipped_human, clipped_machine)
    return {
        "n_test_human": len(humans),
        "n_test_machine": len(machines),
        "raw_tpr": raw_tpr,
        "clipped_tpr": clipped_tpr,
        "paired_tpr_difference": clipped_tpr - raw_tpr,
        "raw_auroc": raw_auc,
        "clipped_auroc": clipped_auc,
        "paired_auroc_difference": clipped_auc - raw_auc,
        "raw_test_fpr": raw_fpr,
        "clipped_test_fpr": clipped_fpr,
    }


def compare_tuning_methods(
    falcon_rows: Sequence[Mapping[str, Any]],
    binoculars_rows: Sequence[Mapping[str, Any]],
    *,
    expected_attacks: Sequence[str] | None = None,
    quantiles: Sequence[float] = EXPLORATORY_QUANTILE_GRID,
    clean_loss_budgets: Sequence[float] = CLEAN_LOSS_BUDGETS,
    crossfit_folds: int = 5,
    crossfit_seed: int = 260826,
) -> dict[str, Any]:
    rows = merge_score_rows(falcon_rows, binoculars_rows)
    attacks, validation = _validate(rows, expected_attacks)
    attach_contamination_rates(rows)
    tuning_humans = [row for row in rows if split(row) == "clipping_tuning" and human(row)]
    clean_machine = [
        row
        for row in rows
        if split(row) == "clipping_tuning" and not human(row) and condition(row) == "none"
    ]
    tuning_attacked = [
        row
        for row in rows
        if split(row) == "clipping_tuning" and not human(row) and condition(row) != "none"
    ]
    calibration_humans = [row for row in rows if split(row) == "calibration" and human(row)]
    test_humans = [row for row in rows if split(row) == "test" and human(row)]
    test_machines = [row for row in rows if split(row) == "test" and not human(row)]
    test_none = [row for row in test_machines if condition(row) == "none"]
    test_attacked = [row for row in test_machines if condition(row) != "none"]
    fold_by_source = _folds(tuning_humans, crossfit_folds, crossfit_seed)

    selected_specs: list[dict[str, Any]] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    universal_attack_results: list[dict[str, Any]] = []
    contamination_rate_results: list[dict[str, Any]] = []
    oracle_attack_rate_results: list[dict[str, Any]] = []
    clean_cost_results: list[dict[str, Any]] = []
    leaderboard: list[dict[str, Any]] = []

    all_tuning_attack_groups = _attack_groups(tuning_attacked)
    eligible_tuning = [row for row in tuning_attacked if 0 < rho(row) <= 0.5]
    eligible_attack_groups = _attack_groups(eligible_tuning)
    all_rate_groups = _rate_groups(tuning_attacked, ALL_RATE_INDICES)
    eligible_rate_groups = _rate_groups(tuning_attacked, RATE_ADAPTIVE_BIN_INDICES)

    for detector in DETECTORS:
        direction = learn_direction(detector, tuning_humans, clean_machine)
        candidates = candidate_specifications(
            detector,
            direction,
            [*tuning_humans, *clean_machine],
            tuple(float(value) for value in quantiles),
        )
        tuning_rows = [*tuning_humans, *clean_machine, *tuning_attacked]
        score_maps = [
            _scores(tuning_rows, detector, direction, specification)
            for specification in candidates
        ]
        diagnostics_by_fit: dict[tuple[str, str, int | None], list[dict[str, Any]]] = {}

        for scope in UNIVERSAL_SCOPES:
            attack_groups = (
                all_tuning_attack_groups
                if scope == "full_universal"
                else eligible_attack_groups
            )
            rate_groups = all_rate_groups if scope == "full_universal" else eligible_rate_groups
            for selector in UNIVERSAL_SELECTORS:
                groups = rate_groups if selector == "rate_balanced_auroc" else attack_groups
                diagnostics = _candidate_diagnostics(
                    detector=detector,
                    scope=scope,
                    selector=selector,
                    bin_index=None,
                    bin_label=None,
                    candidates=candidates,
                    score_maps=score_maps,
                    tuning_humans=tuning_humans,
                    clean_machine=clean_machine,
                    objective_groups=groups,
                    fold_by_source=fold_by_source,
                    folds=crossfit_folds,
                )
                diagnostics_by_fit[(scope, selector, None)] = diagnostics
                candidate_diagnostics.extend(diagnostics)

        for bin_index in RATE_ADAPTIVE_BIN_INDICES:
            bin_rows = [
                row
                for row in eligible_tuning
                if contamination_bin(rho(row), RATE_ADAPTIVE_CUTPOINTS)[0] == bin_index
            ]
            _, bin_label = contamination_bin(
                RATE_ADAPTIVE_CUTPOINTS[bin_index - 1], RATE_ADAPTIVE_CUTPOINTS
            )
            groups = _attack_groups(bin_rows)
            if not groups:
                raise ValueError(f"empty tuning contamination bin {bin_index}")
            for selector in ORACLE_SELECTORS:
                diagnostics = _candidate_diagnostics(
                    detector=detector,
                    scope="rate_specific_oracle",
                    selector=selector,
                    bin_index=bin_index,
                    bin_label=bin_label,
                    candidates=candidates,
                    score_maps=score_maps,
                    tuning_humans=tuning_humans,
                    clean_machine=clean_machine,
                    objective_groups=groups,
                    fold_by_source=fold_by_source,
                    folds=crossfit_folds,
                )
                diagnostics_by_fit[("rate_specific_oracle", selector, bin_index)] = diagnostics
                candidate_diagnostics.extend(diagnostics)

        evaluation_rows = [*calibration_humans, *test_humans, *test_machines]
        evaluation_cache: dict[int, tuple[dict[int, float], dict[str, float]]] = {}

        def evaluated(candidate_index: int) -> tuple[dict[int, float], dict[str, float]]:
            if candidate_index not in evaluation_cache:
                scores = _scores(
                    evaluation_rows, detector, direction, candidates[candidate_index]
                )
                evaluation_cache[candidate_index] = (
                    scores,
                    _thresholds(calibration_humans, scores),
                )
            return evaluation_cache[candidate_index]

        raw_scores, raw_thresholds = evaluated(0)
        method_rate_rows: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
        method_attack_rows: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
        method_clean_rows: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)

        for budget in clean_loss_budgets:
            budget = float(budget)
            for scope in UNIVERSAL_SCOPES:
                for selector in UNIVERSAL_SELECTORS:
                    method = method_id(scope, selector)
                    selected = _select(diagnostics_by_fit[(scope, selector, None)], budget)
                    candidate_index = int(selected["candidate_index"])
                    clipped_scores, clipped_thresholds = evaluated(candidate_index)
                    selected_specs.append(
                        {
                            **selected,
                            "clean_loss_budget": budget,
                            "direction": direction,
                            "quantile_grid": json.dumps(list(quantiles)),
                        }
                    )
                    clean = {
                        "detector": detector,
                        "method_id": method,
                        "scope": scope,
                        "selector": selector,
                        "clean_loss_budget": budget,
                        "contamination_bin_index": None,
                        "contamination_bin": "none",
                        "specification": selected["specification"],
                        **_point(
                            test_humans,
                            test_none,
                            raw_scores,
                            clipped_scores,
                            raw_thresholds,
                            clipped_thresholds,
                        ),
                    }
                    clean_cost_results.append(clean)
                    method_clean_rows[(method, budget)].append(clean)
                    for attack in ["none", *attacks]:
                        machine_rows = [row for row in test_machines if condition(row) == attack]
                        result = {
                            "detector": detector,
                            "method_id": method,
                            "scope": scope,
                            "selector": selector,
                            "clean_loss_budget": budget,
                            "condition": attack,
                            "specification": selected["specification"],
                            **_point(
                                test_humans,
                                machine_rows,
                                raw_scores,
                                clipped_scores,
                                raw_thresholds,
                                clipped_thresholds,
                            ),
                        }
                        universal_attack_results.append(result)
                        if attack != "none":
                            method_attack_rows[(method, budget)].append(result)
                    for rate_index in ALL_RATE_INDICES:
                        machine_rows = [
                            row
                            for row in test_attacked
                            if contamination_bin(rho(row), RATE_ADAPTIVE_CUTPOINTS)[0]
                            == rate_index
                        ]
                        if not machine_rows:
                            continue
                        _, rate_label = contamination_bin(
                            rho(machine_rows[0]), RATE_ADAPTIVE_CUTPOINTS
                        )
                        result = {
                            "detector": detector,
                            "method_id": method,
                            "scope": scope,
                            "selector": selector,
                            "clean_loss_budget": budget,
                            "contamination_bin_index": rate_index,
                            "contamination_bin": rate_label,
                            "specification": selected["specification"],
                            **_point(
                                test_humans,
                                machine_rows,
                                raw_scores,
                                clipped_scores,
                                raw_thresholds,
                                clipped_thresholds,
                            ),
                        }
                        contamination_rate_results.append(result)
                        method_rate_rows[(method, budget)].append(result)

            for selector in ORACLE_SELECTORS:
                method = method_id("rate_specific_oracle", selector)
                for bin_index in RATE_ADAPTIVE_BIN_INDICES:
                    selected = _select(
                        diagnostics_by_fit[("rate_specific_oracle", selector, bin_index)],
                        budget,
                    )
                    candidate_index = int(selected["candidate_index"])
                    clipped_scores, clipped_thresholds = evaluated(candidate_index)
                    selected_specs.append(
                        {
                            **selected,
                            "clean_loss_budget": budget,
                            "direction": direction,
                            "quantile_grid": json.dumps(list(quantiles)),
                        }
                    )
                    _, bin_label = contamination_bin(
                        RATE_ADAPTIVE_CUTPOINTS[bin_index - 1],
                        RATE_ADAPTIVE_CUTPOINTS,
                    )
                    clean = {
                        "detector": detector,
                        "method_id": method,
                        "scope": "rate_specific_oracle",
                        "selector": selector,
                        "clean_loss_budget": budget,
                        "contamination_bin_index": bin_index,
                        "contamination_bin": bin_label,
                        "specification": selected["specification"],
                        **_point(
                            test_humans,
                            test_none,
                            raw_scores,
                            clipped_scores,
                            raw_thresholds,
                            clipped_thresholds,
                        ),
                    }
                    clean_cost_results.append(clean)
                    method_clean_rows[(method, budget)].append(clean)
                    bin_test_rows = [
                        row
                        for row in test_attacked
                        if contamination_bin(rho(row), RATE_ADAPTIVE_CUTPOINTS)[0]
                        == bin_index
                    ]
                    rate_result = {
                        "detector": detector,
                        "method_id": method,
                        "scope": "rate_specific_oracle",
                        "selector": selector,
                        "clean_loss_budget": budget,
                        "contamination_bin_index": bin_index,
                        "contamination_bin": bin_label,
                        "specification": selected["specification"],
                        **_point(
                            test_humans,
                            bin_test_rows,
                            raw_scores,
                            clipped_scores,
                            raw_thresholds,
                            clipped_thresholds,
                        ),
                    }
                    contamination_rate_results.append(rate_result)
                    method_rate_rows[(method, budget)].append(rate_result)
                    for attack in attacks:
                        attack_rows = [
                            row for row in bin_test_rows if condition(row) == attack
                        ]
                        if not attack_rows:
                            continue
                        attack_result = {
                            "detector": detector,
                            "method_id": method,
                            "scope": "rate_specific_oracle",
                            "selector": selector,
                            "clean_loss_budget": budget,
                            "contamination_bin_index": bin_index,
                            "contamination_bin": bin_label,
                            "condition": attack,
                            "specification": selected["specification"],
                            **_point(
                                test_humans,
                                attack_rows,
                                raw_scores,
                                clipped_scores,
                                raw_thresholds,
                                clipped_thresholds,
                            ),
                        }
                        oracle_attack_rate_results.append(attack_result)
                        method_attack_rows[(method, budget)].append(attack_result)

            methods = [
                method_id(scope, selector)
                for scope in UNIVERSAL_SCOPES
                for selector in UNIVERSAL_SELECTORS
            ] + [
                method_id("rate_specific_oracle", selector)
                for selector in ORACLE_SELECTORS
            ]
            for method in methods:
                rate_rows = method_rate_rows[(method, budget)]
                attack_rows = method_attack_rows[(method, budget)]
                clean_rows = method_clean_rows[(method, budget)]
                eligible_rates = [
                    row
                    for row in rate_rows
                    if int(row["contamination_bin_index"]) in RATE_ADAPTIVE_BIN_INDICES
                ]
                dense = [
                    row for row in rate_rows if int(row["contamination_bin_index"]) == 5
                ]
                leaderboard.append(
                    {
                        "detector": detector,
                        "method_id": method,
                        "clean_loss_budget": budget,
                        "mean_clean_tpr_difference": float(
                            np.mean([row["paired_tpr_difference"] for row in clean_rows])
                        ),
                        "mean_clean_auroc_difference": float(
                            np.mean([row["paired_auroc_difference"] for row in clean_rows])
                        ),
                        "mean_eligible_rate_tpr_difference": float(
                            np.mean([row["paired_tpr_difference"] for row in eligible_rates])
                        ),
                        "worst_eligible_rate_tpr_difference": float(
                            min(row["paired_tpr_difference"] for row in eligible_rates)
                        ),
                        "mean_attack_cell_tpr_difference": float(
                            np.mean([row["paired_tpr_difference"] for row in attack_rows])
                        ),
                        "worst_attack_cell_tpr_difference": float(
                            min(row["paired_tpr_difference"] for row in attack_rows)
                        ),
                        "dense_rate_tpr_difference": (
                            None
                            if not dense
                            else float(np.mean([row["paired_tpr_difference"] for row in dense]))
                        ),
                    }
                )

    methods = sorted({row["method_id"] for row in selected_specs})
    if len(methods) != 11:
        raise AssertionError(f"expected 11 unique tuning methods, found {len(methods)}")
    return {
        "manifest": {
            "analysis": "raid_exploratory_tuning_comparison_v1",
            "result_label": "PILOT_DEBUG_NOT_FOR_FINAL_REPORTING",
            "target_fpr": TARGET_FPR,
            "quantile_grid": list(quantiles),
            "clean_auroc_loss_budgets": list(clean_loss_budgets),
            "crossfit_folds": crossfit_folds,
            "crossfit_seed": crossfit_seed,
            "eligible_interval": "0<rho<=0.5",
            "methods": methods,
            "method_count": len(methods),
            "reported_approach_count_including_raw": len(methods) + 1,
            "bootstrap_repetitions": 0,
            "development_source_count": len({source(row) for row in rows}),
            "future_full_run_requirement": (
                "exclude every development_source_id before final splitting"
            ),
            "validation": validation,
        },
        "development_sources": {
            "purpose": "pilot selector and clean-loss-budget model selection",
            "must_be_excluded_from_future_full_benchmark": True,
            "source_count": len({source(row) for row in rows}),
            "source_ids": sorted({source(row) for row in rows}),
        },
        "selected_specs": selected_specs,
        "candidate_diagnostics": candidate_diagnostics,
        "universal_attack_results": universal_attack_results,
        "contamination_rate_results": contamination_rate_results,
        "oracle_attack_rate_results": oracle_attack_rate_results,
        "clean_cost_results": clean_cost_results,
        "leaderboard": leaderboard,
    }


def write_comparison_artifacts(result: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": root / "comparison_manifest.json",
        "development_sources": root / "development_source_ids.json",
        "selected_specs": root / "selected_specs.csv",
        "candidate_diagnostics": root / "candidate_diagnostics.csv",
        "universal_attack_results": root / "universal_attack_results.csv",
        "contamination_rate_results": root / "contamination_rate_results.csv",
        "oracle_attack_rate_results": root / "oracle_attack_rate_results.csv",
        "clean_cost_results": root / "clean_cost_results.csv",
        "leaderboard": root / "leaderboard.csv",
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
    parser.add_argument("--output-name", default="tuning_comparison_v1")
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--crossfit-seed", type=int, default=260826)
    parser.add_argument(
        "--allow-full-run",
        action="store_true",
        help="Permit loading an unbounded full run. Disabled by default for memory safety.",
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
            "refusing to load an unbounded full run; this exploratory command is for "
            "the bounded pilot (use --allow-full-run only after a memory review)"
        )
    falcon_path = run_dir / "falcon_scores.jsonl"
    binoculars_path = run_dir / "binoculars_scores.jsonl"
    if not falcon_path.is_file() or not binoculars_path.is_file():
        raise FileNotFoundError("completed Falcon and Binoculars score packs are required")
    result = compare_tuning_methods(
        list(iter_jsonl(falcon_path)),
        list(iter_jsonl(binoculars_path)),
        expected_attacks=config["dataset"]["required_attacks"],
        crossfit_folds=args.crossfit_folds,
        crossfit_seed=args.crossfit_seed,
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
    paths = write_comparison_artifacts(result, output_dir)
    print(f"raid_tuning_comparison_status=complete")
    print(f"raid_tuning_comparison_dir={output_dir}")
    print(f"raid_tuning_method_count={result['manifest']['method_count']}")
    print(f"raid_tuning_leaderboard={paths['leaderboard']}")


if __name__ == "__main__":
    main()
