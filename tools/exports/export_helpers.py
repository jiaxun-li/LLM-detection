"""Shared table selection helpers for primary result exports."""

from __future__ import annotations

from typing import Iterable, Sequence

CLEAN_FIELDS = (
    "run_id", "dataset", "model", "detector", "aggregation", "analysis",
    "direction", "clipping_specification", "target_fpr",
    "calibration_threshold", "actual_fpr", "actual_fpr_ci_low",
    "actual_fpr_ci_high", "tpr", "tpr_ci_low", "tpr_ci_high", "auroc",
    "auroc_ci_low", "auroc_ci_high", "partial_auroc_0_5_fpr",
    "partial_auroc_ci_low", "partial_auroc_ci_high", "n_calibration_human",
    "n_test_human", "n_test_llm", "n_unique_source_sample_ids",
    "bootstrap_repetitions", "bootstrap_seed",
)
ROBUSTNESS_FIELDS = (
    "run_id", "dataset", "model", "detector", "contamination_mode",
    "aggregation", "analysis", "target_fpr", "direction",
    "clipping_specification", "robustness_auc_tpr_vs_contamination",
    "robustness_auc_ci_low", "robustness_auc_ci_high",
    "n_unique_source_sample_ids", "bootstrap_repetitions", "bootstrap_seed",
)
CLIPPING_FIELDS = (
    "run_id", "dataset", "model", "detector", "contamination_mode",
    "requested_contamination_ratio", "realized_contamination_ratio_mean",
    "target_fpr", "direction", "clipping_specification",
    "clipped_minus_raw_tpr", "clipped_minus_raw_tpr_ci_low",
    "clipped_minus_raw_tpr_ci_high", "clipped_minus_raw_auroc",
    "clipped_minus_raw_auroc_ci_low", "clipped_minus_raw_auroc_ci_high",
    "clipped_minus_raw_partial_auroc",
    "clipped_minus_raw_partial_auroc_ci_low",
    "clipped_minus_raw_partial_auroc_ci_high",
    "n_unique_source_sample_ids", "bootstrap_repetitions", "bootstrap_seed",
)


def select_fields(
    rows: Iterable[dict[str, str]], fields: Sequence[str]
) -> list[dict[str, str]]:
    return [{field: row.get(field, "") for field in fields} for row in rows]


def clean_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    selected: dict[tuple[str, ...], dict[str, str]] = {}
    ordered = sorted(
        rows,
        key=lambda row: (row.get("contamination_mode") != "random",),
    )
    for row in ordered:
        if abs(float(row["requested_contamination_ratio"])) > 1e-12:
            continue
        key = tuple(
            row.get(field, "")
            for field in (
                "run_id", "dataset", "model", "detector", "aggregation",
                "analysis", "target_fpr",
            )
        )
        selected.setdefault(key, row)
    return list(selected.values())


def robustness_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    selected: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(
            row.get(field, "")
            for field in (
                "run_id", "dataset", "model", "detector",
                "contamination_mode", "aggregation", "analysis", "target_fpr",
            )
        )
        selected.setdefault(key, row)
    return list(selected.values())


def clipping_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    selected: dict[tuple[str, ...], dict[str, str]] = {}
    ordered = sorted(
        rows,
        key=lambda row: (row.get("contamination_mode") != "random",),
    )
    for row in ordered:
        if row.get("aggregation") != "clipped":
            continue
        ratio = float(row["requested_contamination_ratio"])
        key_mode = "clean" if abs(ratio) <= 1e-12 else row["contamination_mode"]
        key = tuple(
            row.get(field, "")
            for field in ("run_id", "dataset", "model", "detector", "target_fpr")
        ) + (key_mode, row["requested_contamination_ratio"])
        selected.setdefault(key, row)
    return list(selected.values())



