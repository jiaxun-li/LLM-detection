#!/usr/bin/env python3
"""Export compact plots and table-ready statistics for primary cells and Beemo.

The exporter reads only manifests, result CSV/JSON files, and saved metrics. It
never reads or copies prepared-data or token-feature JSONL files.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from Beemo.beemo_plot import plot_beemo  # noqa: E402
from scripts.export_completed_plots import (  # noqa: E402
    Cell,
    discover_completed_cells,
    export_cell,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/plot_exports"),
    )
    parser.add_argument("--analysis", default="primary_frozen_mixture")
    parser.add_argument("--no-ci", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args()


def primary_cells(config_path: Path) -> set[tuple[str, str]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        (str(dataset), str(model))
        for dataset in config["datasets"]
        for model in config["models"]["primary"]
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty summary: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def discover_beemo_run(runs: Path, results: Path) -> tuple[str, Path, Path]:
    candidates: list[tuple[int, str, Path, Path]] = []
    result_root = results / "beemo"
    run_root = runs / "beemo"
    for result_dir in result_root.iterdir() if result_root.is_dir() else ():
        if not result_dir.is_dir() or not (result_dir / "metrics.csv").is_file():
            continue
        run_dir = run_root / result_dir.name
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stages = set(manifest.get("completed_stages", []))
        if (
            manifest.get("completion_status") == "complete"
            and not bool(manifest.get("debug_only", False))
            and {"prepare", "score", "evaluate", "plot", "validate"}.issubset(stages)
        ):
            candidates.append(
                ((result_dir / "metrics.csv").stat().st_mtime_ns,
                 result_dir.name, run_dir, result_dir)
            )
    if not candidates:
        raise ValueError("no completed non-debug Beemo run was found")
    _, run_id, run_dir, result_dir = max(candidates)
    return run_id, run_dir, result_dir


def copy_beemo(run_dir: Path, result_dir: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(result_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
            shutil.copy2(path, destination / path.name)
            copied.append(path.name)
    shutil.copy2(run_dir / "manifest.json", destination / "manifest.json")
    copied.append("manifest.json")
    plots = plot_beemo(destination / "metrics.csv", destination)
    copied.extend(Path(path).name for path in plots)
    return sorted(set(copied))


def inventory_row(cell: Cell) -> dict[str, Any]:
    manifest = json.loads(cell.manifest_path.read_text(encoding="utf-8"))
    return {
        "run_id": cell.run_id,
        "dataset": cell.dataset,
        "model": cell.model,
        "git_commit": manifest.get("git_commit"),
        "dataset_resolved_revision": manifest.get("dataset", {}).get("resolved_revision"),
        "target_model_resolved_revision": manifest.get("target_model_resolved_revision"),
        "target_tokenizer_resolved_revision": manifest.get("target_tokenizer_resolved_revision"),
        "completion_status": manifest.get("completion_status"),
        "completed_stages": json.dumps(manifest.get("completed_stages", [])),
    }


def main() -> None:
    args = parse_args()
    expected = primary_cells(args.config)
    cells, notes = discover_completed_cells(
        args.runs, args.results, expected, args.analysis
    )
    found = {(cell.dataset, cell.model) for cell in cells}
    missing = sorted(expected - found)
    if missing:
        raise SystemExit(f"missing completed primary cells: {missing}")
    if len(cells) != 9:
        raise SystemExit(f"expected 9 primary cells, found {len(cells)}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    export_dir = args.output_root / f"primary-beemo-summary-{stamp}"
    primary_dir = export_dir / "primary_cells"
    export_dir.mkdir(parents=True, exist_ok=False)
    primary_dir.mkdir()

    all_metrics: list[dict[str, str]] = []
    for cell in cells:
        export_cell(
            cell, primary_dir, args.analysis, None, not args.no_ci
        )
        cell_rows = read_csv(cell.metrics_path)
        all_metrics.extend(cell_rows)

    write_csv(export_dir / "primary_all_metrics.csv", all_metrics)
    write_csv(
        export_dir / "primary_clean_metrics.csv",
        select_fields(clean_rows(all_metrics), CLEAN_FIELDS),
    )
    write_csv(
        export_dir / "primary_robustness_auc.csv",
        select_fields(robustness_rows(all_metrics), ROBUSTNESS_FIELDS),
    )
    write_csv(
        export_dir / "primary_clipping_effects.csv",
        select_fields(clipping_rows(all_metrics), CLIPPING_FIELDS),
    )
    write_csv(
        export_dir / "primary_cell_inventory.csv",
        [inventory_row(cell) for cell in cells],
    )

    beemo_run_id, beemo_run_dir, beemo_result_dir = discover_beemo_run(
        args.runs, args.results
    )
    beemo_files = copy_beemo(
        beemo_run_dir, beemo_result_dir, export_dir / "beemo"
    )
    shutil.copy2(args.config, export_dir / "primary_protocol_config.json")
    shutil.copy2(Path("Beemo/config.json"), export_dir / "beemo_protocol_config.json")

    summary = {
        "created_utc": stamp,
        "primary_cell_count": len(cells),
        "primary_metric_rows": len(all_metrics),
        "primary_cells": [
            {
                "run_id": cell.run_id,
                "dataset": cell.dataset,
                "model": cell.model,
            }
            for cell in cells
        ],
        "beemo_run_id": beemo_run_id,
        "beemo_files": beemo_files,
        "skipped_primary_outputs": notes,
        "large_jsonl_files_included": False,
    }
    (export_dir / "bundle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (export_dir / "README.md").write_text(
        "# Primary + Beemo compact export\n\n"
        "This bundle contains no prepared-text or token-feature JSONL files.\n\n"
        "- `primary_all_metrics.csv`: every released primary metric and confidence interval.\n"
        "- `primary_clean_metrics.csv`: deduplicated clean-text results for tables.\n"
        "- `primary_robustness_auc.csv`: one robustness-curve summary per cell/method.\n"
        "- `primary_clipping_effects.csv`: paired clipped-minus-raw effects by condition.\n"
        "- `primary_cell_inventory.csv`: run IDs and resolved revisions.\n"
        "- `primary_cells/`: per-cell metrics, manifests, and PNG plots.\n"
        "- `beemo/`: metrics, subgroup statistics, frozen specifications, validation, and plots.\n",
        encoding="utf-8",
    )

    archive: Path | None = None
    if not args.no_archive:
        archive = Path(
            shutil.make_archive(
                str(export_dir), "zip", root_dir=export_dir.parent,
                base_dir=export_dir.name,
            )
        )
    print(f"primary_cells_exported={len(cells)}")
    print(f"primary_metric_rows={len(all_metrics)}")
    print(f"beemo_run_id={beemo_run_id}")
    print(f"export_dir={export_dir.resolve()}")
    print(f"archive={archive.resolve() if archive else 'disabled'}")


if __name__ == "__main__":
    main()
