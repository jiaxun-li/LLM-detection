#!/usr/bin/env python3
"""Plot every completed paper cell and create a small downloadable ZIP.

The script reads only completed runs, keeps the newest completed run for each
dataset/model cell, skips debug outputs, and never loads the large JSONL score
packs. It creates an overview plus one compact figure per detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import plot_tpr_contamination as plotter  # noqa: E402


@dataclass(frozen=True)
class Cell:
    run_id: str
    dataset: str
    model: str
    metrics_path: Path
    result_dir: Path
    manifest_path: Path
    modified_ns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/plot_exports"),
        help="Directory that receives timestamped exports and ZIP archives",
    )
    parser.add_argument("--analysis", default="primary_frozen_mixture")
    parser.add_argument(
        "--detectors",
        help="Optional comma-separated detector subset; default: all available",
    )
    parser.add_argument("--no-ci", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args()


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def configured_cells(config_path: Path) -> set[tuple[str, str]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    models = {
        str(model)
        for group, values in config["models"].items()
        if group not in {"revision", "tokenizer_revision", "binoculars"}
        and isinstance(values, list)
        for model in values
    }
    return {
        (str(dataset), model)
        for dataset in config["datasets"]
        for model in models
    }


def _representative_row(metrics_path: Path, analysis: str) -> dict[str, str] | None:
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == "test" and row.get("analysis") == analysis:
                return row
    return None


def discover_completed_cells(
    runs_dir: Path,
    results_dir: Path,
    allowed_cells: set[tuple[str, str]],
    analysis: str,
) -> tuple[list[Cell], list[dict[str, str]]]:
    selected: dict[tuple[str, str], Cell] = {}
    notes: list[dict[str, str]] = []
    for metrics_path in sorted(results_dir.glob("*/metrics.csv")):
        row = _representative_row(metrics_path, analysis)
        if row is None:
            notes.append({"path": str(metrics_path), "status": "wrong-analysis"})
            continue
        if _truthy(row.get("debug_only", "false")):
            notes.append({"path": str(metrics_path), "status": "debug-only"})
            continue
        dataset = row.get("dataset", "")
        model = row.get("model", "")
        if (dataset, model) not in allowed_cells:
            notes.append({"path": str(metrics_path), "status": "outside-paper-matrix"})
            continue
        run_id = row.get("run_id") or metrics_path.parent.name
        manifest_path = runs_dir / run_id / "manifest.json"
        if not manifest_path.exists():
            notes.append({"run_id": run_id, "status": "missing-manifest"})
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = set(manifest.get("completed_stages", []))
        if manifest.get("completion_status") != "complete" or not {
            "prepare",
            "score",
            "evaluate",
        }.issubset(completed):
            notes.append({"run_id": run_id, "status": "incomplete-manifest"})
            continue
        candidate = Cell(
            run_id=run_id,
            dataset=dataset,
            model=model,
            metrics_path=metrics_path,
            result_dir=metrics_path.parent,
            manifest_path=manifest_path,
            modified_ns=metrics_path.stat().st_mtime_ns,
        )
        key = (dataset, model)
        previous = selected.get(key)
        if previous is None or candidate.modified_ns > previous.modified_ns:
            if previous is not None:
                notes.append({"run_id": previous.run_id, "status": "superseded"})
            selected[key] = candidate
        else:
            notes.append({"run_id": candidate.run_id, "status": "superseded"})
    return sorted(selected.values(), key=lambda item: (item.dataset, item.model)), notes


def export_cell(
    cell: Cell,
    destination: Path,
    analysis: str,
    requested_detectors: str | None,
    show_ci: bool,
) -> list[str]:
    rows = plotter.read_metrics(cell.metrics_path, analysis)
    detectors = plotter.selected_detectors(rows, requested_detectors)
    cell_dir = destination / (
        f"{_slug(cell.dataset)}--{_slug(cell.model)}--{_slug(cell.run_id)}"
    )
    cell_dir.mkdir(parents=True, exist_ok=True)
    plotter.plot(rows, detectors, cell_dir / "all_detectors.png", show_ci)
    for detector in detectors:
        plotter.plot(rows, [detector], cell_dir / f"{_slug(detector)}.png", show_ci)
    shutil.copy2(cell.metrics_path, cell_dir / "metrics.csv")
    shutil.copy2(cell.manifest_path, cell_dir / "manifest.json")
    for artifact in sorted(cell.result_dir.glob("*.json")):
        if artifact.name != "manifest.json":
            shutil.copy2(artifact, cell_dir / artifact.name)
    return detectors


def main() -> None:
    args = parse_args()
    allowed = configured_cells(args.config)
    cells, notes = discover_completed_cells(
        args.runs, args.results, allowed, args.analysis
    )
    if not cells:
        raise SystemExit("No completed non-debug paper cells were found.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    export_dir = args.output_root / f"completed-study-plots-{stamp}"
    export_dir.mkdir(parents=True, exist_ok=False)
    exported: list[dict[str, Any]] = []
    for cell in cells:
        detectors = export_cell(
            cell,
            export_dir,
            args.analysis,
            args.detectors,
            not args.no_ci,
        )
        exported.append(
            {
                **{
                    key: value
                    for key, value in asdict(cell).items()
                    if key != "modified_ns"
                },
                "metrics_path": str(cell.metrics_path.resolve()),
                "result_dir": str(cell.result_dir.resolve()),
                "manifest_path": str(cell.manifest_path.resolve()),
                "detectors": detectors,
            }
        )

    expected = sorted({f"{dataset}::{model}" for dataset, model in allowed})
    found = {f"{cell.dataset}::{cell.model}" for cell in cells}
    summary = {
        "created_utc": stamp,
        "analysis": args.analysis,
        "completed_cells_exported": len(cells),
        "expected_paper_cells": len(allowed),
        "exported": exported,
        "missing_cells": [value for value in expected if value not in found],
        "skipped_outputs": notes,
    }
    (export_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    archive: Path | None = None
    if not args.no_archive:
        archive = Path(
            shutil.make_archive(
                str(export_dir),
                "zip",
                root_dir=export_dir.parent,
                base_dir=export_dir.name,
            )
        )
    print(f"completed_cells_exported={len(cells)}")
    print(f"export_dir={export_dir.resolve()}")
    print(f"archive={archive.resolve() if archive else 'disabled'}")
    print(f"missing_cells={len(summary['missing_cells'])}")


if __name__ == "__main__":
    main()
