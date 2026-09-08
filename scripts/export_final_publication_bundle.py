#!/usr/bin/env python3
"""Build one audited, downloadable bundle for the final primary and RAID results.

The export contains compact result artifacts only. It never copies prepared data,
model score packs, or other JSONL files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import plot_tpr_contamination as primary_plotter  # noqa: E402
from RAID.raid_plot import plot_raid  # noqa: E402
from llm_detection.revision_artifacts import RAID_ARTIFACTS, artifact_digest  # noqa: E402
from scripts.export_primary_beemo_bundle import (  # noqa: E402
    CLEAN_FIELDS,
    CLIPPING_FIELDS,
    ROBUSTNESS_FIELDS,
    clean_rows,
    clipping_rows,
    robustness_rows,
    select_fields,
)


PRIMARY_DETECTORS = (
    "log_likelihood",
    "rank",
    "log_rank",
    "lrr",
    "entropy",
    "entropy_gap",
    "binocular_origin",
)
RAID_SOURCE_DETECTORS = (*PRIMARY_DETECTORS[:-1], "binocular_gap", "binocular_origin")
EXPECTED_PRIMARY_CELLS = 9
EXPECTED_PRIMARY_ROWS_PER_CELL = 392
EXPECTED_PRIMARY_ROWS = EXPECTED_PRIMARY_CELLS * EXPECTED_PRIMARY_ROWS_PER_CELL
EXPECTED_RAID_ROWS = 416
EXPECTED_BOOTSTRAPS = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-bundle", type=Path, required=True)
    parser.add_argument("--raid-results", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/final_publication_exports"),
    )
    parser.add_argument("--bundle-id")
    parser.add_argument("--analysis", default="primary_frozen_mixture")
    parser.add_argument("--no-ci", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"inconsistent CSV columns while writing {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_primary_bundle(root: Path) -> dict[str, Any]:
    manifest_path = root / "bundle_manifest.json"
    marker_path = root / "bundle.complete.json"
    metrics_path = root / "primary_metrics.csv"
    for path in (manifest_path, marker_path, metrics_path):
        ensure(path.is_file(), f"primary bundle is missing {path}")

    manifest = read_json(manifest_path)
    marker = read_json(marker_path)
    ensure(manifest.get("status") == "complete", "primary manifest is not complete")
    ensure(marker.get("status") == "complete", "primary completion marker is not complete")
    ensure(manifest.get("cell_count") == EXPECTED_PRIMARY_CELLS, "primary cell count is not 9")
    ensure(manifest.get("combined_metric_rows") == EXPECTED_PRIMARY_ROWS, "primary combined row count is not 3528")
    ensure(manifest.get("metric_rows_per_cell") == EXPECTED_PRIMARY_ROWS_PER_CELL, "primary per-cell row count is not 392")
    ensure(tuple(manifest.get("reported_detectors", [])) == PRIMARY_DETECTORS, "primary detector contract changed")
    ensure(manifest.get("binocular_gap_policy") == "excluded", "primary bundle did not exclude Binocular-gap")
    ensure(
        marker.get("bundle_manifest", {}).get("sha256") == sha256(manifest_path),
        "primary bundle manifest hash disagrees with its completion marker",
    )

    expected_hashes = manifest.get("artifact_hashes", {})
    ensure(expected_hashes, "primary bundle has no artifact hash inventory")
    for relative, expected in expected_hashes.items():
        path = root / relative
        ensure(path.is_file(), f"primary artifact is missing: {path}")
        ensure(artifact_digest(path) == expected, f"primary artifact hash changed: {path}")

    rows = read_csv(metrics_path)
    ensure(len(rows) == EXPECTED_PRIMARY_ROWS, "primary_metrics.csv does not have 3528 rows")
    ensure({row["detector"] for row in rows} == set(PRIMARY_DETECTORS), "primary_metrics.csv detector set changed")
    return manifest


def copy_primary_cell(
    source: Path,
    destination: Path,
    analysis: str,
    show_ci: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    metrics_path = source / "metrics.csv"
    revision_path = source / "revision_manifest.json"
    report_path = source / "validation_report.json"
    for path in (metrics_path, revision_path, report_path, source / "revision.complete.json"):
        ensure(path.is_file(), f"primary cell is missing {path}")

    rows = read_csv(metrics_path)
    ensure(len(rows) == EXPECTED_PRIMARY_ROWS_PER_CELL, f"{source.name} does not have 392 rows")
    ensure({row["detector"] for row in rows} == set(PRIMARY_DETECTORS), f"{source.name} detector set changed")
    report = read_json(report_path)
    ensure(report.get("validation_status") == "pass", f"{source.name} validation did not pass")
    source_detectors = set(report.get("detectors", []))
    ensure(
        source_detectors == set(PRIMARY_DETECTORS)
        or source_detectors == set(RAID_SOURCE_DETECTORS),
        f"{source.name} validation detector set changed",
    )
    ensure(
        report.get("metric_rows") in {
            EXPECTED_PRIMARY_ROWS_PER_CELL,
            EXPECTED_PRIMARY_ROWS_PER_CELL + 56,
        },
        f"{source.name} source validation row count changed",
    )

    destination.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
            shutil.copy2(path, destination / path.name)

    plot_rows = primary_plotter.read_metrics(destination / "metrics.csv", analysis)
    detectors = list(PRIMARY_DETECTORS)
    primary_plotter.plot(plot_rows, detectors, destination / "all_detectors.png", show_ci)
    primary_plotter.plot(
        plot_rows,
        detectors,
        destination / "detectors_2x7.png",
        show_ci,
        layout="detector-columns",
    )
    for target_fpr in (0.01, 0.05):
        primary_plotter.plot(
            plot_rows,
            detectors,
            destination / f"detectors_row_fpr_{target_fpr * 100:.0f}pct.png",
            show_ci,
            layout="detector-columns",
            target_fprs=(target_fpr,),
        )
    for detector in detectors:
        primary_plotter.plot(
            plot_rows,
            [detector],
            destination / f"{slug(detector)}.png",
            show_ci,
        )

    revision = read_json(revision_path)
    first = rows[0]
    inventory = {
        "cell": source.name,
        "dataset": first["dataset"],
        "model": first["model"],
        "source_run_id": revision["source_run_id"],
        "source_revision_id": revision["revision_id"],
        "metric_rows": len(rows),
        "source_validation_metric_rows": report["metric_rows"],
        "bootstrap_repetitions": report["bootstrap_repetitions"],
        "validation_status": report["validation_status"],
        "reported_detectors": json.dumps(list(PRIMARY_DETECTORS)),
    }
    return rows, inventory


def build_primary(
    source: Path,
    destination: Path,
    config: Path,
    analysis: str,
    show_ci: bool,
) -> dict[str, Any]:
    manifest = verify_primary_bundle(source)
    destination.mkdir(parents=True, exist_ok=False)
    cell_root = destination / "primary_cells"
    cell_root.mkdir()

    all_rows: list[dict[str, str]] = []
    inventory: list[dict[str, Any]] = []
    for record in manifest["cells"]:
        source_cell = source / record["cell"]
        revision = read_json(source_cell / "revision_manifest.json")
        metrics = read_csv(source_cell / "metrics.csv")
        first = metrics[0]
        cell_name = "--".join(
            (
                slug(first["dataset"]),
                slug(first["model"]),
                slug(revision["source_run_id"]),
            )
        )
        rows, cell_inventory = copy_primary_cell(
            source_cell,
            cell_root / cell_name,
            analysis,
            show_ci,
        )
        all_rows.extend(rows)
        inventory.append(cell_inventory)

    ensure(len(all_rows) == EXPECTED_PRIMARY_ROWS, "primary export did not collect 3528 rows")
    write_csv(destination / "primary_all_metrics.csv", all_rows)
    write_csv(destination / "primary_clean_metrics.csv", select_fields(clean_rows(all_rows), CLEAN_FIELDS))
    write_csv(destination / "primary_robustness_auc.csv", select_fields(robustness_rows(all_rows), ROBUSTNESS_FIELDS))
    write_csv(destination / "primary_clipping_effects.csv", select_fields(clipping_rows(all_rows), CLIPPING_FIELDS))
    write_csv(destination / "primary_cell_inventory.csv", inventory)
    shutil.copy2(config, destination / "primary_protocol_config.json")
    shutil.copy2(source / "bundle_manifest.json", destination / "source_bundle_manifest.json")
    shutil.copy2(source / "bundle.complete.json", destination / "source_bundle.complete.json")

    summary = {
        "source_bundle_id": manifest["bundle_id"],
        "cell_count": len(inventory),
        "metric_rows": len(all_rows),
        "reported_detectors": list(PRIMARY_DETECTORS),
        "plots_per_cell": 11,
        "total_cell_plots": 11 * len(inventory),
        "large_jsonl_files_included": False,
    }
    (destination / "bundle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def verify_raid_results(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "revision_manifest.json"
    report_path = root / "validation_report.json"
    marker_path = root / "revision.complete.json"
    for path in (manifest_path, report_path, marker_path):
        ensure(path.is_file(), f"RAID result is missing {path}")
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    marker = read_json(marker_path)
    ensure(manifest.get("completion_status") == "complete", "RAID revision is not complete")
    ensure(report.get("validation_status") == "pass", "RAID validation did not pass")
    ensure(marker == report, "RAID completion marker disagrees with validation report")
    ensure(report.get("bootstrap_repetitions") == EXPECTED_BOOTSTRAPS, "RAID bootstrap count is not 2000")
    ensure(report.get("metrics_rows") == EXPECTED_RAID_ROWS, "RAID metrics row count is not 416")
    ensure(tuple(report.get("detectors", [])) == RAID_SOURCE_DETECTORS, "RAID eight-detector source contract changed")
    expected = report.get("artifacts", {})
    ensure(set(expected) == set(RAID_ARTIFACTS), "RAID artifact inventory changed")
    for relative in RAID_ARTIFACTS:
        path = root / relative
        ensure(path.is_file(), f"RAID artifact is missing: {path}")
        ensure(artifact_digest(path) == expected[relative], f"RAID artifact hash changed: {path}")
    return manifest, report


def filtered_csv(source: Path, destination: Path) -> tuple[int, set[str]]:
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "detector" in fields:
        rows = [row for row in rows if row["detector"] in PRIMARY_DETECTORS]
    if not rows:
        raise ValueError(f"filtering produced an empty RAID CSV: {source}")
    write_csv(destination, rows)
    detectors = {row["detector"] for row in rows} if "detector" in fields else set()
    return len(rows), detectors


def copy_result_tree_without_jsonl(source: Path, destination: Path) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name.endswith(".jsonl")}

    shutil.copytree(source, destination, ignore=ignore)
    ensure(not list(destination.rglob("*.jsonl")), "JSONL files entered the compact export")


def build_raid(source: Path, destination: Path) -> dict[str, Any]:
    manifest, report = verify_raid_results(source)
    destination.mkdir(parents=True, exist_ok=False)
    exact = destination / "source_eight_detector_result"
    paper = destination / "paper_seven_detector_result"
    copy_result_tree_without_jsonl(source, exact)
    paper.mkdir()

    csv_inventory: dict[str, Any] = {}
    for path in sorted(source.glob("*.csv")):
        count, detectors = filtered_csv(path, paper / path.name)
        csv_inventory[path.name] = {
            "rows": count,
            "detectors": sorted(detectors),
        }
    plot_paths = plot_raid(paper)
    ensure(len(plot_paths) == 3 and all(path.is_file() for path in plot_paths), "RAID paper plots were not created")

    paper_metrics = read_csv(paper / "metrics.csv")
    ensure({row["detector"] for row in paper_metrics} == set(PRIMARY_DETECTORS), "RAID paper metrics detector set changed")
    ensure(all(row["detector"] != "binocular_gap" for row in paper_metrics), "Binocular-gap entered RAID paper metrics")

    source_reference = {
        "revision_id": manifest.get("revision_id"),
        "detector_revision": report.get("detector_revision"),
        "bootstrap_repetitions": report["bootstrap_repetitions"],
        "source_detectors": list(RAID_SOURCE_DETECTORS),
        "reported_detectors": list(PRIMARY_DETECTORS),
        "binocular_gap_policy": "retained only in source_eight_detector_result",
        "paper_csv_inventory": csv_inventory,
        "paper_plots": [str(path.relative_to(paper)) for path in plot_paths],
        "large_jsonl_files_included": False,
    }
    (destination / "raid_export_summary.json").write_text(
        json.dumps(source_reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (paper / "README.md").write_text(
        "# RAID seven-detector paper result\n\n"
        "These CSVs and plots exclude the legacy Binocular-gap diagnostic and retain "
        "the corrected Binocular-origin detector. The exact validated eight-detector "
        "result is preserved in `../source_eight_detector_result/`.\n",
        encoding="utf-8",
    )
    return source_reference


def file_inventory(root: Path, excluded: Iterable[Path] = ()) -> dict[str, Any]:
    excluded_set = {path.resolve() for path in excluded}
    inventory: dict[str, Any] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in excluded_set:
            continue
        inventory[path.relative_to(root).as_posix()] = artifact_digest(path)
    return inventory


def main() -> None:
    args = parse_args()
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "final publication export requires matplotlib; activate the Delta "
            "delta-smoke environment before running it"
        ) from exc
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_id = args.bundle_id or f"primary-nine-plus-raid-final-{stamp}"
    final = args.output_root / bundle_id
    building = args.output_root / f"{bundle_id}.building"
    ensure(not final.exists(), f"final export already exists: {final}")
    ensure(not building.exists(), f"incomplete export already exists: {building}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    building.mkdir()

    primary_summary = build_primary(
        args.primary_bundle,
        building / "primary",
        args.config,
        args.analysis,
        not args.no_ci,
    )
    raid_summary = build_raid(args.raid_results, building / "raid")
    (building / "README.md").write_text(
        "# Final primary-nine-cell + RAID publication export\n\n"
        "This archive restores the per-cell primary plots present in the earlier "
        "download and adds the corrected final RAID result. It contains no prepared "
        "data or model-score JSONL files.\n\n"
        "- `primary/`: nine cells, summary CSVs, manifests, and 11 plots per cell.\n"
        "- `raid/paper_seven_detector_result/`: paper-facing RAID CSVs and plots.\n"
        "- `raid/source_eight_detector_result/`: exact validated RAID result, including "
        "the retained Binocular-gap diagnostic for provenance.\n",
        encoding="utf-8",
    )
    manifest_path = building / "bundle_manifest.json"
    marker_path = building / "bundle.complete.json"
    bundle_manifest = {
        "bundle_id": bundle_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "primary": primary_summary,
        "raid": raid_summary,
        "large_jsonl_files_included": False,
    }
    manifest_path.write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inventory = file_inventory(building, excluded=(marker_path,))
    marker = {
        "bundle_id": bundle_id,
        "status": "complete",
        "file_count": len(inventory),
        "artifact_hashes": inventory,
    }
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    building.rename(final)

    archive: Path | None = None
    checksum: Path | None = None
    if not args.no_archive:
        archive = Path(
            shutil.make_archive(
                str(final),
                "gztar",
                root_dir=final.parent,
                base_dir=final.name,
            )
        )
        checksum = Path(f"{archive}.sha256")
        checksum.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")

    print("FINAL PUBLICATION EXPORT: PASS")
    print(f"bundle_id={bundle_id}")
    print(f"files={marker['file_count']}")
    print(f"primary_cell_plots={primary_summary['total_cell_plots']}")
    print(f"export_dir={final.resolve()}")
    print(f"archive={archive.resolve() if archive else 'disabled'}")
    print(f"checksum={checksum.resolve() if checksum else 'disabled'}")


if __name__ == "__main__":
    main()
