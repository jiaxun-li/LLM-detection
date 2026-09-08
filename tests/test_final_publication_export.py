from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from RAID.raid_plot import _labels
from scripts.export_final_publication_bundle import (
    EXPECTED_PRIMARY_ROWS,
    PRIMARY_DETECTORS,
    RAID_SOURCE_DETECTORS,
    filtered_csv,
    verify_primary_bundle,
    verify_raid_results,
)
from llm_detection.revision_artifacts import RAID_ARTIFACTS


class FinalPublicationExportTests(unittest.TestCase):
    def test_raid_labels_include_only_available_detectors(self) -> None:
        rows = [
            {"detector": "log_likelihood"},
            {"detector": "binocular_origin"},
        ]
        self.assertEqual(
            _labels(rows),
            {
                "log_likelihood": "Log likelihood",
                "binocular_origin": "Binocular-origin",
            },
        )

    def test_filtered_csv_removes_only_binocular_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            destination = root / "destination.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["detector", "value"])
                writer.writeheader()
                for detector in (*PRIMARY_DETECTORS, "binocular_gap"):
                    writer.writerow({"detector": detector, "value": "1"})
            count, detectors = filtered_csv(source, destination)
            with destination.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(count, len(PRIMARY_DETECTORS))
        self.assertEqual(detectors, set(PRIMARY_DETECTORS))
        self.assertEqual({row["detector"] for row in rows}, set(PRIMARY_DETECTORS))

    def test_primary_bundle_validation_checks_recorded_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "primary_metrics.csv"
            fields = ["detector"]
            with metrics.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index in range(EXPECTED_PRIMARY_ROWS):
                    writer.writerow(
                        {"detector": PRIMARY_DETECTORS[index % len(PRIMARY_DETECTORS)]}
                    )
            digest = hashlib.sha256(metrics.read_bytes()).hexdigest()
            manifest = {
                "status": "complete",
                "bundle_id": "test",
                "cell_count": 9,
                "combined_metric_rows": EXPECTED_PRIMARY_ROWS,
                "metric_rows_per_cell": 392,
                "reported_detectors": list(PRIMARY_DETECTORS),
                "binocular_gap_policy": "excluded",
                "artifact_hashes": {
                    "primary_metrics.csv": {
                        "size": metrics.stat().st_size,
                        "sha256": digest,
                    }
                },
            }
            manifest_path = root / "bundle_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            marker = {
                "status": "complete",
                "bundle_manifest": {
                    "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                },
            }
            (root / "bundle.complete.json").write_text(
                json.dumps(marker), encoding="utf-8"
            )
            loaded = verify_primary_bundle(root)
        self.assertEqual(loaded["bundle_id"], "test")

    def test_raid_validation_checks_every_recorded_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = {}
            for relative in RAID_ARTIFACTS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"artifact: {relative}\n", encoding="utf-8")
                artifacts[relative] = {
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            manifest = {
                "revision_id": "raid-test",
                "completion_status": "complete",
            }
            report = {
                "validation_status": "pass",
                "bootstrap_repetitions": 2000,
                "metrics_rows": 416,
                "detectors": list(RAID_SOURCE_DETECTORS),
                "artifacts": artifacts,
            }
            (root / "revision_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            for name in ("validation_report.json", "revision.complete.json"):
                (root / name).write_text(json.dumps(report), encoding="utf-8")
            loaded_manifest, loaded_report = verify_raid_results(root)
        self.assertEqual(loaded_manifest["revision_id"], "raid-test")
        self.assertEqual(loaded_report["bootstrap_repetitions"], 2000)


if __name__ == "__main__":
    unittest.main()
