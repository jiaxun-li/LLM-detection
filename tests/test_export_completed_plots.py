from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_completed_plots import discover_completed_cells
from scripts.export_primary_beemo_bundle import (
    clean_rows,
    clipping_rows,
    discover_beemo_run,
    primary_cells,
    robustness_rows,
)


FIELDS = [
    "run_id",
    "debug_only",
    "dataset",
    "model",
    "split",
    "analysis",
]


class ExportCompletedPlotsTests(unittest.TestCase):
    def test_primary_bundle_configuration_is_exactly_nine_cells(self) -> None:
        cells = primary_cells(Path("configs/paper.json"))
        self.assertEqual(len(cells), 9)

    def test_primary_summary_views_deduplicate_repeated_curve_values(self) -> None:
        rows = []
        for mode in ("random", "tail"):
            for ratio in ("0.0", "0.5"):
                for aggregation in ("raw", "clipped"):
                    rows.append(
                        {
                            "run_id": "run",
                            "dataset": "xsum",
                            "model": "model",
                            "detector": "rank",
                            "analysis": "primary_frozen_mixture",
                            "target_fpr": "0.01",
                            "contamination_mode": mode,
                            "requested_contamination_ratio": ratio,
                            "aggregation": aggregation,
                        }
                    )
        self.assertEqual(len(clean_rows(rows)), 2)
        self.assertEqual(len(robustness_rows(rows)), 4)
        clipped = clipping_rows(rows)
        self.assertEqual(len(clipped), 3)
        self.assertEqual(
            sum(row["requested_contamination_ratio"] == "0.0" for row in clipped),
            1,
        )

    def test_beemo_discovery_selects_completed_non_debug_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for run_id, complete, debug in (
                ("failed", False, False),
                ("debug", True, True),
                ("released", True, False),
            ):
                run = root / "runs" / "beemo" / run_id
                result = root / "results" / "beemo" / run_id
                run.mkdir(parents=True)
                result.mkdir(parents=True)
                (result / "metrics.csv").write_text("metric\n1\n", encoding="utf-8")
                (run / "manifest.json").write_text(
                    json.dumps(
                        {
                            "completion_status": "complete" if complete else "failed",
                            "completed_stages": (
                                ["prepare", "score", "evaluate", "plot", "validate"]
                                if complete
                                else ["prepare"]
                            ),
                            "debug_only": debug,
                        }
                    ),
                    encoding="utf-8",
                )
            run_id, _, _ = discover_beemo_run(root / "runs", root / "results")
        self.assertEqual(run_id, "released")

    def _write_run(
        self,
        root: Path,
        run_id: str,
        *,
        debug: bool = False,
        complete: bool = True,
    ) -> None:
        run_dir = root / "runs" / run_id
        result_dir = root / "results" / run_id
        run_dir.mkdir(parents=True)
        result_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "completion_status": "complete" if complete else "failed",
                    "completed_stages": (
                        ["prepare", "score", "evaluate"] if complete else ["prepare"]
                    ),
                }
            ),
            encoding="utf-8",
        )
        with (result_dir / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "run_id": run_id,
                    "debug_only": debug,
                    "dataset": "xsum",
                    "model": "model/example",
                    "split": "test",
                    "analysis": "primary_frozen_mixture",
                }
            )

    def test_discovers_only_complete_non_debug_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "complete")
            self._write_run(root, "debug", debug=True)
            self._write_run(root, "failed", complete=False)
            cells, notes = discover_completed_cells(
                root / "runs",
                root / "results",
                {("xsum", "model/example")},
                "primary_frozen_mixture",
            )
        self.assertEqual([cell.run_id for cell in cells], ["complete"])
        self.assertEqual(
            {note["status"] for note in notes},
            {"debug-only", "incomplete-manifest"},
        )


if __name__ == "__main__":
    unittest.main()
