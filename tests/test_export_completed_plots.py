from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_completed_plots import discover_completed_cells


FIELDS = [
    "run_id",
    "debug_only",
    "dataset",
    "model",
    "split",
    "analysis",
]


class ExportCompletedPlotsTests(unittest.TestCase):
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
