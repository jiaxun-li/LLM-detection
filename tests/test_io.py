from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment_core.infrastructure.io import (
    AppendSafeJsonlWriter,
    append_jsonl,
    completed_keys,
    iter_jsonl,
    repair_partial_jsonl,
    validate_unique_jsonl,
    write_completion_marker,
)


class JsonlRestartTests(unittest.TestCase):
    def test_partial_last_line_is_ignored_then_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            append_jsonl(path, {"id": 1})
            with path.open("ab") as handle:
                handle.write(b'{"id": 2')
            self.assertEqual(list(iter_jsonl(path)), [{"id": 1}])
            self.assertTrue(repair_partial_jsonl(path))
            append_jsonl(path, {"id": 2})
            self.assertEqual(validate_unique_jsonl(path, lambda row: (row["id"],)), 2)

    def test_duplicate_completed_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            append_jsonl(path, {"id": 1})
            append_jsonl(path, {"id": 1})
            with self.assertRaisesRegex(ValueError, "duplicate"):
                completed_keys(path, lambda row: (row["id"],))

    def test_completion_marker_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "complete.json"
            write_completion_marker(path, {"rows": 2})
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "complete")
            self.assertEqual(value["validation"]["rows"], 2)

    def test_buffered_appender_checkpoints_complete_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "buffered.jsonl"
            with AppendSafeJsonlWriter(path, checkpoint_interval=2) as writer:
                writer.write({"id": 1})
                writer.write({"id": 2})
                writer.write({"id": 3})
            self.assertEqual(
                [row["id"] for row in iter_jsonl(path)],
                [1, 2, 3],
            )


if __name__ == "__main__":
    unittest.main()
