"""Streaming, restart-safe JSONL and completion marker helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


def iter_jsonl(path: str | Path, tolerate_partial_last_line: bool = True) -> Iterator[dict[str, Any]]:
    """Stream valid JSONL rows.

    A malformed final non-newline-terminated record is treated as an interrupted
    append and ignored. Malformed complete lines remain hard errors.
    """
    source = Path(path)
    with source.open("rb") as handle:
        line_number = 0
        while True:
            raw = handle.readline()
            if not raw:
                break
            line_number += 1
            if not raw.strip():
                continue
            try:
                yield json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                at_eof = handle.peek(1) == b""
                if tolerate_partial_last_line and at_eof and not raw.endswith(b"\n"):
                    break
                raise ValueError(f"invalid JSONL record at {source}:{line_number}")


def repair_partial_jsonl(path: str | Path) -> bool:
    """Truncate only an invalid partial final line; return whether repair occurred."""
    target = Path(path)
    if not target.exists() or target.stat().st_size == 0:
        return False
    with target.open("rb+") as handle:
        valid_end = 0
        while True:
            start = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                valid_end = handle.tell()
                continue
            try:
                json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if not raw.endswith(b"\n") and handle.read(1) == b"":
                    handle.truncate(start)
                    handle.flush()
                    os.fsync(handle.fileno())
                    return True
                raise ValueError(f"invalid complete JSONL line in {target}")
            valid_end = handle.tell()
        handle.truncate(valid_end)
    return False


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one complete record and fsync it before returning."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    with target.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


class AppendSafeJsonlWriter:
    """Buffered JSONL appender with periodic durable checkpoints.

    A crash can lose only rows since the last fsync. A torn final record is
    repaired on resume and completed row keys prevent duplicates.
    """

    def __init__(self, path: str | Path, checkpoint_interval: int = 100) -> None:
        self.path = Path(path)
        self.checkpoint_interval = max(int(checkpoint_interval), 1)
        self.handle: Any = None
        self.since_checkpoint = 0

    def __enter__(self) -> "AppendSafeJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        repair_partial_jsonl(self.path)
        self.handle = self.path.open("ab")
        return self

    def write(self, row: dict[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("JSONL writer is not open")
        payload = (
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        self.handle.write(payload)
        self.since_checkpoint += 1
        if self.since_checkpoint >= self.checkpoint_interval:
            self.checkpoint()

    def checkpoint(self) -> None:
        if self.handle is None or self.since_checkpoint == 0:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.since_checkpoint = 0

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self.checkpoint()
        finally:
            if self.handle is not None:
                self.handle.close()
                self.handle = None


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def completed_keys(
    path: str | Path,
    key_fn: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> set[tuple[Any, ...]]:
    target = Path(path)
    if not target.exists():
        return set()
    repair_partial_jsonl(target)
    keys: set[tuple[Any, ...]] = set()
    for row in iter_jsonl(target):
        key = key_fn(row)
        if key in keys:
            raise ValueError(f"duplicate completed row key {key!r} in {target}")
        keys.add(key)
    return keys


def validate_unique_jsonl(
    path: str | Path,
    key_fn: Callable[[dict[str, Any]], tuple[Any, ...]],
    expected_count: int | None = None,
) -> int:
    keys = completed_keys(path, key_fn)
    if expected_count is not None and len(keys) != expected_count:
        raise ValueError(f"{path} has {len(keys)} unique rows; expected {expected_count}")
    return len(keys)


def write_completion_marker(path: str | Path, validated: dict[str, Any]) -> None:
    """Write a marker only after the caller's validation has succeeded."""
    atomic_write_json(path, {"status": "complete", "validation": validated})


def stream_copy(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    count = 0
    for row in rows:
        append_jsonl(path, row)
        count += 1
    return count
