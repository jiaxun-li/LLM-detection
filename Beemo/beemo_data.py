"""Download, validate, split, and normalize the Beemo benchmark."""

from __future__ import annotations

import ast
import difflib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from llm_detection.data import data_row_key, stable_int
from llm_detection.io import AppendSafeJsonlWriter, iter_jsonl


VARIANTS = (
    "human",
    "original",
    "expert",
    "llama_p1",
    "llama_p2",
    "llama_p3",
    "gpt_p1",
    "gpt_p2",
    "gpt_p3",
)


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Beemo field {field!r} must be a non-empty string")
    return value.strip()


def parse_edit_versions(value: Any, field: str) -> dict[str, str]:
    """Accept both the released string encoding and already-decoded objects."""
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"cannot parse Beemo edit field {field!r}") from exc
    if isinstance(parsed, dict):
        parsed = [{key: item} for key, item in parsed.items()]
    if not isinstance(parsed, list):
        raise ValueError(f"Beemo edit field {field!r} is not a list")
    versions: dict[str, str] = {}
    for item in parsed:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"malformed item in Beemo edit field {field!r}")
        prompt_name, text = next(iter(item.items()))
        key = str(prompt_name).upper()
        if key in versions:
            raise ValueError(f"duplicate {key} in Beemo edit field {field!r}")
        versions[key] = _required_text(text, f"{field}.{key}")
    if set(versions) != {"P1", "P2", "P3"}:
        raise ValueError(f"Beemo edit field {field!r} must contain P1, P2, and P3")
    return versions


def word_edit_ratio(original: str, edited: str) -> float:
    """Return one minus word-sequence similarity, bounded to [0, 1]."""
    first = original.split()
    second = edited.split()
    if not first and not second:
        return 0.0
    return float(
        1.0 - difflib.SequenceMatcher(None, first, second, autojunk=False).ratio()
    )


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    record_id = str(record.get("id", "")).strip()
    if not record_id:
        raise ValueError("Beemo record is missing id")
    llama = parse_edit_versions(record.get("llama-3.1-70b_edits"), "llama-3.1-70b_edits")
    gpt = parse_edit_versions(record.get("gpt-4o_edits"), "gpt-4o_edits")
    return {
        "id": record_id,
        "category": _required_text(record.get("category"), "category"),
        "source_generator_model": _required_text(record.get("model"), "model"),
        "prompt_id": _required_text(record.get("prompt_id"), "prompt_id"),
        "prompt": _required_text(record.get("prompt"), "prompt"),
        "human": _required_text(record.get("human_output"), "human_output"),
        "original": _required_text(record.get("model_output"), "model_output"),
        "expert": _required_text(record.get("human_edits"), "human_edits"),
        "llama_p1": llama["P1"],
        "llama_p2": llama["P2"],
        "llama_p3": llama["P3"],
        "gpt_p1": gpt["P1"],
        "gpt_p2": gpt["P2"],
        "gpt_p3": gpt["P3"],
    }


def split_records(
    records: Sequence[dict[str, Any]],
    split_counts: dict[str, int],
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Hash-order record groups, then assign exact, disjoint split counts."""
    expected = sum(int(split_counts[name]) for name in ("clipping_tuning", "calibration", "test"))
    if len(records) != expected:
        raise ValueError(f"split counts require {expected} records, found {len(records)}")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Beemo record ids are not unique")
    ordered = sorted(records, key=lambda row: (stable_int(seed, row["id"]), row["id"]))
    assigned: list[tuple[str, dict[str, Any]]] = []
    offset = 0
    for split in ("clipping_tuning", "calibration", "test"):
        count = int(split_counts[split])
        assigned.extend((split, record) for record in ordered[offset : offset + count])
        offset += count
    return assigned


def debug_split_counts(record_count: int) -> dict[str, int]:
    if record_count < 3:
        raise ValueError("a Beemo debug run requires at least three records")
    tuning = max(1, int(round(record_count * 0.2)))
    remaining = record_count - tuning
    calibration = max(1, remaining // 2)
    test = record_count - tuning - calibration
    if test < 1:
        calibration -= 1
        test += 1
    return {"clipping_tuning": tuning, "calibration": calibration, "test": test}


def record_rows(
    record: dict[str, Any],
    split: str,
    dataset_config: dict[str, Any],
    dataset_fingerprint: str | None,
    dataset_resolved_revision: str | None,
    prepared_target_model: str,
) -> list[dict[str, Any]]:
    original = record["original"]
    rows = []
    for draw_id, variant in enumerate(VARIANTS):
        text = record[variant]
        if variant == "human":
            label = "human"
            editor_family = "human_reference"
            edit_ratio = None
        elif variant == "original":
            label = "llm"
            editor_family = "original"
            edit_ratio = 0.0
        elif variant == "expert":
            label = "llm"
            editor_family = "expert"
            edit_ratio = word_edit_ratio(original, text)
        elif variant.startswith("llama_"):
            label = "llm"
            editor_family = "llama"
            edit_ratio = word_edit_ratio(original, text)
        else:
            label = "llm"
            editor_family = "gpt"
            edit_ratio = word_edit_ratio(original, text)
        rows.append(
            {
                "dataset": "beemo",
                "dataset_id": dataset_config["id"],
                "dataset_revision": dataset_config.get("revision"),
                "dataset_resolved_revision": dataset_resolved_revision,
                "dataset_fingerprint": dataset_fingerprint,
                # Beemo contains outputs from multiple source generators. The
                # external reference scorer is attached only to score rows so
                # this prepared row can be reused by every scorer.
                "target_model": prepared_target_model,
                "sample_id": record["id"],
                "source_id": record["id"],
                "prompt_id": record["prompt_id"],
                "split": split,
                "label": label,
                "prompt": record["prompt"],
                "original_prompt": record["prompt"],
                "prompt_response_separator": "",
                "text": text,
                "contamination_mode": variant,
                "requested_contamination_ratio": 0.0,
                "realized_contamination_ratio": 0.0,
                "corruption_draw_id": draw_id,
                "beemo_variant": variant,
                "editor_family": editor_family,
                "machine_origin": variant != "human",
                "source_generator_model": record["source_generator_model"],
                "category": record["category"],
                "word_edit_ratio_from_original": edit_ratio,
            }
        )
    return rows


def prepare_from_records(
    records: Iterable[dict[str, Any]],
    output_path: str | Path,
    config: dict[str, Any],
    *,
    dataset_fingerprint: str | None = None,
    dataset_resolved_revision: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    normalized = [normalize_record(record) for record in records]
    normalized.sort(key=lambda row: (stable_int(config["splits"]["selection_seed"], row["id"]), row["id"]))
    if limit is not None:
        normalized = normalized[: int(limit)]
        counts = debug_split_counts(len(normalized))
    else:
        expected = int(config["dataset"]["expected_records"])
        if len(normalized) != expected:
            raise ValueError(f"expected {expected} Beemo records, found {len(normalized)}")
        counts = {name: int(config["splits"][name]) for name in ("clipping_tuning", "calibration", "test")}
    assigned = split_records(normalized, counts, int(config["splits"]["selection_seed"]))
    target = Path(output_path)
    completed = set()
    if target.exists():
        completed = {data_row_key(row) for row in iter_jsonl(target)}
    with AppendSafeJsonlWriter(target) as writer:
        for split, record in assigned:
            for row in record_rows(
                record,
                split,
                config["dataset"],
                dataset_fingerprint,
                dataset_resolved_revision,
                config.get("prepared_target_model", "beemo-mixed-source-generators"),
            ):
                key = data_row_key(row)
                if key not in completed:
                    writer.write(row)
                    completed.add(key)
        writer.checkpoint()
    rows = list(iter_jsonl(target))
    expected_rows = len(normalized) * len(VARIANTS)
    keys = [data_row_key(row) for row in rows]
    if len(rows) != expected_rows or len(keys) != len(set(keys)):
        raise ValueError("prepared Beemo rows failed count or unique-key validation")
    return {
        "records": len(normalized),
        "rows": len(rows),
        "variants_per_record": len(VARIANTS),
        "split_records": counts,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_resolved_revision": dataset_resolved_revision,
    }


def load_official_beemo(
    config: dict[str, Any],
) -> tuple[Iterable[dict[str, Any]], str | None, str | None]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("loading Beemo requires the datasets package") from exc
    dataset_config = config["dataset"]
    dataset = load_dataset(
        dataset_config["id"],
        dataset_config.get("config"),
        split=dataset_config["split"],
        revision=dataset_config.get("revision"),
    )
    resolved_revision = None
    try:
        from huggingface_hub import HfApi

        resolved_revision = HfApi().dataset_info(
            dataset_config["id"], revision=dataset_config.get("revision")
        ).sha
    except Exception:
        # A cached/offline preparation remains usable and records the dataset
        # fingerprint; a connected release run should also record the Hub SHA.
        resolved_revision = None
    return dataset, getattr(dataset, "_fingerprint", None), resolved_revision
