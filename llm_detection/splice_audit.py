"""Paired controls for separating donor authorship from splice artifacts."""

from __future__ import annotations

import csv
import difflib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .data import (
    apply_token_replacement_plan,
    contamination_counts,
    corruption_seed as primary_corruption_seed,
    random_token_replacement_plan,
    sentence_spans,
    stable_int,
)
from .evaluation import detector_local_values, oriented_score, robustness_auc
from .generation import GenerationRequest
from .io import AppendSafeJsonlWriter, atomic_write_json, iter_jsonl
from .runtime import Throughput, peak_gpu_memory_bytes


AUDIT_CONDITIONS = (
    "token_human",
    "token_llm",
    "sentence_human",
    "sentence_llm",
)
PLACEMENT_PAIRS = {
    "token": ("token_human", "token_llm"),
    "sentence": ("sentence_human", "sentence_llm"),
}
CORE_DETECTORS = ("log_likelihood", "rank", "log_rank", "lrr")


@dataclass(frozen=True)
class FrozenDetectorSpec:
    detector: str
    aggregation: str
    target_fpr: float
    direction: int
    clipping_specification: dict[str, float]
    calibration_threshold: float
    source_actual_fpr: float
    source_clean_tpr: float


def select_test_base_rows(
    base_path: str | Path,
    sample_count: int,
    selection_seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic target-independent subset of test sources."""
    rows = [row for row in iter_jsonl(base_path) if row.get("split") == "test"]
    rows.sort(
        key=lambda row: (
            stable_int(selection_seed, row["sample_id"]),
            str(row["sample_id"]),
        )
    )
    requested = int(sample_count)
    if requested <= 0:
        raise ValueError("sample_count must be positive")
    if len(rows) < requested:
        raise ValueError(f"requested {requested} test sources, found {len(rows)}")
    return rows[:requested]


def _decode_visible(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _stable_visible_tokens(
    tokenizer: Any,
    token_ids: Sequence[int],
    maximum: int,
    max_iterations: int = 12,
) -> tuple[str, list[int]]:
    current = list(token_ids)[: int(maximum)]
    for _ in range(max_iterations):
        text = _decode_visible(tokenizer, current)
        encoded = list(tokenizer.encode(text, add_special_tokens=False))
        if len(encoded) > maximum:
            encoded = encoded[:maximum]
        if encoded == current:
            return text, current
        current = encoded
    raise ValueError("alternative continuation tokenization did not stabilize")


def generate_alternative_continuations(
    selected_rows: Sequence[dict[str, Any]],
    output_path: str | Path,
    backend: Any,
    generation_seed: int,
    checkpoint_examples: int,
    minimum_tokens: int,
) -> int:
    """Generate one resumable same-prompt alternative continuation per source."""
    target = Path(output_path)
    completed = {
        str(row["sample_id"])
        for row in iter_jsonl(target)
    } if target.exists() else set()
    pending = [
        row for row in selected_rows if str(row["sample_id"]) not in completed
    ]
    progress = Throughput(len(selected_rows), "audit_generation")
    progress.examples = len(completed)
    requests = [
        GenerationRequest(
            key=str(row["sample_id"]),
            prompt=str(row["prompt"]),
            seed=int(generation_seed),
        )
        for row in pending
    ]
    by_id = {str(row["sample_id"]): row for row in pending}
    checkpoint = max(int(checkpoint_examples), 1)
    with AppendSafeJsonlWriter(target) as writer:
        for start in range(0, len(requests), checkpoint):
            batch = requests[start : start + checkpoint]
            generated = backend.generate(batch)
            for request in batch:
                base = by_id[request.key]
                raw_ids = list(generated[request.key])
                maximum = int(base["continuation_token_count"])
                if len(raw_ids) < int(minimum_tokens):
                    raise ValueError(
                        f"alternative generation for {request.key} stopped at "
                        f"{len(raw_ids)} tokens; minimum is {minimum_tokens}"
                    )
                text, token_ids = _stable_visible_tokens(
                    backend.tokenizer, raw_ids, maximum
                )
                if len(token_ids) < math.ceil(maximum * 0.5):
                    raise ValueError(
                        f"alternative generation for {request.key} has too few "
                        "stable visible tokens for the 50% audit condition"
                    )
                writer.write(
                    {
                        "sample_id": request.key,
                        "prompt": base["prompt"],
                        "alternative_generation_seed": int(generation_seed),
                        "alternative_llm_continuation": text,
                        "alternative_llm_token_ids": token_ids,
                        "raw_generated_token_count": len(raw_ids),
                        "stable_visible_token_count": len(token_ids),
                        "target_model": base["target_model"],
                        "target_model_resolved_revision": (
                            backend.resolved_model_revision
                        ),
                        "target_tokenizer_resolved_revision": (
                            backend.resolved_tokenizer_revision
                        ),
                    }
                )
                progress.update(1, len(token_ids), peak_gpu_memory_bytes())
            writer.checkpoint()
    count = sum(1 for _ in iter_jsonl(target))
    if count != len(selected_rows):
        raise ValueError(
            f"alternative generation produced {count} rows; "
            f"expected {len(selected_rows)}"
        )
    return count


def _matched_contiguous_chunks(
    donor_ids: Sequence[int],
    lengths: Sequence[int],
    seed: int,
) -> list[list[int]]:
    budget = sum(int(value) for value in lengths)
    values = list(donor_ids)
    if len(values) < budget:
        raise ValueError("alternative LLM donor is shorter than the token budget")
    rng = random.Random(int(seed))
    start = rng.randrange(len(values) - budget + 1)
    selected = values[start : start + budget]
    chunks: list[list[int]] = []
    cursor = 0
    for length in lengths:
        width = int(length)
        chunks.append(selected[cursor : cursor + width])
        cursor += width
    return chunks


def _map_token_boundaries(
    planned: Sequence[int],
    realized: Sequence[int],
    boundaries: Iterable[int],
) -> list[int]:
    """Map planned token boundaries through decode/re-tokenize drift."""
    matcher = difflib.SequenceMatcher(a=list(planned), b=list(realized), autojunk=False)
    opcodes = matcher.get_opcodes()
    mapped: list[int] = []
    for boundary in boundaries:
        value = int(boundary)
        value = max(0, min(value, len(planned)))
        target = len(realized)
        for _tag, first_start, first_end, second_start, second_end in opcodes:
            if first_start <= value <= first_end:
                first_width = first_end - first_start
                second_width = second_end - second_start
                if first_width == 0:
                    target = second_start
                else:
                    fraction = (value - first_start) / first_width
                    target = second_start + round(fraction * second_width)
                break
        mapped.append(max(0, min(int(target), len(realized))))
    return sorted(set(mapped))


def _sentence_segments(text: str) -> list[str]:
    """Split text into sentence-like segments while preserving exact whitespace."""
    if not text:
        return []
    pattern = re.compile(r".+?(?:[.!?](?:\s+|$)|\n+|$)", re.DOTALL)
    segments = [match.group(0) for match in pattern.finditer(text)]
    if "".join(segments) != text:
        return [text]
    return [segment for segment in segments if segment]


def _segment_content(segment: str) -> tuple[str, str]:
    match = re.search(r"\s*$", segment)
    suffix = match.group(0) if match else ""
    content = segment[: len(segment) - len(suffix)] if suffix else segment
    return content, suffix


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _recipient_sentence_indices(
    tokenizer: Any,
    recipient_segments: Sequence[str],
    budget: int,
    seed: int,
) -> list[int]:
    lengths = [_token_count(tokenizer, _segment_content(value)[0]) for value in recipient_segments]
    candidates = [index for index, length in enumerate(lengths) if length > 0]
    if not candidates:
        raise ValueError("recipient continuation has no sentence segments")
    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    cumulative = 0
    options: list[tuple[int, int]] = []
    for count, index in enumerate(candidates, start=1):
        cumulative += lengths[index]
        options.append((abs(cumulative - budget), count))
        if cumulative >= budget:
            break
    count = min(options)[1]
    return sorted(candidates[:count])


def _matched_donor_segments(
    tokenizer: Any,
    donor_text: str,
    recipient_lengths: Sequence[int],
    seed: int,
) -> tuple[list[str], list[int]]:
    donor_segments = _sentence_segments(donor_text)
    contents = [_segment_content(value)[0].strip() for value in donor_segments]
    candidates = [index for index, value in enumerate(contents) if value]
    if not candidates:
        raise ValueError("donor continuation has no sentence segments")
    rng = random.Random(int(seed))
    tie_order = list(candidates)
    rng.shuffle(tie_order)
    tie_rank = {value: index for index, value in enumerate(tie_order)}
    available = set(candidates)
    selected_text: list[str] = []
    selected_ids: list[int] = []
    for recipient_length in recipient_lengths:
        pool = available or set(candidates)
        chosen = min(
            pool,
            key=lambda index: (
                abs(_token_count(tokenizer, contents[index]) - recipient_length),
                tie_rank[index],
            ),
        )
        selected_text.append(contents[chosen])
        selected_ids.append(chosen)
        available.discard(chosen)
    return selected_text, selected_ids


def sentence_aligned_splice(
    tokenizer: Any,
    recipient_text: str,
    donor_text: str,
    ratio: float,
    recipient_seed: int,
    donor_seed: int,
    recipient_indices: Sequence[int] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Replace complete recipient sentences with length-matched donor sentences."""
    recipient_segments = _sentence_segments(recipient_text)
    original_count = _token_count(tokenizer, recipient_text)
    budget = max(1, int(round(original_count * float(ratio))))
    indices = list(recipient_indices) if recipient_indices is not None else (
        _recipient_sentence_indices(
            tokenizer, recipient_segments, budget, recipient_seed
        )
    )
    recipient_lengths = [
        _token_count(tokenizer, _segment_content(recipient_segments[index])[0])
        for index in indices
    ]
    donors, donor_indices = _matched_donor_segments(
        tokenizer, donor_text, recipient_lengths, donor_seed
    )
    replacements = dict(zip(indices, donors))
    pieces: list[str] = []
    character_windows: list[tuple[int, int]] = []
    cursor = 0
    for index, segment in enumerate(recipient_segments):
        if index in replacements:
            _, suffix = _segment_content(segment)
            value = replacements[index] + suffix
            start = cursor
            pieces.append(value)
            cursor += len(value)
            character_windows.append((start, cursor - len(suffix)))
        else:
            pieces.append(segment)
            cursor += len(segment)
    mixed_text = "".join(pieces)
    final_ids = list(tokenizer.encode(mixed_text, add_special_tokens=False))
    boundaries = _character_to_token_boundaries(
        tokenizer, mixed_text, [value for window in character_windows for value in window]
    )
    donor_tokens = sum(_token_count(tokenizer, value) for value in donors)
    metadata = contamination_counts(
        float(ratio), original_count, donor_tokens, sum(recipient_lengths)
    )
    metadata["final_token_count"] = len(final_ids)
    metadata["length_delta_tokens"] = len(final_ids) - original_count
    metadata["realized_contamination_ratio"] = (
        donor_tokens / len(final_ids) if final_ids else 0.0
    )
    metadata.update(
        {
            "recipient_sentence_indices": indices,
            "donor_sentence_indices": donor_indices,
            "recipient_sentence_token_count": sum(recipient_lengths),
            "splice_character_windows": [list(value) for value in character_windows],
            "splice_boundary_token_indices": boundaries,
        }
    )
    return mixed_text, metadata


def _character_to_token_boundaries(
    tokenizer: Any,
    text: str,
    character_boundaries: Sequence[int],
) -> list[int]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
    except (TypeError, KeyError, NotImplementedError):
        return sorted(
            {
                len(tokenizer.encode(text[:value], add_special_tokens=False))
                for value in character_boundaries
            }
        )
    result = []
    for boundary in character_boundaries:
        token_boundary = len(offsets)
        for index, (_start, end) in enumerate(offsets):
            if end > boundary:
                token_boundary = index
                break
        result.append(token_boundary)
    return sorted(set(result))


def _audit_base_row(
    audit_id: str,
    source_run_id: str,
    base: dict[str, Any],
    text: str,
    condition: str,
    ratio: float,
    seed: int | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": audit_id,
        "audit_protocol": "paired-splice-artifact-v1",
        "source_run_id": source_run_id,
        "dataset": base["dataset"],
        "dataset_id": base["dataset_id"],
        "dataset_config": base.get("dataset_config"),
        "dataset_revision": base.get("dataset_revision"),
        "dataset_resolved_revision": base.get("dataset_resolved_revision"),
        "source_id": base["source_id"],
        "sample_id": base["sample_id"],
        "split": "test",
        "generation_seed": base["generation_seed"],
        "target_model": base["target_model"],
        "target_model_revision": base.get("target_model_revision"),
        "target_model_resolved_revision": base.get(
            "target_model_resolved_revision"
        ),
        "target_tokenizer_revision": base.get("target_tokenizer_revision"),
        "target_tokenizer_resolved_revision": base.get(
            "target_tokenizer_resolved_revision"
        ),
        "prompt": base["prompt"],
        "text": text,
        "label": "llm",
        "contamination_mode": condition,
        "audit_condition": condition,
        "donor_authorship": (
            "none" if condition == "none" else condition.split("_", 1)[1]
        ),
        "corruption_seed": seed,
        "corruption_draw_id": 0,
        **metadata,
    }


def construct_splice_audit_data(
    audit_id: str,
    source_run_id: str,
    selected_rows: Sequence[dict[str, Any]],
    alternative_path: str | Path,
    output_path: str | Path,
    tokenizer: Any,
    ratios: Sequence[float],
    corruption_seed: int,
) -> int:
    """Construct one clean row and a paired 2x2 audit per source and ratio."""
    alternatives = {
        str(row["sample_id"]): row for row in iter_jsonl(alternative_path)
    }
    expected = len(selected_rows) * (1 + len(AUDIT_CONDITIONS) * len(ratios))
    target = Path(output_path)
    if target.exists():
        existing = sum(1 for _ in iter_jsonl(target))
        if existing == expected:
            return existing
        raise ValueError(
            f"partial audit data exists with {existing} rows; construction is "
            "atomic, so choose a new audit ID or remove only that incomplete audit"
        )
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for base in selected_rows:
            sample_id = str(base["sample_id"])
            alternative = alternatives[sample_id]
            original_ids = list(base["llm_token_ids"])
            original_text = str(base["llm_continuation"])
            human_text = str(base["human_continuation"])
            alt_text = str(alternative["alternative_llm_continuation"])
            alt_ids = list(alternative["alternative_llm_token_ids"])
            clean_metadata = contamination_counts(
                0.0, len(original_ids), 0, 0
            )
            clean_metadata["splice_boundary_token_indices"] = []
            clean = _audit_base_row(
                audit_id,
                source_run_id,
                base,
                original_text,
                "none",
                0.0,
                None,
                clean_metadata,
            )
            handle.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")
            human_spans = sentence_spans(tokenizer, human_text)
            for ratio in ratios:
                seed = primary_corruption_seed(
                    int(corruption_seed),
                    str(base["dataset"]),
                    str(base["target_model"]),
                    sample_id,
                    0,
                )
                human_chunks, windows = random_token_replacement_plan(
                    len(original_ids), human_spans, float(ratio), seed
                )
                lengths = [len(value) for value in human_chunks]
                llm_chunks = _matched_contiguous_chunks(
                    alt_ids,
                    lengths,
                    stable_int(seed, "alternative-llm-donor", bits=31),
                )
                for condition, chunks in (
                    ("token_human", human_chunks),
                    ("token_llm", llm_chunks),
                ):
                    planned = apply_token_replacement_plan(
                        original_ids, chunks, windows
                    )
                    text = _decode_visible(tokenizer, planned)
                    realized = list(tokenizer.encode(text, add_special_tokens=False))
                    boundaries = _map_token_boundaries(
                        planned,
                        realized,
                        [value for window in windows for value in window],
                    )
                    metadata = contamination_counts(
                        float(ratio), len(original_ids), sum(lengths), sum(lengths)
                    )
                    metadata["final_token_count"] = len(realized)
                    metadata["length_delta_tokens"] = len(realized) - len(original_ids)
                    metadata["realized_contamination_ratio"] = (
                        sum(lengths) / len(realized) if realized else 0.0
                    )
                    metadata["splice_token_windows_pre_roundtrip"] = [
                        list(value) for value in windows
                    ]
                    metadata["splice_boundary_token_indices"] = boundaries
                    metadata["donor_token_count"] = sum(lengths)
                    metadata["llm_donor_token_count"] = (
                        sum(lengths) if condition == "token_llm" else 0
                    )
                    if condition == "token_llm":
                        metadata["human_token_count"] = 0
                    row = _audit_base_row(
                        audit_id,
                        source_run_id,
                        base,
                        text,
                        condition,
                        float(ratio),
                        seed,
                        metadata,
                    )
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                recipient_segments = _sentence_segments(original_text)
                budget = max(1, int(round(len(original_ids) * float(ratio))))
                recipient_indices = _recipient_sentence_indices(
                    tokenizer, recipient_segments, budget, seed
                )
                for condition, donor_text in (
                    ("sentence_human", human_text),
                    ("sentence_llm", alt_text),
                ):
                    text, metadata = sentence_aligned_splice(
                        tokenizer,
                        original_text,
                        donor_text,
                        float(ratio),
                        seed,
                        stable_int(seed, condition, bits=31),
                        recipient_indices,
                    )
                    donor_count = int(metadata["human_token_count"])
                    metadata["donor_token_count"] = donor_count
                    metadata["llm_donor_token_count"] = (
                        donor_count if condition == "sentence_llm" else 0
                    )
                    if condition == "sentence_llm":
                        metadata["human_token_count"] = 0
                    row = _audit_base_row(
                        audit_id,
                        source_run_id,
                        base,
                        text,
                        condition,
                        float(ratio),
                        seed,
                        metadata,
                    )
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
        handle.flush()
    temporary.replace(target)
    count = sum(1 for _ in iter_jsonl(target))
    if count != expected:
        raise ValueError(f"audit construction produced {count} rows; expected {expected}")
    return count


def load_frozen_detector_specs(
    metrics_path: str | Path,
    analysis: str = "primary_frozen_mixture",
) -> dict[tuple[str, str, float], FrozenDetectorSpec]:
    """Load and validate frozen choices from the completed source cell."""
    grouped: dict[tuple[str, str, float], list[dict[str, str]]] = defaultdict(list)
    with Path(metrics_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "test" or row.get("analysis") != analysis:
                continue
            key = (
                str(row["detector"]),
                str(row["aggregation"]),
                float(row["target_fpr"]),
            )
            grouped[key].append(row)
    if not grouped:
        raise ValueError("source metrics contain no matching frozen test analysis")
    result: dict[tuple[str, str, float], FrozenDetectorSpec] = {}
    for key, rows in grouped.items():
        detector, aggregation, target_fpr = key
        directions = {int(row["direction"]) for row in rows}
        specifications = {row["clipping_specification"] for row in rows}
        thresholds = {float(row["calibration_threshold"]) for row in rows}
        actual_fprs = {float(row["actual_fpr"]) for row in rows}
        clean_tprs = {
            float(row["tpr"])
            for row in rows
            if float(row["requested_contamination_ratio"]) == 0.0
        }
        if not (
            len(directions)
            == len(specifications)
            == len(thresholds)
            == len(actual_fprs)
            == len(clean_tprs)
            == 1
        ):
            raise ValueError(f"frozen source metrics disagree for {key}")
        result[key] = FrozenDetectorSpec(
            detector=detector,
            aggregation=aggregation,
            target_fpr=target_fpr,
            direction=next(iter(directions)),
            clipping_specification=json.loads(next(iter(specifications))),
            calibration_threshold=next(iter(thresholds)),
            source_actual_fpr=next(iter(actual_fprs)),
            source_clean_tpr=next(iter(clean_tprs)),
        )
    return result


def _percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(array):
        return float("nan"), float("nan")
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def _score_records(
    score_path: str | Path,
    specs: dict[tuple[str, str, float], FrozenDetectorSpec],
    detectors: set[str],
    boundary_radius: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, float, str], dict[str, float]]]:
    relevant = [spec for spec in specs.values() if spec.detector in detectors]
    records: list[dict[str, Any]] = []
    diagnostics: dict[tuple[str, float, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in iter_jsonl(score_path):
        condition = str(row.get("audit_condition", row["contamination_mode"]))
        ratio = float(row["requested_contamination_ratio"])
        scores: dict[str, dict[str, float]] = defaultdict(dict)
        seen: set[tuple[str, str]] = set()
        for spec in relevant:
            detector_aggregation = (spec.detector, spec.aggregation)
            if detector_aggregation in seen:
                continue
            seen.add(detector_aggregation)
            scores[spec.detector][spec.aggregation] = oriented_score(
                row,
                spec.detector,
                spec.direction,
                spec.clipping_specification
                if spec.aggregation == "clipped"
                else {},
            )
        records.append(
            {
                "sample_id": str(row["sample_id"]),
                "condition": condition,
                "ratio": ratio,
                "scores": {key: dict(value) for key, value in scores.items()},
            }
        )
        boundaries = [int(value) for value in row.get("splice_boundary_token_indices", [])]
        if not boundaries or condition == "none":
            continue
        for detector in detectors - {"lrr"}:
            key = next(
                (value for value in relevant if value.detector == detector),
                None,
            )
            if key is None:
                continue
            local = key.direction * detector_local_values(row, detector)
            mask = np.zeros(len(local), dtype=bool)
            for boundary in boundaries:
                start = max(0, boundary - int(boundary_radius))
                end = min(len(local), boundary + int(boundary_radius) + 1)
                mask[start:end] = True
            group = diagnostics[(condition, ratio, detector)]
            group["boundary_sum"] += float(local[mask].sum())
            group["boundary_count"] += int(mask.sum())
            group["interior_sum"] += float(local[~mask].sum())
            group["interior_count"] += int((~mask).sum())
            clipped_spec = next(
                (
                    value.clipping_specification
                    for value in relevant
                    if value.detector == detector
                    and value.aggregation == "clipped"
                ),
                {},
            )
            if "lower" in clipped_spec:
                group["boundary_clipped"] += int(
                    np.sum(local[mask] < clipped_spec["lower"])
                )
                group["interior_clipped"] += int(
                    np.sum(local[~mask] < clipped_spec["lower"])
                )
    return records, diagnostics


def _group_score_matrix(
    records: Sequence[dict[str, Any]],
    detector: str,
    aggregation: str,
    condition: str,
    ratios: Sequence[float],
) -> tuple[list[str], np.ndarray]:
    by_key = {
        (str(row["sample_id"]), str(row["condition"]), float(row["ratio"])): float(
            row["scores"][detector][aggregation]
        )
        for row in records
        if detector in row["scores"]
    }
    clean_ids = sorted(
        sample_id
        for sample_id, row_condition, ratio in by_key
        if row_condition == "none" and ratio == 0.0
    )
    matrix = np.empty((len(clean_ids), len(ratios)), dtype=float)
    for row_index, sample_id in enumerate(clean_ids):
        for column_index, ratio in enumerate(ratios):
            row_condition = "none" if ratio == 0.0 else condition
            key = (sample_id, row_condition, float(ratio))
            if key not in by_key:
                raise ValueError(f"missing paired audit score {key}")
            matrix[row_index, column_index] = by_key[key]
    return clean_ids, matrix


def _bootstrap_curve(
    matrix: np.ndarray,
    ratios: Sequence[float],
    threshold: float,
    repetitions: int,
    seed: int,
) -> tuple[list[float], tuple[float, float], list[float]]:
    curve = [float(np.mean(matrix[:, index] >= threshold)) for index in range(matrix.shape[1])]
    auc = robustness_auc(ratios, curve)
    rng = np.random.default_rng(int(seed))
    auc_values: list[float] = []
    for _ in range(int(repetitions)):
        sampled = rng.integers(0, len(matrix), size=len(matrix))
        sampled_curve = [
            float(np.mean(matrix[sampled, index] >= threshold))
            for index in range(matrix.shape[1])
        ]
        auc_values.append(robustness_auc(ratios, sampled_curve))
    return curve, _percentile_interval(auc_values), auc_values


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate_splice_audit(
    target_score_path: str | Path,
    source_metrics_path: str | Path,
    output_dir: str | Path,
    ratios: Sequence[float],
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    boundary_radius: int,
    binoculars_score_path: str | Path | None = None,
) -> dict[str, Any]:
    """Apply source-cell frozen choices and write paired audit reports."""
    frozen = load_frozen_detector_specs(source_metrics_path)
    target_detectors = {
        detector for detector, _aggregation, _fpr in frozen if detector != "binoculars"
    }
    records, diagnostics = _score_records(
        target_score_path, frozen, target_detectors, boundary_radius
    )
    if binoculars_score_path is not None:
        pair_records, pair_diagnostics = _score_records(
            binoculars_score_path, frozen, {"binoculars"}, boundary_radius
        )
        records.extend(pair_records)
        for key, values in pair_diagnostics.items():
            for name, value in values.items():
                diagnostics[key][name] += value
    available_detectors = sorted(
        {detector for row in records for detector in row["scores"]}
    )
    curve_ratios = [0.0] + [float(value) for value in ratios]
    metrics_rows: list[dict[str, Any]] = []
    curve_cache: dict[tuple[str, str, float, str], dict[str, Any]] = {}
    for detector in available_detectors:
        for aggregation in ("raw", "clipped"):
            for target_fpr in (0.01, 0.05):
                spec = frozen[(detector, aggregation, target_fpr)]
                for condition in AUDIT_CONDITIONS:
                    ids, matrix = _group_score_matrix(
                        records,
                        detector,
                        aggregation,
                        condition,
                        curve_ratios,
                    )
                    seed = int(bootstrap_seed) + stable_int(
                        "splice-audit",
                        detector,
                        aggregation,
                        target_fpr,
                        condition,
                        bits=20,
                    )
                    curve, auc_ci, auc_values = _bootstrap_curve(
                        matrix,
                        curve_ratios,
                        spec.calibration_threshold,
                        bootstrap_repetitions,
                        seed,
                    )
                    auc = robustness_auc(curve_ratios, curve)
                    curve_cache[(detector, aggregation, target_fpr, condition)] = {
                        "auc": auc,
                        "auc_ci": auc_ci,
                        "bootstrap_auc": auc_values,
                    }
                    for index, ratio in enumerate(curve_ratios):
                        rng = np.random.default_rng(seed + index + 1)
                        tpr_values = []
                        for _ in range(int(bootstrap_repetitions)):
                            sampled = rng.integers(0, len(matrix), size=len(matrix))
                            tpr_values.append(
                                float(
                                    np.mean(
                                        matrix[sampled, index]
                                        >= spec.calibration_threshold
                                    )
                                )
                            )
                        tpr_ci = _percentile_interval(tpr_values)
                        metrics_rows.append(
                            {
                                "detector": detector,
                                "aggregation": aggregation,
                                "target_fpr": target_fpr,
                                "condition": condition,
                                "placement": condition.split("_", 1)[0],
                                "donor": condition.split("_", 1)[1],
                                "requested_contamination_ratio": ratio,
                                "tpr": curve[index],
                                "tpr_ci_low": tpr_ci[0],
                                "tpr_ci_high": tpr_ci[1],
                                "robustness_auc": auc,
                                "robustness_auc_ci_low": auc_ci[0],
                                "robustness_auc_ci_high": auc_ci[1],
                                "direction": spec.direction,
                                "clipping_specification": json.dumps(
                                    spec.clipping_specification, sort_keys=True
                                ),
                                "calibration_threshold": spec.calibration_threshold,
                                "source_actual_fpr": spec.source_actual_fpr,
                                "source_clean_tpr": spec.source_clean_tpr,
                                "audit_clean_tpr": curve[0],
                                "n_sources": len(ids),
                                "bootstrap_repetitions": bootstrap_repetitions,
                                "bootstrap_seed": bootstrap_seed,
                            }
                        )
    artifact_rows: list[dict[str, Any]] = []
    for detector in available_detectors:
        for aggregation in ("raw", "clipped"):
            for target_fpr in (0.01, 0.05):
                clean_tpr = next(
                    row["audit_clean_tpr"]
                    for row in metrics_rows
                    if row["detector"] == detector
                    and row["aggregation"] == aggregation
                    and row["target_fpr"] == target_fpr
                )
                for placement, (human_condition, llm_condition) in PLACEMENT_PAIRS.items():
                    human = curve_cache[
                        (detector, aggregation, target_fpr, human_condition)
                    ]
                    llm = curve_cache[
                        (detector, aggregation, target_fpr, llm_condition)
                    ]
                    human_drop = clean_tpr - human["auc"]
                    boundary_drop = clean_tpr - llm["auc"]
                    share = (
                        boundary_drop / human_drop
                        if abs(human_drop) > 1e-12
                        else float("nan")
                    )
                    human_ids, human_matrix = _group_score_matrix(
                        records,
                        detector,
                        aggregation,
                        human_condition,
                        curve_ratios,
                    )
                    llm_ids, llm_matrix = _group_score_matrix(
                        records,
                        detector,
                        aggregation,
                        llm_condition,
                        curve_ratios,
                    )
                    if human_ids != llm_ids:
                        raise ValueError(
                            f"paired source IDs disagree for {detector} {placement}"
                        )
                    threshold = frozen[
                        (detector, aggregation, target_fpr)
                    ].calibration_threshold
                    rng = np.random.default_rng(
                        int(bootstrap_seed)
                        + stable_int(
                            "artifact-share",
                            detector,
                            aggregation,
                            target_fpr,
                            placement,
                            bits=20,
                        )
                    )
                    bootstrap_shares = []
                    for _ in range(int(bootstrap_repetitions)):
                        sampled = rng.integers(
                            0, len(human_matrix), size=len(human_matrix)
                        )
                        clean_bootstrap_tpr = float(
                            np.mean(human_matrix[sampled, 0] >= threshold)
                        )
                        human_curve = [
                            float(
                                np.mean(human_matrix[sampled, index] >= threshold)
                            )
                            for index in range(human_matrix.shape[1])
                        ]
                        llm_curve = [
                            float(np.mean(llm_matrix[sampled, index] >= threshold))
                            for index in range(llm_matrix.shape[1])
                        ]
                        human_auc = robustness_auc(curve_ratios, human_curve)
                        llm_auc = robustness_auc(curve_ratios, llm_curve)
                        denominator = clean_bootstrap_tpr - human_auc
                        if abs(denominator) > 1e-12:
                            bootstrap_shares.append(
                                (clean_bootstrap_tpr - llm_auc) / denominator
                            )
                    share_ci = _percentile_interval(bootstrap_shares)
                    band = (
                        "not-interpretable"
                        if not math.isfinite(share) or human_drop <= 0.02
                        else "low"
                        if share < 0.3
                        else "mixed"
                        if share < 0.7
                        else "high"
                    )
                    artifact_rows.append(
                        {
                            "detector": detector,
                            "core_detector": detector in CORE_DETECTORS,
                            "aggregation": aggregation,
                            "target_fpr": target_fpr,
                            "placement": placement,
                            "clean_tpr": clean_tpr,
                            "human_robustness_auc": human["auc"],
                            "llm_robustness_auc": llm["auc"],
                            "human_degradation": human_drop,
                            "boundary_degradation": boundary_drop,
                            "boundary_artifact_share": share,
                            "boundary_artifact_share_ci_low": share_ci[0],
                            "boundary_artifact_share_ci_high": share_ci[1],
                            "decision_band": band,
                        }
                    )
    diagnostic_rows = []
    for (condition, ratio, detector), values in sorted(diagnostics.items()):
        boundary_count = int(values["boundary_count"])
        interior_count = int(values["interior_count"])
        boundary_mean = (
            values["boundary_sum"] / boundary_count if boundary_count else float("nan")
        )
        interior_mean = (
            values["interior_sum"] / interior_count if interior_count else float("nan")
        )
        diagnostic_rows.append(
            {
                "condition": condition,
                "requested_contamination_ratio": ratio,
                "detector": detector,
                "boundary_radius_tokens": boundary_radius,
                "boundary_token_count": boundary_count,
                "interior_token_count": interior_count,
                "mean_oriented_boundary_contribution": boundary_mean,
                "mean_oriented_interior_contribution": interior_mean,
                "boundary_minus_interior": boundary_mean - interior_mean,
                "boundary_clipped_fraction": (
                    values["boundary_clipped"] / boundary_count
                    if boundary_count
                    else float("nan")
                ),
                "interior_clipped_fraction": (
                    values["interior_clipped"] / interior_count
                    if interior_count
                    else float("nan")
                ),
            }
        )
    output = Path(output_dir)
    _write_csv(output / "metrics.csv", metrics_rows)
    _write_csv(output / "artifact_share.csv", artifact_rows)
    if diagnostic_rows:
        _write_csv(output / "boundary_diagnostics.csv", diagnostic_rows)
    interpretable_core = [
        row
        for row in artifact_rows
        if row["core_detector"]
        and row["decision_band"] != "not-interpretable"
    ]
    median_share = (
        float(np.median([row["boundary_artifact_share"] for row in interpretable_core]))
        if interpretable_core
        else float("nan")
    )
    high_count = sum(row["decision_band"] == "high" for row in interpretable_core)
    summary = {
        "audit_status": "complete",
        "protocol": "paired-splice-artifact-v1",
        "detectors": available_detectors,
        "conditions": list(AUDIT_CONDITIONS),
        "ratios": curve_ratios,
        "metric_rows": len(metrics_rows),
        "artifact_share_rows": len(artifact_rows),
        "boundary_diagnostic_rows": len(diagnostic_rows),
        "interpretable_core_comparisons": len(interpretable_core),
        "median_core_boundary_artifact_share": median_share,
        "high_artifact_core_comparisons": high_count,
        "decision_note": (
            "Heuristic bands are diagnostic, not a formal hypothesis test. "
            "Inspect paired confidence intervals and sentence-aligned results."
        ),
    }
    atomic_write_json(output / "summary.json", summary)
    return summary
