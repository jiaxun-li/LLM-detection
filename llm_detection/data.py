"""Deterministic source splits and token-budget contamination."""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


SPLIT_ORDER = ("clipping_tuning", "calibration", "test")


def stable_int(*parts: Any, bits: int = 63) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value & ((1 << bits) - 1)


def assign_splits_and_generation_seeds(
    source_rows: Sequence[dict[str, Any]],
    split_counts: dict[str, int],
    generation_seeds: Sequence[int],
    split_seed: int,
) -> list[dict[str, Any]]:
    """Select and label source rows independently of any target model.

    Generation seeds partition the requested total: every selected source occurs
    once and receives exactly one generation seed.
    """
    total = sum(int(split_counts[name]) for name in SPLIT_ORDER)
    if len(source_rows) < total:
        raise ValueError(f"need {total} unique sources, found {len(source_rows)}")
    if not generation_seeds:
        raise ValueError("at least one generation seed is required")

    deduped: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        source_id = str(row["source_id"])
        deduped.setdefault(source_id, row)
    if len(deduped) < total:
        raise ValueError(f"need {total} unique source IDs, found {len(deduped)}")

    ordered = sorted(
        deduped.values(),
        key=lambda row: (stable_int(split_seed, row["source_id"]), str(row["source_id"])),
    )[:total]
    result: list[dict[str, Any]] = []
    cursor = 0
    global_index = 0
    for split in SPLIT_ORDER:
        for _ in range(int(split_counts[split])):
            source = dict(ordered[cursor])
            source["sample_id"] = str(source["source_id"])
            source["split"] = split
            source["generation_seed"] = int(
                generation_seeds[global_index % len(generation_seeds)]
            )
            source["source_selection_seed"] = int(split_seed)
            result.append(source)
            cursor += 1
            global_index += 1
    validate_disjoint_splits(result, split_counts)
    return result


def validate_disjoint_splits(
    rows: Sequence[dict[str, Any]], expected_counts: dict[str, int] | None = None
) -> None:
    seen: dict[str, str] = {}
    counts = {name: 0 for name in SPLIT_ORDER}
    for row in rows:
        split = row["split"]
        sample_id = str(row["sample_id"])
        if split not in counts:
            raise ValueError(f"unknown split {split!r}")
        if sample_id in seen:
            raise ValueError(
                f"sample_id {sample_id!r} appears in both {seen[sample_id]} and {split}"
            )
        seen[sample_id] = split
        counts[split] += 1
    if expected_counts is not None:
        expected = {name: int(expected_counts[name]) for name in SPLIT_ORDER}
        if counts != expected:
            raise ValueError(f"split counts {counts} do not match {expected}")


def sentence_spans(tokenizer: Any, text: str) -> list[list[int]]:
    pieces = [
        piece.strip()
        for piece in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if piece.strip()
    ]
    spans = [tokenizer.encode(piece, add_special_tokens=False) for piece in pieces]
    return [list(span) for span in spans if span]


@dataclass(frozen=True)
class TailCandidate:
    candidate_id: int
    token_ids: tuple[int, ...]
    nll: float


def rank_tail_candidates_once(
    candidate_spans: Sequence[Sequence[int]],
    score_span: Callable[[Sequence[int]], float],
) -> list[TailCandidate]:
    """Score every tail candidate exactly once and return a frozen ordering."""
    scored = [
        TailCandidate(index, tuple(span), float(score_span(span)))
        for index, span in enumerate(candidate_spans)
        if span
    ]
    return sorted(scored, key=lambda item: (-item.nll, item.candidate_id))


def _token_budget(original_count: int, ratio: float) -> int:
    if ratio <= 0:
        return 0
    return min(original_count, max(1, int(round(original_count * ratio))))


def _take_human_tokens(
    spans: Sequence[Sequence[int]],
    budget: int,
    rng: random.Random | None = None,
) -> list[int]:
    order = list(range(len(spans)))
    if rng is not None:
        rng.shuffle(order)
    tokens: list[int] = []
    for index in order:
        span = list(spans[index])
        if rng is not None and len(span) > 1:
            start = rng.randrange(len(span))
            span = span[start:] + span[:start]
        need = budget - len(tokens)
        tokens.extend(span[:need])
        if len(tokens) >= budget:
            break
    if len(tokens) < budget:
        flattened = [token for span in spans for token in span]
        if not flattened:
            raise ValueError("human continuation has no tokens")
        while len(tokens) < budget:
            tokens.extend(flattened[: budget - len(tokens)])
    return tokens[:budget]


def _take_human_chunks(
    spans: Sequence[Sequence[int]],
    budget: int,
    rng: random.Random,
) -> list[list[int]]:
    """Choose intact random human spans, truncating only the final budget span."""
    candidates = [list(span) for span in spans if span]
    if not candidates:
        raise ValueError("human continuation has no tokens")
    rng.shuffle(candidates)
    chunks: list[list[int]] = []
    consumed = 0
    cursor = 0
    while consumed < budget:
        span = candidates[cursor % len(candidates)]
        need = budget - consumed
        chunk = span[:need]
        if chunk:
            chunks.append(chunk)
            consumed += len(chunk)
        cursor += 1
    return chunks


def _random_nonoverlapping_windows(
    total_length: int,
    chunk_lengths: Sequence[int],
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Place chunks with a randomized order and randomized separating gaps."""
    if sum(chunk_lengths) > total_length:
        raise ValueError("contamination chunks exceed continuation length")
    indexed = list(enumerate(chunk_lengths))
    rng.shuffle(indexed)
    free = total_length - sum(chunk_lengths)
    gaps = [0] * (len(indexed) + 1)
    for _ in range(free):
        gaps[rng.randrange(len(gaps))] += 1
    placements: dict[int, tuple[int, int]] = {}
    cursor = gaps[0]
    for placement_index, (original_index, length) in enumerate(indexed):
        placements[original_index] = (cursor, cursor + length)
        cursor += length + gaps[placement_index + 1]
    return [placements[index] for index in range(len(chunk_lengths))]


def _replace_at_positions(
    original_ids: Sequence[int],
    human_ids: Sequence[int],
    positions: Sequence[int],
) -> list[int]:
    mixed = list(original_ids)
    for position, token_id in zip(positions, human_ids):
        mixed[position] = int(token_id)
    return mixed


def random_token_contamination(
    original_ids: Sequence[int],
    human_spans: Sequence[Sequence[int]],
    ratio: float,
    corruption_seed: int,
) -> tuple[list[int], dict[str, Any]]:
    """Replace an exact target-tokenizer token budget at random positions."""
    original_count = len(original_ids)
    budget = _token_budget(original_count, ratio)
    if budget == 0:
        return list(original_ids), contamination_counts(ratio, original_count, 0, 0)
    rng = random.Random(corruption_seed)
    chunks = _take_human_chunks(human_spans, budget, rng)
    windows = _random_nonoverlapping_windows(
        original_count, [len(chunk) for chunk in chunks], rng
    )
    mixed = list(original_ids)
    for chunk, (start, end) in zip(chunks, windows):
        mixed[start:end] = chunk
    return mixed, contamination_counts(ratio, original_count, budget, budget)


def tail_token_contamination(
    original_ids: Sequence[int],
    ranked_candidates: Sequence[TailCandidate],
    ratio: float,
) -> tuple[list[int], dict[str, Any]]:
    """White-box replacement using a once-scored highest-NLL candidate order.

    Candidate spans are consumed in frozen order until the token budget is met.
    Replacements are placed as an approximately length-matched suffix attack.
    """
    original_count = len(original_ids)
    budget = _token_budget(original_count, ratio)
    if budget == 0:
        return list(original_ids), contamination_counts(ratio, original_count, 0, 0)
    spans = [candidate.token_ids for candidate in ranked_candidates]
    human_ids = _take_human_tokens(spans, budget)
    selected_ids: list[int] = []
    consumed = 0
    for candidate in ranked_candidates:
        if not candidate.token_ids or consumed >= budget:
            continue
        selected_ids.append(candidate.candidate_id)
        consumed += min(len(candidate.token_ids), budget - consumed)
    start = original_count - budget
    positions = list(range(start, original_count))
    mixed = _replace_at_positions(original_ids, human_ids, positions)
    metadata = contamination_counts(ratio, original_count, len(human_ids), budget)
    metadata["tail_candidate_ids"] = selected_ids
    return mixed, metadata


def contamination_counts(
    requested_ratio: float,
    original_token_count: int,
    human_token_count: int,
    replaced_token_count: int,
) -> dict[str, Any]:
    final_count = original_token_count
    realized = (human_token_count / final_count) if final_count else 0.0
    return {
        "requested_contamination_ratio": float(requested_ratio),
        "realized_contamination_ratio": float(realized),
        "original_token_count": int(original_token_count),
        "human_token_count": int(human_token_count),
        "replaced_token_count": int(replaced_token_count),
        "final_token_count": int(final_count),
        "length_delta_tokens": 0,
    }


def corruption_seed(
    base_seed: int,
    dataset: str,
    target_model: str,
    sample_id: str,
    draw_id: int,
) -> int:
    return stable_int(base_seed, dataset, target_model, sample_id, draw_id, bits=31)


def data_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["dataset"],
        row["target_model"],
        row["sample_id"],
        row["split"],
        row["label"],
        row["contamination_mode"],
        float(row["requested_contamination_ratio"]),
        int(row.get("corruption_draw_id", 0)),
    )


def expected_rows_per_source(ratios: Sequence[float], random_draws: int) -> int:
    positive = sum(float(ratio) > 0 for ratio in ratios)
    # Human clean + LLM clean once; random has N draws and tail has one.
    return 2 + positive * (int(random_draws) + 1)
