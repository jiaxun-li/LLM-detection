#!/usr/bin/env python3
"""Audit decode/re-tokenize drift for every cached contamination construction."""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from llm_detection.config import load_config
from llm_detection.data import (
    TailCandidate,
    corruption_seed,
    random_token_contamination,
    sentence_spans,
    tail_token_contamination,
)
from llm_detection.io import atomic_write_json, iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--output")
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def _percentile(values: Sequence[int], quantile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def audit_cached_constructions(
    base_rows: Iterable[dict[str, Any]],
    tail_rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    config: dict[str, Any],
    *,
    top_k: int = 30,
) -> dict[str, Any]:
    """Reconstruct every positive-ratio row and summarize textual round trips."""
    base = list(base_rows)
    tail_cache = {str(row["sample_id"]): row for row in tail_rows}
    if len(base) != len(tail_cache):
        raise ValueError(
            f"base/tail cache count mismatch: {len(base)} versus {len(tail_cache)}"
        )
    if {str(row["sample_id"]) for row in base} != set(tail_cache):
        raise ValueError("base and tail cache sample IDs disagree")

    ratios = [
        float(value)
        for value in config["contamination"]["ratios"]
        if float(value) > 0.0
    ]
    random_draws = int(config["contamination"]["random_draws"])
    tolerance = int(config["contamination"]["max_length_delta_tokens"])
    conditions: dict[tuple[str, float], list[int]] = collections.defaultdict(list)
    worst: list[dict[str, Any]] = []

    def measure(
        sample_id: str,
        mode: str,
        ratio: float,
        draw_id: int,
        token_ids: Sequence[int],
        original_count: int,
        metadata: dict[str, Any],
    ) -> None:
        text = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        final_count = len(tokenizer.encode(text, add_special_tokens=False))
        delta = int(final_count - original_count)
        conditions[(mode, ratio)].append(delta)
        worst.append(
            {
                "sample_id": sample_id,
                "contamination_mode": mode,
                "requested_contamination_ratio": ratio,
                "corruption_draw_id": draw_id,
                "original_token_count": original_count,
                "final_token_count": final_count,
                "length_delta_tokens": delta,
                "absolute_length_delta_tokens": abs(delta),
                "human_token_count": int(metadata["human_token_count"]),
                "realized_contamination_ratio": (
                    float(metadata["human_token_count"]) / final_count
                    if final_count
                    else 0.0
                ),
                "tail_candidate_ids": metadata.get("tail_candidate_ids"),
            }
        )

    for index, row in enumerate(base, start=1):
        sample_id = str(row["sample_id"])
        original_ids = list(row["llm_token_ids"])
        human_spans = sentence_spans(tokenizer, row["human_continuation"])
        cached = tail_cache[sample_id]
        ranked = [
            TailCandidate(
                int(item["candidate_id"]),
                tuple(item["token_ids"]),
                float(item["nll"]),
            )
            for item in cached["ranked_candidates"]
        ]
        for ratio in ratios:
            for draw_id in range(random_draws):
                seed = corruption_seed(
                    int(config["contamination"]["corruption_seed"]),
                    str(config.get("dataset", row.get("dataset", "writingprompts"))),
                    str(config.get("target_model", row.get("target_model", ""))),
                    sample_id,
                    draw_id,
                )
                mixed, metadata = random_token_contamination(
                    original_ids, human_spans, ratio, seed
                )
                measure(
                    sample_id,
                    "random",
                    ratio,
                    draw_id,
                    mixed,
                    len(original_ids),
                    metadata,
                )
            mixed, metadata = tail_token_contamination(original_ids, ranked, ratio)
            measure(
                sample_id,
                "tail",
                ratio,
                0,
                mixed,
                len(original_ids),
                metadata,
            )
        if index % 100 == 0 or index == len(base):
            print(f"roundtrip_audit: sources={index}/{len(base)}", flush=True)

    condition_rows = []
    for (mode, ratio), deltas in sorted(conditions.items()):
        absolute = [abs(value) for value in deltas]
        condition_rows.append(
            {
                "contamination_mode": mode,
                "requested_contamination_ratio": ratio,
                "rows": len(deltas),
                "minimum_delta": min(deltas),
                "maximum_delta": max(deltas),
                "mean_absolute_delta": statistics.fmean(absolute),
                "p95_absolute_delta": _percentile(absolute, 0.95),
                "p99_absolute_delta": _percentile(absolute, 0.99),
                "maximum_absolute_delta": max(absolute),
                "rows_exceeding_tolerance": sum(
                    value > tolerance for value in absolute
                ),
            }
        )
    violations = [
        row for row in worst if row["absolute_length_delta_tokens"] > tolerance
    ]
    worst.sort(
        key=lambda row: (
            -int(row["absolute_length_delta_tokens"]),
            str(row["sample_id"]),
            str(row["contamination_mode"]),
            float(row["requested_contamination_ratio"]),
            int(row["corruption_draw_id"]),
        )
    )
    return {
        "audit_status": "complete",
        "source_rows": len(base),
        "constructed_rows_checked": len(worst),
        "configured_tolerance": tolerance,
        "condition_summary": condition_rows,
        "rows_exceeding_tolerance": len(violations),
        "unique_sources_exceeding_tolerance": len(
            {str(row["sample_id"]) for row in violations}
        ),
        "worst_cases": worst[: max(int(top_k), 0)],
    }


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace / config_path
    config = load_config(config_path)
    run_dir = workspace / config["paths"]["runs"] / args.run_id
    base_path = run_dir / "base_generations.jsonl"
    tail_path = run_dir / "tail_candidate_cache.jsonl"
    for path in (base_path, tail_path):
        if not path.is_file():
            raise FileNotFoundError(f"required cached artifact is missing: {path}")
    first = next(iter_jsonl(base_path))
    model_id = str(first["target_model"])
    if model_id != "Qwen/Qwen2.5-32B":
        raise ValueError(f"expected the Qwen-32B failed cell, found {model_id}")
    config["dataset"] = str(first["dataset"])
    config["target_model"] = model_id
    revision = first.get("target_tokenizer_resolved_revision")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("round-trip audit requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        use_fast=True,
    )
    report = audit_cached_constructions(
        iter_jsonl(base_path),
        iter_jsonl(tail_path),
        tokenizer,
        config,
        top_k=args.top_k,
    )
    output = (
        Path(args.output)
        if args.output
        else run_dir / "roundtrip_drift_audit.json"
    )
    if not output.is_absolute():
        output = workspace / output
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"roundtrip_audit_report={output}")


if __name__ == "__main__":
    main()
