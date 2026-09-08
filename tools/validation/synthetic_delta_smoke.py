#!/usr/bin/env python3
"""No-network synthetic dry run of Delta smoke orchestration and validation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment_core.infrastructure.config import load_config, resolved_run_config
from experiment_core.preparation.data import data_row_key
from experiment_core.analysis.evaluation import evaluate
from experiment_core.infrastructure.io import (
    AppendSafeJsonlWriter,
    atomic_write_json,
    atomic_write_text,
    completed_keys,
)
from experiment_core.detectors.scoring import single_model_doc_scores
from experiment_core.validation.smoke_validation import DEBUG_LABEL, MODEL_ID, validate_smoke_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/delta_smoke_qwen_0_5b.json"
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _resume_append(
    path: Path,
    rows: Sequence[dict[str, Any]],
    key_fn: Any = data_row_key,
) -> None:
    done = completed_keys(path, key_fn)
    with AppendSafeJsonlWriter(path, checkpoint_interval=2) as writer:
        for row in rows:
            if key_fn(row) not in done:
                writer.write(row)


def _data_row(
    run_config: dict[str, Any],
    sample_id: str,
    split: str,
    label: str,
    mode: str,
    ratio: float,
) -> dict[str, Any]:
    return {
        "run_id": run_config["run_id"],
        "dataset": "xsum",
        "dataset_id": "synthetic/xsum",
        "dataset_config": None,
        "dataset_revision": "synthetic-dataset-revision",
        "dataset_resolved_revision": "synthetic-dataset-revision",
        "source_id": sample_id,
        "sample_id": sample_id,
        "split": split,
        "generation_seed": 101,
        "target_model": MODEL_ID,
        "target_model_revision": None,
        "target_model_resolved_revision": "synthetic-model-revision",
        "target_tokenizer_revision": None,
        "target_tokenizer_resolved_revision": "synthetic-tokenizer-revision",
        "prompt": "synthetic prompt",
        "text": f"synthetic {label} {mode} {ratio} continuation",
        "label": label,
        "contamination_mode": mode,
        "corruption_seed": 99173 if mode == "random" else None,
        "corruption_draw_id": 0,
        "requested_contamination_ratio": ratio,
        "realized_contamination_ratio": ratio,
        "original_token_count": 8,
        "human_token_count": int(round(8 * ratio)),
        "replaced_token_count": int(round(8 * ratio)),
        "final_token_count": 8,
        "length_delta_tokens": 0,
    }


def _score_row(row: dict[str, Any], local_index: int) -> dict[str, Any]:
    if row["label"] == "human":
        quality = -0.3 + local_index * 0.01
    else:
        quality = 0.9 - float(row["requested_contamination_ratio"]) * (
            0.8 if row["contamination_mode"] == "random" else 1.1
        )
        quality -= local_index * 0.01
    logp = [-2.4 + quality + offset * 0.01 for offset in range(8)]
    rank_value = max(1, int(round(10 - 5 * quality)))
    rank = [rank_value] * 8
    log_rank = [math.log(rank_value)] * 8
    entropy = [max(0.1, 2.8 - quality + offset * 0.005) for offset in range(8)]
    top_k_token_ids = [
        [100 + token_index * 10 + rank for rank in range(10)]
        for token_index in range(8)
    ]
    top_k_logprobs = [
        [logp[token_index] + 0.2 - rank * 0.1 for rank in range(10)]
        for token_index in range(8)
    ]
    features = {
        "logp": logp,
        "rank": rank,
        "log_rank": log_rank,
        "entropy": entropy,
        "top_k_token_ids": top_k_token_ids,
        "top_k_logprobs": top_k_logprobs,
        "top1_top2_logprob_margin": [0.1] * 8,
        "target_top1_logprob_margin": [-0.2] * 8,
    }
    return {
        **row,
        "scoring_model": MODEL_ID,
        "scoring_model_revision": "synthetic-model-revision",
        "scoring_tokenizer_revision": "synthetic-tokenizer-revision",
        "scoring_feature_schema": "target-token-features-v2",
        "num_scored_tokens": 8,
        "token_features": features,
        "document_features": {
            "mean_pooled_final_hidden_state": [
                quality,
                quality + 0.1,
                quality + 0.2,
                quality + 0.3,
            ],
            "pooling_token_count": 8,
            "hidden_size": 4,
        },
        "doc_scores": single_model_doc_scores(features),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"synthetic output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    run_config = resolved_run_config(
        config, "xsum", MODEL_ID, "synthetic-delta-smoke"
    )

    split_names = ["clipping_tuning"] * 4 + ["calibration"] * 4 + ["test"] * 4
    sources = [
        {
            "dataset": "xsum",
            "dataset_id": "synthetic/xsum",
            "dataset_split": "validation",
            "dataset_resolved_revision": "synthetic-dataset-revision",
            "source_id": f"source-{index:02d}",
            "sample_id": f"source-{index:02d}",
            "split": split,
            "generation_seed": 101,
            "source_text": f"synthetic source document {index}",
        }
        for index, split in enumerate(split_names)
    ]
    source_path = output / "selected_sources.jsonl"
    _write_jsonl(source_path, sources)
    base_rows = [
        {
            **source,
            "target_model": MODEL_ID,
            "prompt": "synthetic prompt",
            "human_continuation": "synthetic human continuation",
            "llm_continuation": "synthetic llm continuation",
            "continuation_token_count": 8,
        }
        for source in sources
    ]
    tail_rows = [
        {
            "dataset": "xsum",
            "sample_id": source["sample_id"],
            "target_model": MODEL_ID,
            "ranked_candidate_spans": [],
        }
        for source in sources
    ]
    base_path = output / "base_generations.jsonl"
    tail_path = output / "tail_candidate_cache.jsonl"
    _write_jsonl(base_path, base_rows)
    _write_jsonl(tail_path, tail_rows)

    data_rows: list[dict[str, Any]] = []
    for source in sources:
        sample_id = source["sample_id"]
        split = source["split"]
        data_rows.append(
            _data_row(run_config, sample_id, split, "human", "none", 0.0)
        )
        data_rows.append(
            _data_row(run_config, sample_id, split, "llm", "none", 0.0)
        )
        for mode in ("random", "tail"):
            for ratio in (0.2, 0.5):
                data_rows.append(
                    _data_row(run_config, sample_id, split, "llm", mode, ratio)
                )
    score_rows = [
        _score_row(row, index % 4) for index, row in enumerate(data_rows)
    ]
    data_path = output / "data.jsonl"
    score_path = output / "target_scores.jsonl"
    _write_jsonl(data_path, data_rows)
    _write_jsonl(score_path, score_rows)

    metrics_path = output / "metrics.csv"
    evaluate(score_path, metrics_path, run_config)
    config_snapshot = output / "config.snapshot.json"
    atomic_write_text(
        config_snapshot, Path(args.config).read_text(encoding="utf-8")
    )
    manifest = {
        "run_id": run_config["run_id"],
        "experiment_name": run_config["experiment_name"],
        "result_label": DEBUG_LABEL,
        "debug_only": True,
        "dataset": {"name": "xsum"},
        "target_model": MODEL_ID,
        "target_model_resolved_revision": "synthetic-model-revision",
        "target_tokenizer_resolved_revision": "synthetic-tokenizer-revision",
        "target_score_feature_schema": "target-token-features-v2",
        "source_sample_manifest": str(source_path),
        "completion_status": "complete",
        "completed_stages": ["prepare", "score", "evaluate"],
        "explicitly_skipped_detectors": ["binoculars"],
        "outputs": {
            "run_directory": str(output),
            "results_directory": str(output),
            "data": str(data_path),
            "base_generations": str(base_path),
            "tail_candidate_cache": str(tail_path),
            "target_scores": str(score_path),
            "metrics_csv": str(metrics_path),
        },
    }
    atomic_write_json(output / "manifest.json", manifest)
    for name in ("prepare", "score", "evaluate"):
        atomic_write_json(
            output / f"{name}.complete.json",
            {"status": "complete", "synthetic": True},
        )

    baseline = output / "resume_baseline.json"
    validate_smoke_run(
        args.config,
        output,
        report_path=output / "validation_report.first_pass.json",
        write_baseline=baseline,
        materialize_views=True,
    )
    # Simulate the append-safe resume path. All keys are complete, so no writes
    # occur and the second evaluation atomically replaces, rather than appends.
    sample_key = lambda row: (str(row["sample_id"]),)
    _resume_append(base_path, base_rows, sample_key)
    _resume_append(tail_path, tail_rows, sample_key)
    _resume_append(data_path, data_rows)
    _resume_append(score_path, score_rows)
    evaluate(score_path, metrics_path, run_config)
    report = validate_smoke_run(
        args.config,
        output,
        report_path=output / "validation_report.json",
        compare_baseline=baseline,
        write_completion_marker=True,
        materialize_views=True,
    )
    print("SYNTHETIC DELTA SMOKE DRY RUN: PASS")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
