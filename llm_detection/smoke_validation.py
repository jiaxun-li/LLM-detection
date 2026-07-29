"""Strict validation for the one-GPU Delta Qwen smoke workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .config import load_config, resolved_run_config, total_examples
from .data import data_row_key
from .evaluation import (
    actual_fpr,
    auroc,
    calibration_threshold,
    oriented_score,
    partial_auroc,
    tpr,
)
from .io import atomic_write_json, atomic_write_text, iter_jsonl


MODEL_ID = "Qwen/Qwen2.5-0.5B"
EXPECTED_DETECTORS = {
    "log_likelihood",
    "rank",
    "log_rank",
    "lrr",
    "entropy",
    "entropy_gap",
}
TOKEN_FEATURES = {
    "logp",
    "rank",
    "log_rank",
    "entropy",
    "top_k_token_ids",
    "top_k_logprobs",
    "top1_top2_logprob_margin",
    "target_top1_logprob_margin",
}
DEBUG_LABEL = "SMOKE_TEST_DEBUG_ONLY_NOT_FOR_SCIENTIFIC_USE"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _canonical_digest(values: Sequence[Any]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        hasher.update(payload.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _row_key_text(row: dict[str, Any]) -> list[Any]:
    return list(data_row_key(row))


def resume_snapshot(
    source_rows: Sequence[dict[str, Any]],
    base_rows: Sequence[dict[str, Any]],
    tail_rows: Sequence[dict[str, Any]],
    data_rows: Sequence[dict[str, Any]],
    score_rows: Sequence[dict[str, Any]],
    metrics_rows: Sequence[dict[str, str]],
) -> dict[str, Any]:
    """Content/key snapshot used to prove a second pass did not duplicate rows."""
    return {
        "source_count": len(source_rows),
        "source_key_digest": _canonical_digest(
            sorted(str(row["sample_id"]) for row in source_rows)
        ),
        "base_count": len(base_rows),
        "base_key_digest": _canonical_digest(
            sorted(str(row["sample_id"]) for row in base_rows)
        ),
        "tail_cache_count": len(tail_rows),
        "tail_cache_key_digest": _canonical_digest(
            sorted(str(row["sample_id"]) for row in tail_rows)
        ),
        "data_count": len(data_rows),
        "data_key_digest": _canonical_digest(
            sorted(_row_key_text(row) for row in data_rows)
        ),
        "score_count": len(score_rows),
        "score_key_digest": _canonical_digest(
            sorted(_row_key_text(row) for row in score_rows)
        ),
        "metrics_count": len(metrics_rows),
        "metrics_identity_digest": _canonical_digest(
            sorted(
                [
                    row["detector"],
                    row["aggregation"],
                    row["contamination_mode"],
                    row["requested_contamination_ratio"],
                    row["target_fpr"],
                    row["analysis"],
                ]
                for row in metrics_rows
            )
        ),
    }


def _materialize_views(run_dir: Path, data_rows: Sequence[dict[str, Any]]) -> None:
    clean = [row for row in data_rows if row["contamination_mode"] == "none"]
    contaminated = [
        row for row in data_rows if row["contamination_mode"] in {"random", "tail"}
    ]
    atomic_write_text(
        run_dir / "clean_data.jsonl",
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in clean),
    )
    atomic_write_text(
        run_dir / "contaminated_data.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in contaminated
        ),
    )


def _validate_calibration_thresholds(
    score_rows: Sequence[dict[str, Any]],
    metrics_rows: Sequence[dict[str, str]],
) -> int:
    calibration_human = [
        row
        for row in score_rows
        if row["split"] == "calibration"
        and row["label"] == "human"
        and row["contamination_mode"] == "none"
    ]
    checked: set[tuple[str, str, float, str]] = set()
    for metric in metrics_rows:
        identity = (
            metric["detector"],
            metric["aggregation"],
            float(metric["target_fpr"]),
            metric["clipping_specification"],
        )
        if identity in checked:
            continue
        checked.add(identity)
        detector, aggregation, target_fpr, serialized_spec = identity
        direction = int(metric["direction"])
        spec = json.loads(serialized_spec) if aggregation == "clipped" else {}
        scores = [
            oriented_score(row, detector, direction, spec)
            for row in calibration_human
        ]
        expected = calibration_threshold(scores, target_fpr)
        observed = float(metric["calibration_threshold"])
        _require(
            math.isclose(expected, observed, rel_tol=1e-12, abs_tol=1e-12),
            f"{detector}/{aggregation}/{target_fpr} threshold was not reproduced "
            "from calibration-human rows",
        )
    return len(checked)


def _validate_test_metrics(
    score_rows: Sequence[dict[str, Any]],
    metrics_rows: Sequence[dict[str, str]],
    partial_max_fpr: float,
) -> int:
    test_human = [
        row
        for row in score_rows
        if row["split"] == "test"
        and row["label"] == "human"
        and row["contamination_mode"] == "none"
    ]
    for metric in metrics_rows:
        ratio = float(metric["requested_contamination_ratio"])
        mode = metric["contamination_mode"]
        if ratio == 0.0:
            test_llm = [
                row
                for row in score_rows
                if row["split"] == "test"
                and row["label"] == "llm"
                and row["contamination_mode"] == "none"
            ]
        else:
            test_llm = [
                row
                for row in score_rows
                if row["split"] == "test"
                and row["label"] == "llm"
                and row["contamination_mode"] == mode
                and math.isclose(
                    float(row["requested_contamination_ratio"]), ratio
                )
            ]
        _require(
            len(test_human) == 4 and len(test_llm) == 4,
            f"test score selection is not 4 human/4 LLM for {mode}/{ratio}",
        )
        spec = (
            json.loads(metric["clipping_specification"])
            if metric["aggregation"] == "clipped"
            else {}
        )
        direction = int(metric["direction"])
        human_scores = [
            oriented_score(row, metric["detector"], direction, spec)
            for row in test_human
        ]
        llm_scores = [
            oriented_score(row, metric["detector"], direction, spec)
            for row in test_llm
        ]
        threshold = float(metric["calibration_threshold"])
        expected = {
            "actual_fpr": actual_fpr(human_scores, threshold),
            "tpr": tpr(llm_scores, threshold),
            "auroc": auroc(human_scores, llm_scores),
            "partial_auroc_0_5_fpr": partial_auroc(
                human_scores, llm_scores, partial_max_fpr
            ),
        }
        for field, value in expected.items():
            _require(
                math.isclose(
                    value,
                    float(metric[field]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                f"{field} was not reproduced from final-test rows for "
                f"{metric['detector']}/{metric['aggregation']}/{mode}/{ratio}",
            )
    return len(metrics_rows)


def validate_smoke_run(
    config_path: str | Path,
    run_dir: str | Path,
    *,
    report_path: str | Path | None = None,
    write_baseline: str | Path | None = None,
    compare_baseline: str | Path | None = None,
    write_completion_marker: bool = False,
    materialize_views: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    directory = Path(run_dir).resolve()
    manifest_path = directory / "manifest.json"
    base_path = directory / "base_generations.jsonl"
    tail_path = directory / "tail_candidate_cache.jsonl"
    data_path = directory / "data.jsonl"
    score_path = directory / "target_scores.jsonl"
    metrics_path = directory / "metrics.csv"
    _require(manifest_path.exists(), "manifest.json is missing")
    _require(base_path.exists(), "base_generations.jsonl is missing")
    _require(tail_path.exists(), "tail_candidate_cache.jsonl is missing")
    _require(data_path.exists(), "data.jsonl is missing")
    _require(score_path.exists(), "target_scores.jsonl is missing")
    _require(metrics_path.exists(), "metrics.csv is missing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_copy = directory / "selected_sources.jsonl"
    source_path = (
        source_copy
        if source_copy.exists()
        else Path(manifest["source_sample_manifest"])
    )
    _require(source_path.exists(), f"source manifest is missing: {source_path}")
    source_rows = list(iter_jsonl(source_path))
    base_rows = list(iter_jsonl(base_path))
    tail_rows = list(iter_jsonl(tail_path))
    data_rows = list(iter_jsonl(data_path))
    score_rows = list(iter_jsonl(score_path))
    metrics_rows = _read_csv(metrics_path)

    expected_total = 12
    _require(total_examples(config) == expected_total, "config total is not 12")
    _require(len(source_rows) == expected_total, "source manifest must have 12 rows")
    source_ids = [str(row["sample_id"]) for row in source_rows]
    _require(len(set(source_ids)) == expected_total, "source sample IDs are not distinct")
    split_sets = {
        split: {
            str(row["sample_id"])
            for row in source_rows
            if row["split"] == split
        }
        for split in ("clipping_tuning", "calibration", "test")
    }
    _require(
        {key: len(value) for key, value in split_sets.items()}
        == {"clipping_tuning": 4, "calibration": 4, "test": 4},
        "source split counts are not 4/4/4",
    )
    _require(
        not (split_sets["clipping_tuning"] & split_sets["calibration"])
        and not (split_sets["clipping_tuning"] & split_sets["test"])
        and not (split_sets["calibration"] & split_sets["test"]),
        "source splits overlap",
    )
    _require(
        all(row["dataset"] == "xsum" for row in source_rows),
        "source manifest contains a non-XSum row",
    )
    source_split = {
        str(row["sample_id"]): row["split"] for row in source_rows
    }
    for name, rows in (
        ("base generation", base_rows),
        ("tail cache", tail_rows),
    ):
        row_ids = [str(row["sample_id"]) for row in rows]
        _require(len(rows) == expected_total, f"{name} must have 12 rows")
        _require(
            len(set(row_ids)) == expected_total and set(row_ids) == set(source_ids),
            f"{name} sample IDs are duplicated or do not match the source manifest",
        )
    _require(
        all(row["target_model"] == MODEL_ID for row in base_rows),
        "base generation model provenance is incorrect",
    )

    expected_data_rows = 72
    _require(len(data_rows) == expected_data_rows, "prepared data must have 72 rows")
    data_keys = [data_row_key(row) for row in data_rows]
    _require(len(set(data_keys)) == len(data_keys), "prepared data has duplicate keys")
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data_rows:
        sample_id = str(row["sample_id"])
        by_sample[sample_id].append(row)
        _require(
            row["dataset"] == "xsum"
            and row["target_model"] == MODEL_ID
            and row["split"] == source_split.get(sample_id),
            f"data provenance/split mismatch for {sample_id}",
        )
        _require(
            "requested_contamination_ratio" in row
            and "realized_contamination_ratio" in row,
            "contamination ratio provenance is missing",
        )
    for sample_id in source_ids:
        rows = by_sample[sample_id]
        clean_human = [
            row
            for row in rows
            if row["label"] == "human" and row["contamination_mode"] == "none"
        ]
        clean_llm = [
            row
            for row in rows
            if row["label"] == "llm" and row["contamination_mode"] == "none"
        ]
        _require(len(clean_human) == 1, f"{sample_id} lacks one clean human row")
        _require(
            len(clean_llm) == 1,
            f"{sample_id} must have exactly one uncontaminated LLM baseline",
        )
        for mode in ("random", "tail"):
            for ratio in (0.2, 0.5):
                attacked = [
                    row
                    for row in rows
                    if row["contamination_mode"] == mode
                    and math.isclose(
                        float(row["requested_contamination_ratio"]), ratio
                    )
                ]
                _require(
                    len(attacked) == 1,
                    f"{sample_id} lacks exactly one {mode}/{ratio} row",
                )

    _require(len(score_rows) == expected_data_rows, "score file must have 72 rows")
    score_keys = [data_row_key(row) for row in score_rows]
    _require(len(set(score_keys)) == len(score_keys), "score file has duplicate keys")
    _require(set(score_keys) == set(data_keys), "score keys do not match data keys")
    for row in score_rows:
        _require(
            row.get("scoring_model") == MODEL_ID,
            "score row was not produced by the Qwen 0.5B target model",
        )
        features = row.get("token_features", {})
        _require(TOKEN_FEATURES <= set(features), "required token features are missing")
        lengths = {len(features[name]) for name in TOKEN_FEATURES}
        _require(len(lengths) == 1, "token-feature arrays have inconsistent lengths")
        feature_length = next(iter(lengths))
        _require(
            feature_length == int(row["num_scored_tokens"]) and feature_length > 0,
            "num_scored_tokens does not match token-feature arrays",
        )
        top_k = int(config["scoring"]["saved_top_k"])
        for token_index in range(feature_length):
            top_ids = features["top_k_token_ids"][token_index]
            top_logprobs = features["top_k_logprobs"][token_index]
            _require(
                len(top_ids) == top_k
                and len(set(top_ids)) == top_k
                and len(top_logprobs) == top_k,
                "top-k token features have an incorrect width or duplicate IDs",
            )
            _require(
                all(
                    math.isfinite(float(value))
                    for value in top_logprobs
                )
                and all(
                    float(top_logprobs[index])
                    >= float(top_logprobs[index + 1])
                    for index in range(top_k - 1)
                ),
                "top-k log-probabilities are non-finite or not sorted",
            )
            top_margin = float(
                features["top1_top2_logprob_margin"][token_index]
            )
            target_margin = float(
                features["target_top1_logprob_margin"][token_index]
            )
            _require(
                math.isclose(
                    top_margin,
                    float(top_logprobs[0]) - float(top_logprobs[1]),
                    rel_tol=1e-5,
                    abs_tol=1e-5,
                )
                and math.isclose(
                    target_margin,
                    float(features["logp"][token_index])
                    - float(top_logprobs[0]),
                    rel_tol=1e-5,
                    abs_tol=1e-5,
                )
                and top_margin >= -1e-6
                and target_margin <= 1e-6,
                "saved probability margins are inconsistent",
            )
        _require(
            row.get("scoring_model_revision")
            and row.get("scoring_tokenizer_revision")
            and row.get("scoring_feature_schema") == "target-token-features-v2",
            "score feature schema or resolved model/tokenizer revision is missing",
        )
        if config["scoring"]["save_mean_pooled_final_hidden_state"]:
            document_features = row.get("document_features", {})
            pooled = document_features.get(
                "mean_pooled_final_hidden_state", []
            )
            _require(
                pooled
                and len(pooled) == int(document_features.get("hidden_size", 0))
                and int(document_features.get("pooling_token_count", 0))
                == feature_length
                and all(math.isfinite(float(value)) for value in pooled),
                "mean-pooled final hidden state is missing or invalid",
            )
        _require(
            "lrr" in row["doc_scores"]
            and math.isfinite(float(row["doc_scores"]["lrr"])),
            "LRR is missing or non-finite",
        )

    expected_metric_rows = 144
    _require(len(metrics_rows) == expected_metric_rows, "metrics.csv must have 144 rows")
    _require(
        {row["detector"] for row in metrics_rows} == EXPECTED_DETECTORS,
        "metrics detector set is incomplete or contains Binoculars",
    )
    _require(
        {row["aggregation"] for row in metrics_rows} == {"raw", "clipped"},
        "raw/clipped metric rows are incomplete",
    )
    expected_metric_identities = {
        (
            detector,
            aggregation,
            mode,
            ratio,
            target_fpr,
            "primary_frozen_mixture",
        )
        for detector in EXPECTED_DETECTORS
        for aggregation in ("raw", "clipped")
        for mode in ("random", "tail")
        for ratio in (0.0, 0.2, 0.5)
        for target_fpr in (0.01, 0.05)
    }
    observed_metric_identities = [
        (
            row["detector"],
            row["aggregation"],
            row["contamination_mode"],
            float(row["requested_contamination_ratio"]),
            float(row["target_fpr"]),
            row["analysis"],
        )
        for row in metrics_rows
    ]
    _require(
        len(set(observed_metric_identities)) == len(observed_metric_identities)
        and set(observed_metric_identities) == expected_metric_identities,
        "metric cross-product is incomplete or contains duplicate identities",
    )
    _require(
        all(row["result_label"] == DEBUG_LABEL for row in metrics_rows)
        and all(row["debug_only"].lower() == "true" for row in metrics_rows),
        "metrics are not visibly labeled debug-only",
    )
    _require(
        all(
            row["dataset"] == "xsum"
            and row["model"] == MODEL_ID
            and row["score_source"] == "target_model_single_pass"
            for row in metrics_rows
        ),
        "metrics contain incorrect dataset/model/score-source provenance",
    )
    _require(
        all(row["split"] == "test" for row in metrics_rows)
        and all(int(row["n_test_human"]) == 4 for row in metrics_rows)
        and all(int(row["n_test_llm"]) == 4 for row in metrics_rows),
        "final metrics are not exclusively based on four test IDs",
    )
    _require(
        all(int(row["n_calibration_human"]) == 4 for row in metrics_rows),
        "metrics do not record four calibration humans",
    )
    threshold_checks = _validate_calibration_thresholds(score_rows, metrics_rows)
    test_metric_checks = _validate_test_metrics(
        score_rows,
        metrics_rows,
        float(config["evaluation"]["partial_auroc_max_fpr"]),
    )

    _require(manifest["target_model"] == MODEL_ID, "manifest target model is incorrect")
    _require(
        manifest.get("target_model_resolved_revision")
        and manifest.get("target_tokenizer_resolved_revision")
        and manifest.get("target_score_feature_schema")
        == "target-token-features-v2",
        "manifest lacks resolved scoring revisions or feature schema",
    )
    _require(
        manifest.get("dataset", {}).get("name") == "xsum",
        "manifest dataset is not XSum",
    )
    _require(manifest.get("debug_only") is True, "manifest is not debug-only")
    _require(manifest.get("result_label") == DEBUG_LABEL, "manifest debug label is missing")
    _require(
        manifest.get("completion_status") == "complete",
        "run manifest is not complete",
    )
    _require(
        manifest.get("explicitly_skipped_detectors") == ["binoculars"],
        "manifest does not record the intentional Binoculars skip",
    )
    for marker in (
        directory / "prepare.complete.json",
        directory / "score.complete.json",
        directory / "evaluate.complete.json",
    ):
        _require(marker.exists(), f"stage completion marker is missing: {marker.name}")

    if materialize_views:
        if not source_copy.exists():
            atomic_write_text(
                source_copy,
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in source_rows
                ),
            )
        _materialize_views(directory, data_rows)

    snapshot = resume_snapshot(
        source_rows,
        base_rows,
        tail_rows,
        data_rows,
        score_rows,
        metrics_rows,
    )
    if write_baseline is not None:
        atomic_write_json(write_baseline, snapshot)
    if compare_baseline is not None:
        baseline = json.loads(Path(compare_baseline).read_text(encoding="utf-8"))
        _require(
            snapshot == baseline,
            "resume snapshot changed; rows were duplicated, removed, or re-keyed",
        )

    report = {
        "status": "PASS",
        "label": DEBUG_LABEL,
        "run_id": manifest["run_id"],
        "run_dir": str(directory),
        "model": MODEL_ID,
        "source_rows": len(source_rows),
        "split_counts": {key: len(value) for key, value in split_sets.items()},
        "base_rows": len(base_rows),
        "tail_cache_rows": len(tail_rows),
        "data_rows": len(data_rows),
        "score_rows": len(score_rows),
        "metrics_rows": len(metrics_rows),
        "detectors": sorted(EXPECTED_DETECTORS),
        "calibration_threshold_checks": threshold_checks,
        "test_metric_checks": test_metric_checks,
        "resume_baseline_compared": compare_baseline is not None,
        "slurm_logs": (
            json.loads((directory / "slurm_logs.json").read_text(encoding="utf-8"))
            if (directory / "slurm_logs.json").exists()
            else None
        ),
    }
    if report_path is not None:
        atomic_write_json(report_path, report)
    if write_completion_marker:
        _require(
            compare_baseline is not None,
            "final smoke completion requires a resume-baseline comparison",
        )
        atomic_write_json(
            directory / "smoke_validation.complete.json",
            {
                "status": "complete",
                "label": DEBUG_LABEL,
                "validation_report": str(
                    Path(report_path).resolve()
                    if report_path is not None
                    else directory / "validation_report.json"
                ),
                "resume_validated": True,
            },
        )
    return report
