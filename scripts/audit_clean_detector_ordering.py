#!/usr/bin/env python3
"""Audit clean LogP/Rank/LogRank behavior for a completed experiment run.

This is a read-only audit of existing ``target_scores.jsonl`` and
``metrics.csv`` artifacts.  It performs no model loading, generation, scoring,
or evaluation rerun.  The optional JSON report is the only file it writes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from llm_detection.evaluation import (
    actual_fpr,
    auroc,
    calibration_threshold,
    partial_auroc,
    roc_points,
    tpr,
)


DETECTORS = ("log_likelihood", "rank", "log_rank")
SPLITS = ("clipping_tuning", "calibration", "test")
LABELS = ("human", "llm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit clean raw LogP/Rank/LogRank ordering without loading a model"
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace containing runs/ and results/ (default: current directory)",
    )
    parser.add_argument(
        "--scores",
        help="Override target_scores.jsonl path",
    )
    parser.add_argument(
        "--metrics",
        help="Override metrics.csv path",
    )
    parser.add_argument(
        "--output",
        help=(
            "JSON report path (default: results/<run-id>/"
            "clean_detector_audit.json)"
        ),
    )
    return parser.parse_args()


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0}
    quantiles = np.quantile(array, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "min": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p05": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": float(quantiles[6]),
    }


def _oracle_tpr_at_fpr(
    human_scores: Sequence[float],
    llm_scores: Sequence[float],
    target_fpr: float,
) -> float:
    """Diagnostic test-set ROC point; not the released calibrated metric."""
    fpr_values, tpr_values = roc_points(human_scores, llm_scores)
    allowed = tpr_values[fpr_values <= target_fpr + 1e-15]
    return float(allowed.max()) if len(allowed) else float("nan")


def _rank_token_summary(chunks: list[np.ndarray]) -> dict[str, float | int]:
    if not chunks:
        return {"n": 0}
    ranks = np.concatenate(chunks).astype(np.float64, copy=False)
    ordered = np.sort(ranks)
    top_count = max(1, int(math.ceil(0.01 * len(ordered))))
    total = float(ordered.sum())
    return {
        **_summary(ranks),
        "fraction_rank_1": float(np.mean(ranks == 1)),
        "fraction_rank_le_10": float(np.mean(ranks <= 10)),
        "fraction_rank_gt_1000": float(np.mean(ranks > 1000)),
        "largest_1pct_share_of_rank_sum": (
            float(ordered[-top_count:].sum() / total) if total else 0.0
        ),
    }


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _load_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    run_dir = workspace / "runs" / args.run_id
    result_dir = workspace / "results" / args.run_id
    score_path = (
        Path(args.scores).expanduser().resolve()
        if args.scores
        else run_dir / "target_scores.jsonl"
    )
    metrics_path = (
        Path(args.metrics).expanduser().resolve()
        if args.metrics
        else result_dir / "metrics.csv"
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else result_dir / "clean_detector_audit.json"
    )

    if not score_path.exists():
        raise SystemExit(f"target score file not found: {score_path}")

    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    doc_scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    test_rank_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    test_logp_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    test_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    schemas: set[str] = set()
    model_ids: set[str] = set()
    model_revisions: set[str] = set()
    tokenizer_revisions: set[str] = set()
    issues: list[str] = []
    clean_rows = 0
    checked_tokens = 0
    doc_formula_failures = 0
    log_rank_failures = 0
    rank_domain_failures = 0
    margin_failures = 0

    with score_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("contamination_mode") != "none":
                continue
            split = str(row.get("split"))
            label = str(row.get("label"))
            if split not in SPLITS or label not in LABELS:
                issues.append(
                    f"unexpected clean row split/label at line {line_number}: "
                    f"{split}/{label}"
                )
                continue

            clean_rows += 1
            schemas.add(str(row.get("scoring_feature_schema")))
            model_ids.add(str(row.get("scoring_model")))
            model_revisions.add(str(row.get("scoring_model_revision")))
            tokenizer_revisions.add(str(row.get("scoring_tokenizer_revision")))

            features = row.get("token_features", {})
            scores = row.get("doc_scores", {})
            logp_values = np.asarray(features.get("logp", []), dtype=np.float64)
            rank_values = np.asarray(features.get("rank", []), dtype=np.float64)
            log_rank_values = np.asarray(
                features.get("log_rank", []), dtype=np.float64
            )
            lengths = {len(logp_values), len(rank_values), len(log_rank_values)}
            if len(lengths) != 1 or not len(rank_values):
                issues.append(
                    f"token feature length mismatch/empty row at line {line_number}"
                )
                continue
            if int(row.get("num_scored_tokens", -1)) != len(rank_values):
                issues.append(f"num_scored_tokens mismatch at line {line_number}")

            checked_tokens += len(rank_values)
            if np.any(rank_values < 1) or np.any(rank_values != np.floor(rank_values)):
                rank_domain_failures += 1
            if not np.allclose(
                log_rank_values,
                np.log(rank_values),
                rtol=2e-6,
                atol=2e-6,
            ):
                log_rank_failures += 1

            expected_docs = {
                "log_likelihood": float(logp_values.mean()),
                "rank": float(rank_values.mean()),
                "log_rank": float(log_rank_values.mean()),
            }
            for detector, expected in expected_docs.items():
                observed = float(scores[detector])
                if not math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-10):
                    doc_formula_failures += 1
                doc_scores[split][label][detector].append(observed)

            margins = np.asarray(
                features.get("target_top1_logprob_margin", []), dtype=np.float64
            )
            if len(margins) == len(rank_values):
                bad_top = (rank_values == 1) & (np.abs(margins) > 2e-5)
                bad_non_top = (rank_values > 1) & (margins >= 0)
                if np.any(bad_top | bad_non_top):
                    margin_failures += 1

            if split == "test":
                test_rank_chunks[label].append(rank_values)
                test_logp_chunks[label].append(logp_values)
                test_records[label].append(
                    {
                        "sample_id": str(row.get("sample_id")),
                        **expected_docs,
                    }
                )

    if doc_formula_failures:
        issues.append(
            f"{doc_formula_failures} document scores disagree with token means"
        )
    if log_rank_failures:
        issues.append(f"{log_rank_failures} rows violate log_rank == log(rank)")
    if rank_domain_failures:
        issues.append(f"{rank_domain_failures} rows contain invalid rank values")
    if margin_failures:
        issues.append(
            f"{margin_failures} rows disagree between rank and target/top-1 margin"
        )
    for name, values in (
        ("schemas", schemas),
        ("model IDs", model_ids),
        ("model revisions", model_revisions),
        ("tokenizer revisions", tokenizer_revisions),
    ):
        if len(values) != 1:
            issues.append(f"expected exactly one {name}, found {sorted(values)}")

    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts[split] = {
            label: len(doc_scores[split][label]["rank"]) for label in LABELS
        }
        if counts[split]["human"] != counts[split]["llm"]:
            issues.append(
                f"clean human/LLM count mismatch in {split}: {counts[split]}"
            )
    expected_full_counts = {
        "clipping_tuning": {"human": 500, "llm": 500},
        "calibration": {"human": 500, "llm": 500},
        "test": {"human": 2000, "llm": 2000},
    }
    if counts != expected_full_counts:
        issues.append(
            f"clean split counts differ from the full paper configuration: {counts}"
        )
    if not manifest:
        issues.append(f"manifest is missing: {manifest_path}")
    else:
        if manifest.get("completion_status") != "complete":
            issues.append(
                "manifest completion_status is not complete: "
                f"{manifest.get('completion_status')!r}"
            )
        completed_stages = set(manifest.get("completed_stages", []))
        required_stages = {"prepare", "score", "evaluate"}
        if not required_stages.issubset(completed_stages):
            issues.append(
                "manifest is missing completed stages: "
                f"{sorted(required_stages - completed_stages)}"
            )

    detector_results: dict[str, dict[str, Any]] = {}
    for detector in DETECTORS:
        tuning_human = np.asarray(
            doc_scores["clipping_tuning"]["human"][detector], dtype=np.float64
        )
        tuning_llm = np.asarray(
            doc_scores["clipping_tuning"]["llm"][detector], dtype=np.float64
        )
        direction = 1 if float(tuning_llm.mean()) >= float(tuning_human.mean()) else -1
        calibration_human = direction * np.asarray(
            doc_scores["calibration"]["human"][detector], dtype=np.float64
        )
        test_human = direction * np.asarray(
            doc_scores["test"]["human"][detector], dtype=np.float64
        )
        test_llm = direction * np.asarray(
            doc_scores["test"]["llm"][detector], dtype=np.float64
        )

        result: dict[str, Any] = {
            "direction": direction,
            "tuning_human_mean_raw": float(tuning_human.mean()),
            "tuning_llm_mean_raw": float(tuning_llm.mean()),
            "test_human_raw": _summary(
                doc_scores["test"]["human"][detector]
            ),
            "test_llm_raw": _summary(doc_scores["test"]["llm"][detector]),
            "auroc": auroc(test_human, test_llm),
            "partial_auroc_0_05": partial_auroc(test_human, test_llm, 0.05),
            "fpr_targets": {},
        }
        for target_fpr in (0.01, 0.05):
            threshold = calibration_threshold(calibration_human, target_fpr)
            result["fpr_targets"][str(target_fpr)] = {
                "calibration_threshold_oriented": threshold,
                "actual_test_fpr": actual_fpr(test_human, threshold),
                "calibrated_test_tpr": tpr(test_llm, threshold),
                "diagnostic_test_roc_tpr": _oracle_tpr_at_fpr(
                    test_human, test_llm, target_fpr
                ),
            }
        detector_results[detector] = result

    metric_rows = _load_metrics(metrics_path)
    metric_comparison: dict[str, dict[str, Any]] = {}
    if not metric_rows:
        issues.append(f"metrics file not found or empty: {metrics_path}")
    else:
        for detector in DETECTORS:
            metric_comparison[detector] = {}
            for target_fpr in (0.01, 0.05):
                matches = [
                    row
                    for row in metric_rows
                    if row.get("analysis") == "primary_frozen_mixture"
                    and row.get("aggregation") == "raw"
                    and row.get("detector") == detector
                    and abs(float(row["requested_contamination_ratio"])) < 1e-15
                    and abs(float(row["target_fpr"]) - target_fpr) < 1e-15
                ]
                distinct = {
                    (
                        int(row["direction"]),
                        float(row["actual_fpr"]),
                        float(row["tpr"]),
                        float(row["auroc"]),
                        float(row["partial_auroc_0_5_fpr"]),
                    )
                    for row in matches
                }
                recomputed = detector_results[detector]
                expected = (
                    recomputed["direction"],
                    recomputed["fpr_targets"][str(target_fpr)]["actual_test_fpr"],
                    recomputed["fpr_targets"][str(target_fpr)][
                        "calibrated_test_tpr"
                    ],
                    recomputed["auroc"],
                    recomputed["partial_auroc_0_05"],
                )
                agrees = len(distinct) == 1 and all(
                    math.isclose(observed, wanted, rel_tol=1e-12, abs_tol=1e-12)
                    for observed, wanted in zip(next(iter(distinct)), expected)
                )
                metric_comparison[detector][str(target_fpr)] = {
                    "matching_csv_rows": len(matches),
                    "distinct_csv_values": [list(item) for item in sorted(distinct)],
                    "recomputed_values": list(expected),
                    "agrees": agrees,
                }
                if not agrees:
                    issues.append(
                        f"metrics.csv disagreement for {detector} at FPR={target_fpr}"
                    )

    scientific_flags = {
        "rank_auroc_exceeds_log_likelihood": (
            detector_results["rank"]["auroc"]
            > detector_results["log_likelihood"]["auroc"]
        ),
        "rank_calibrated_tpr_1pct_exceeds_log_likelihood": (
            detector_results["rank"]["fpr_targets"]["0.01"][
                "calibrated_test_tpr"
            ]
            > detector_results["log_likelihood"]["fpr_targets"]["0.01"][
                "calibrated_test_tpr"
            ]
        ),
        "rank_calibrated_tpr_5pct_exceeds_log_likelihood": (
            detector_results["rank"]["fpr_targets"]["0.05"][
                "calibrated_test_tpr"
            ]
            > detector_results["log_likelihood"]["fpr_targets"]["0.05"][
                "calibrated_test_tpr"
            ]
        ),
    }

    token_diagnostics: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        logp_tokens = (
            np.concatenate(test_logp_chunks[label])
            if test_logp_chunks[label]
            else np.asarray([], dtype=np.float64)
        )
        token_diagnostics[label] = {
            "rank": _rank_token_summary(test_rank_chunks[label]),
            "logp": _summary(logp_tokens),
        }

    extreme_llm_docs = sorted(
        test_records["llm"], key=lambda row: float(row["rank"]), reverse=True
    )[:10]

    report = {
        "run_id": args.run_id,
        "workspace": str(workspace),
        "target_scores": str(score_path),
        "metrics": str(metrics_path),
        "manifest": str(manifest_path),
        "manifest_completion_status": manifest.get("completion_status"),
        "manifest_completed_stages": manifest.get("completed_stages", []),
        "clean_rows": clean_rows,
        "checked_tokens": checked_tokens,
        "counts": counts,
        "scoring_feature_schema": sorted(schemas),
        "scoring_model": sorted(model_ids),
        "scoring_model_revision": sorted(model_revisions),
        "scoring_tokenizer_revision": sorted(tokenizer_revisions),
        "detectors": detector_results,
        "token_diagnostics": token_diagnostics,
        "highest_mean_rank_clean_llm_documents": extreme_llm_docs,
        "metrics_csv_comparison": metric_comparison,
        "scientific_flags": scientific_flags,
        "issues": issues,
        "audit_status": "PASS" if not issues else "REVIEW",
        "limitations": [
            "This CPU audit validates stored-feature identities and reproduces "
            "released metrics, but it does not rerun Qwen logits.",
            "The diagnostic test-ROC TPR chooses a threshold on test data and must "
            "not replace the leakage-safe calibrated result.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("=== Clean detector ordering audit ===")
    print(f"run_id: {args.run_id}")
    print(f"scores: {score_path}")
    print(f"clean rows: {clean_rows}; checked tokens: {checked_tokens}")
    print(f"counts: {counts}")
    print()
    print(
        "detector          dir   AUROC  pAUROC5  "
        "calTPR@1  FPR@1  rocTPR@1  calTPR@5  FPR@5  rocTPR@5"
    )
    for detector in DETECTORS:
        result = detector_results[detector]
        one = result["fpr_targets"]["0.01"]
        five = result["fpr_targets"]["0.05"]
        print(
            f"{detector:<17} {result['direction']:>3d} "
            f"{result['auroc']:>7.4f} {result['partial_auroc_0_05']:>8.4f} "
            f"{one['calibrated_test_tpr']:>9.4f} "
            f"{one['actual_test_fpr']:>6.4f} "
            f"{one['diagnostic_test_roc_tpr']:>9.4f} "
            f"{five['calibrated_test_tpr']:>9.4f} "
            f"{five['actual_test_fpr']:>6.4f} "
            f"{five['diagnostic_test_roc_tpr']:>9.4f}"
        )

    print("\nRaw document-score summaries (test split):")
    for detector in DETECTORS:
        print(f"  {detector}")
        for label in LABELS:
            summary = detector_results[detector][f"test_{label}_raw"]
            print(
                f"    {label:<5} mean={_fmt(summary.get('mean'))} "
                f"median={_fmt(summary.get('median'))} "
                f"p95={_fmt(summary.get('p95'))} "
                f"p99={_fmt(summary.get('p99'))} "
                f"max={_fmt(summary.get('max'))}"
            )

    print("\nToken-rank tail diagnostics (test split):")
    for label in LABELS:
        summary = token_diagnostics[label]["rank"]
        print(
            f"  {label:<5} n={summary['n']} mean={_fmt(summary.get('mean'))} "
            f"median={_fmt(summary.get('median'))} "
            f"p99={_fmt(summary.get('p99'))} max={_fmt(summary.get('max'))} "
            f"rank1={_fmt(summary.get('fraction_rank_1'))} "
            f"rank<=10={_fmt(summary.get('fraction_rank_le_10'))} "
            f"rank>1000={_fmt(summary.get('fraction_rank_gt_1000'))} "
            f"top1%-sum-share={_fmt(summary.get('largest_1pct_share_of_rank_sum'))}"
        )

    print("\nScientific ordering flags:")
    for name, active in scientific_flags.items():
        print(f"  {name}: {active}")

    print(f"\naudit_status: {report['audit_status']}")
    if issues:
        for issue in issues:
            print(f"  REVIEW: {issue}")
    else:
        print("  Stored token features, document means, and metrics.csv agree.")
    print(f"report: {output_path}")
    print(
        "Note: rocTPR columns choose a threshold on the test ROC and are diagnostic "
        "only; calTPR columns are the leakage-safe released metrics."
    )


if __name__ == "__main__":
    main()
