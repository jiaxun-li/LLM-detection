from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from llm_detection.config import load_config, resolved_run_config
from llm_detection.evaluation import (
    auroc,
    calibration_threshold,
    evaluate,
    partial_auroc,
)
from llm_detection.io import append_jsonl


def feature_row(
    config,
    sample_id: str,
    split: str,
    label: str,
    quality: float,
    mode: str = "none",
    ratio: float = 0.0,
):
    # Higher quality is more machine-like for every synthetic detector.
    logp = [-2.0 + quality, -1.5 + quality]
    rank = [max(1.0, 8.0 - 4.0 * quality)] * 2
    log_rank = [max(0.05, 2.0 - quality)] * 2
    entropy = [max(0.1, 2.5 - quality)] * 2
    nll = -sum(logp) / 2
    mean_log_rank = sum(log_rank) / 2
    features = {
        "logp": logp,
        "rank": rank,
        "log_rank": log_rank,
        "entropy": entropy,
    }
    return {
        "run_id": config["run_id"],
        "dataset": config["dataset"],
        "target_model": config["target_model"],
        "sample_id": sample_id,
        "split": split,
        "generation_seed": 101,
        "label": label,
        "contamination_mode": mode,
        "requested_contamination_ratio": ratio,
        "realized_contamination_ratio": ratio,
        "corruption_draw_id": 0,
        "doc_scores": {
            "log_likelihood": sum(logp) / 2,
            "rank": sum(rank) / 2,
            "log_rank": mean_log_rank,
            "lrr": nll / mean_log_rank,
            "entropy": sum(entropy) / 2,
            "entropy_gap": nll - sum(entropy) / 2,
        },
        "token_features": features,
    }


def binoculars_row(row):
    copied = {key: value for key, value in row.items() if key not in {"doc_scores", "token_features"}}
    quality = row["doc_scores"]["log_likelihood"] + 1.75
    gaps = [quality, quality]
    copied["token_features"] = {
        "performer_nll": [2.0 + quality, 2.0 + quality],
        "observer_to_performer_cross_entropy": [2.0, 2.0],
    }
    copied["doc_scores"] = {"binoculars": float(__import__("math").exp(quality))}
    return copied


class EvaluationProtocolTests(unittest.TestCase):
    def test_metric_primitives(self) -> None:
        self.assertEqual(auroc([0, 1], [2, 3]), 1.0)
        self.assertEqual(auroc([0, 1], [1, 2]), 0.875)
        self.assertGreater(partial_auroc([0, 1], [2, 3], 0.05), 0.0)
        threshold = calibration_threshold([0, 1, 2, 3], 0.01)
        self.assertGreater(threshold, 3)

    def test_independent_protocol_outputs_all_detectors(self) -> None:
        base = load_config("configs/smoke.json")
        base["evaluation"]["bootstrap_repetitions"] = 4
        config = resolved_run_config(
            base, "xsum", "Qwen/Qwen2.5-0.5B", "test-run"
        )
        target_rows = []
        split_counts = {"clipping_tuning": 4, "calibration": 4, "test": 8}
        index = 0
        for split, count in split_counts.items():
            for local in range(count):
                sample_id = f"id-{index}"
                index += 1
                human_quality = -0.2 + local * 0.01
                machine_quality = 0.8 - local * 0.01
                target_rows.append(
                    feature_row(config, sample_id, split, "human", human_quality)
                )
                target_rows.append(
                    feature_row(config, sample_id, split, "llm", machine_quality)
                )
                for mode in ("random", "tail"):
                    for ratio in (0.1, 0.5):
                        attacked_quality = machine_quality - ratio * (
                            0.7 if mode == "random" else 1.0
                        )
                        target_rows.append(
                            feature_row(
                                config,
                                sample_id,
                                split,
                                "llm",
                                attacked_quality,
                                mode,
                                ratio,
                            )
                        )

        with tempfile.TemporaryDirectory() as temporary:
            target_path = Path(temporary) / "target.jsonl"
            binoculars_path = Path(temporary) / "binoculars.jsonl"
            output = Path(temporary) / "metrics.csv"
            for row in target_rows:
                append_jsonl(target_path, row)
                append_jsonl(binoculars_path, binoculars_row(row))
            metrics = evaluate(target_path, output, config, binoculars_path)

        detectors = {row["detector"] for row in metrics}
        self.assertEqual(
            detectors,
            {
                "log_likelihood",
                "rank",
                "log_rank",
                "lrr",
                "entropy",
                "entropy_gap",
                "binoculars",
            },
        )
        self.assertTrue(all(row["split"] == "test" for row in metrics))
        primary_clipped = [
            row
            for row in metrics
            if row["detector"] == "log_likelihood"
            and row["aggregation"] == "clipped"
        ]
        self.assertEqual(
            len({row["clipping_specification"] for row in primary_clipped}), 1
        )
        self.assertEqual(
            {row["contamination_mode"] for row in primary_clipped},
            {"random", "tail"},
        )
        self.assertTrue(
            all(row["n_calibration_human"] == 4 for row in primary_clipped)
        )
        self.assertTrue(
            all(row["analysis"] == "primary_frozen_mixture" for row in metrics)
        )


if __name__ == "__main__":
    unittest.main()
