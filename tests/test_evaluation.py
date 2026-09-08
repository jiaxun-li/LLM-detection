from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiment_core.infrastructure.config import load_config, resolved_run_config
from experiment_core.analysis.evaluation import (
    _bootstrap_index_values,
    _metric_bootstrap,
    _robustness_bootstrap,
    _selected_source_indices,
    actual_fpr,
    auroc,
    calibration_threshold,
    compact_evaluation_row,
    evaluate,
    orientation,
    oriented_score,
    partial_auroc,
    percentile_interval,
    precompute_evaluation_scores,
    robustness_auc,
    tpr,
    tune_clipping_spec,
)
from experiment_core.infrastructure.io import append_jsonl


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
    copied["doc_scores"] = {"binoculars": float(math.exp(quality))}
    return copied


def protocol_rows(config):
    rows = []
    split_counts = {"clipping_tuning": 4, "calibration": 4, "test": 8}
    index = 0
    for split, count in split_counts.items():
        for local in range(count):
            sample_id = f"id-{index}"
            index += 1
            human_quality = -0.2 + local * 0.01
            machine_quality = 0.8 - local * 0.01
            rows.append(
                feature_row(config, sample_id, split, "human", human_quality)
            )
            rows.append(
                feature_row(config, sample_id, split, "llm", machine_quality)
            )
            for mode in ("random", "tail"):
                for ratio in (0.1, 0.5):
                    attacked_quality = machine_quality - ratio * (
                        0.7 if mode == "random" else 1.0
                    )
                    rows.append(
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
    return rows


class EvaluationProtocolTests(unittest.TestCase):
    def test_compact_evaluation_row_discards_saved_feature_pack_bulk(self) -> None:
        row = {
            "dataset": "xsum",
            "target_model": "model",
            "sample_id": "sample",
            "split": "test",
            "label": "llm",
            "contamination_mode": "none",
            "requested_contamination_ratio": 0.0,
            "text": "large scored document",
            "prompt": "large generation prompt",
            "doc_scores": {"log_likelihood": -1.0},
            "token_features": {
                "logp": [-1.0, -2.0],
                "rank": [1, 2],
                "log_rank": [0.0, 0.7],
                "entropy": [1.5, 1.6],
                "top_k_token_ids": [[1, 2], [3, 4]],
                "top_k_logprobs": [[-0.1, -0.2], [-0.3, -0.4]],
                "top1_top2_logprob_margin": [0.1, 0.1],
                "target_top1_logprob_margin": [-0.9, -1.7],
            },
            "document_features": {
                "mean_pooled_final_hidden_state": [0.1] * 16
            },
        }

        compact = compact_evaluation_row(row)

        self.assertNotIn("text", compact)
        self.assertNotIn("prompt", compact)
        self.assertNotIn("document_features", compact)
        self.assertEqual(
            set(compact["token_features"]),
            {"logp", "rank", "log_rank", "entropy"},
        )
        self.assertTrue(
            all(
                isinstance(values, np.ndarray)
                and values.dtype == np.float64
                for values in compact["token_features"].values()
            )
        )
        self.assertEqual(compact["doc_scores"], row["doc_scores"])

    def test_metric_primitives(self) -> None:
        self.assertEqual(auroc([0, 1], [2, 3]), 1.0)
        self.assertEqual(auroc([0, 1], [1, 2]), 0.875)
        self.assertGreater(partial_auroc([0, 1], [2, 3], 0.05), 0.0)
        threshold = calibration_threshold([0, 1, 2, 3], 0.01)
        self.assertGreater(threshold, 3)

    def test_precomputed_scores_match_direct_functions_for_all_detectors(self) -> None:
        base = load_config("configs/smoke.json")
        config = resolved_run_config(
            base, "xsum", "Qwen/Qwen2.5-0.5B", "precompute-test"
        )
        target_rows = [
            compact_evaluation_row(
                feature_row(
                    config,
                    f"source-{index}",
                    "test",
                    "llm",
                    0.2 + index * 0.1,
                )
            )
            for index in range(3)
        ]
        pair_rows = [
            compact_evaluation_row(binoculars_row(row))
            for row in target_rows
        ]
        timing_events = []
        for detector in (
            "log_likelihood",
            "rank",
            "log_rank",
            "lrr",
            "entropy",
            "entropy_gap",
            "binoculars",
        ):
            rows = pair_rows if detector == "binoculars" else target_rows
            direction = -1 if detector in {"rank", "log_rank", "entropy"} else 1
            clip_spec = (
                {"nll_upper": 1.9, "log_rank_upper": 1.7}
                if detector == "lrr"
                else {"lower": -1.25}
            )
            expected_raw = np.asarray(
                [oriented_score(row, detector, direction) for row in rows]
            )
            expected_clipped = np.asarray(
                [
                    oriented_score(row, detector, direction, clip_spec)
                    for row in rows
                ]
            )
            with patch(
                "experiment_core.analysis.evaluation.oriented_score",
                wraps=oriented_score,
            ) as score_spy:
                precomputed = precompute_evaluation_scores(
                    rows,
                    detector,
                    direction,
                    clip_spec,
                    "equivalence",
                    timing_events.append,
                )
            self.assertEqual(score_spy.call_count, 2 * len(rows))
            np.testing.assert_array_equal(precomputed.raw, expected_raw)
            np.testing.assert_array_equal(
                precomputed.clipped, expected_clipped
            )
            self.assertEqual(precomputed.raw.dtype, np.float64)
            self.assertEqual(precomputed.clipped.dtype, np.float64)
            self.assertEqual(
                sorted(precomputed.source_row_indices),
                [f"source-{index}" for index in range(3)],
            )

        self.assertEqual(len(timing_events), 7)
        self.assertTrue(
            all(
                event["raw_score_count"] == 3
                and event["clipped_score_count"] == 3
                and event["row_count"] == 3
                and event["elapsed_seconds"] >= 0.0
                for event in timing_events
            )
        )

    def test_random_draw_rows_remain_clustered_by_source(self) -> None:
        base = load_config("configs/smoke.json")
        config = resolved_run_config(
            base, "xsum", "Qwen/Qwen2.5-0.5B", "draw-clustering-test"
        )
        rows = []
        for source_index, sample_id in enumerate(("source-a", "source-b")):
            for draw_id in range(3):
                row = feature_row(
                    config,
                    sample_id,
                    "test",
                    "llm",
                    0.4 + source_index * 0.1 + draw_id * 0.01,
                    "random",
                    0.5,
                )
                row["corruption_draw_id"] = draw_id
                rows.append(compact_evaluation_row(row))
        precomputed = precompute_evaluation_scores(
            rows,
            "log_likelihood",
            1,
            {"lower": -2.0},
            "draw-clustering",
        )
        grouped = _selected_source_indices(
            precomputed, np.arange(len(rows), dtype=np.int64)
        )
        np.testing.assert_array_equal(grouped["source-a"], [0, 1, 2])
        np.testing.assert_array_equal(grouped["source-b"], [3, 4, 5])
        np.testing.assert_array_equal(
            _bootstrap_index_values(
                np.arange(len(rows), dtype=np.float64),
                grouped,
                ["source-b", "source-a", "source-b"],
            ),
            [3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )

    def test_orientation_and_tuning_finish_before_any_split_precompute(self) -> None:
        base = load_config("configs/smoke.json")
        base["scoring"]["detectors"] = ["log_likelihood"]
        base["evaluation"]["oracle_mode_specific_clipping"] = True
        base["evaluation"]["bootstrap_repetitions"] = 2
        config = resolved_run_config(
            base, "xsum", "Qwen/Qwen2.5-0.5B", "split-safety-test"
        )
        rows = protocol_rows(config)
        events = []

        def orientation_spy(tuning_human, tuning_llm, detector):
            self.assertEqual(
                {row["split"] for row in tuning_human + tuning_llm},
                {"clipping_tuning"},
            )
            events.append(("orientation", detector))
            return orientation(tuning_human, tuning_llm, detector)

        def tuning_spy(
            detector,
            direction,
            tuning_human,
            tuning_clean_llm,
            tuning_mixture,
            quantiles,
        ):
            self.assertEqual(
                {
                    row["split"]
                    for row in (
                        list(tuning_human)
                        + list(tuning_clean_llm)
                        + list(tuning_mixture)
                    )
                },
                {"clipping_tuning"},
            )
            events.append(("tune", detector))
            return tune_clipping_spec(
                detector,
                direction,
                tuning_human,
                tuning_clean_llm,
                tuning_mixture,
                quantiles,
            )

        def precompute_spy(
            score_rows,
            detector,
            direction,
            clip_spec,
            analysis,
            timing_callback,
        ):
            self.assertEqual(events.count(("tune", detector)), 3)
            self.assertIn("calibration", {row["split"] for row in score_rows})
            self.assertIn("test", {row["split"] for row in score_rows})
            events.append(("precompute", detector))
            return precompute_evaluation_scores(
                score_rows,
                detector,
                direction,
                clip_spec,
                analysis,
                timing_callback,
            )

        with tempfile.TemporaryDirectory() as temporary:
            target_path = Path(temporary) / "target.jsonl"
            output = Path(temporary) / "metrics.csv"
            for row in rows:
                append_jsonl(target_path, row)
            with (
                patch(
                    "experiment_core.analysis.evaluation.orientation",
                    side_effect=orientation_spy,
                ),
                patch(
                    "experiment_core.analysis.evaluation.tune_clipping_spec",
                    side_effect=tuning_spy,
                ),
                patch(
                    "experiment_core.analysis.evaluation.precompute_evaluation_scores",
                    side_effect=precompute_spy,
                ),
            ):
                evaluate(
                    target_path,
                    output,
                    config,
                    timing_callback=None,
                )

        self.assertEqual(events[0], ("orientation", "log_likelihood"))
        self.assertEqual(
            events[1:4],
            [("tune", "log_likelihood")] * 3,
        )
        self.assertEqual(
            events[4:],
            [("precompute", "log_likelihood")] * 3,
        )

    def test_index_bootstrap_matches_fixed_seed_serial_reference(self) -> None:
        base = load_config("configs/smoke.json")
        config = resolved_run_config(
            base, "xsum", "Qwen/Qwen2.5-0.5B", "bootstrap-equivalence"
        )
        rows = []
        for index in range(6):
            sample_id = f"cluster-{index}"
            rows.extend(
                [
                    feature_row(
                        config,
                        sample_id,
                        "test",
                        "human",
                        -0.35 + index * 0.04,
                    ),
                    feature_row(
                        config,
                        sample_id,
                        "test",
                        "llm",
                        0.85 - index * 0.03,
                    ),
                    feature_row(
                        config,
                        sample_id,
                        "test",
                        "llm",
                        0.45 - index * 0.03,
                        "tail",
                        0.5,
                    ),
                ]
            )
        rows = [compact_evaluation_row(row) for row in rows]
        direction = 1
        clip_spec = {"nll_upper": 1.8, "log_rank_upper": 1.6}
        precomputed = precompute_evaluation_scores(
            rows,
            "lrr",
            direction,
            clip_spec,
            "bootstrap-equivalence",
        )
        human_indices = np.asarray(
            [index for index, row in enumerate(rows) if row["label"] == "human"]
        )
        clean_indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["label"] == "llm"
                and row["contamination_mode"] == "none"
            ]
        )
        attacked_indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["contamination_mode"] == "tail"
            ]
        )
        raw_fn = lambda row: oriented_score(row, "lrr", direction)
        clipped_fn = lambda row: oriented_score(
            row, "lrr", direction, clip_spec
        )
        raw_threshold = calibration_threshold(
            [raw_fn(rows[index]) for index in human_indices], 0.05
        )
        clipped_threshold = calibration_threshold(
            [clipped_fn(rows[index]) for index in human_indices], 0.05
        )
        seed = 17831
        repetitions = 40

        def serial_bootstrap(llm_indices):
            human_rows = [rows[index] for index in human_indices]
            llm_rows = [rows[index] for index in llm_indices]
            grouped_human = {}
            grouped_llm = {}
            for row in human_rows:
                grouped_human.setdefault(str(row["sample_id"]), []).append(row)
            for row in llm_rows:
                grouped_llm.setdefault(str(row["sample_id"]), []).append(row)
            ids = sorted(set(grouped_human) | set(grouped_llm))
            rng = random.Random(seed)
            samples = {
                name: []
                for name in (
                    "raw_actual_fpr",
                    "clipped_actual_fpr",
                    "raw_tpr",
                    "clipped_tpr",
                    "paired_tpr_difference",
                    "raw_auroc",
                    "clipped_auroc",
                    "paired_auroc_difference",
                    "raw_partial_auroc",
                    "clipped_partial_auroc",
                    "paired_partial_auroc_difference",
                )
            }
            for _ in range(repetitions):
                sampled = [rng.choice(ids) for _ in ids]
                raw_h = [
                    raw_fn(row)
                    for source_id in sampled
                    for row in grouped_human.get(source_id, [])
                ]
                raw_m = [
                    raw_fn(row)
                    for source_id in sampled
                    for row in grouped_llm.get(source_id, [])
                ]
                clipped_h = [
                    clipped_fn(row)
                    for source_id in sampled
                    for row in grouped_human.get(source_id, [])
                ]
                clipped_m = [
                    clipped_fn(row)
                    for source_id in sampled
                    for row in grouped_llm.get(source_id, [])
                ]
                raw_tpr = tpr(raw_m, raw_threshold)
                clipped_tpr = tpr(clipped_m, clipped_threshold)
                raw_auc = auroc(raw_h, raw_m)
                clipped_auc = auroc(clipped_h, clipped_m)
                raw_partial = partial_auroc(raw_h, raw_m, 0.05)
                clipped_partial = partial_auroc(
                    clipped_h, clipped_m, 0.05
                )
                samples["raw_actual_fpr"].append(
                    actual_fpr(raw_h, raw_threshold)
                )
                samples["clipped_actual_fpr"].append(
                    actual_fpr(clipped_h, clipped_threshold)
                )
                samples["raw_tpr"].append(raw_tpr)
                samples["clipped_tpr"].append(clipped_tpr)
                samples["paired_tpr_difference"].append(
                    clipped_tpr - raw_tpr
                )
                samples["raw_auroc"].append(raw_auc)
                samples["clipped_auroc"].append(clipped_auc)
                samples["paired_auroc_difference"].append(
                    clipped_auc - raw_auc
                )
                samples["raw_partial_auroc"].append(raw_partial)
                samples["clipped_partial_auroc"].append(clipped_partial)
                samples["paired_partial_auroc_difference"].append(
                    clipped_partial - raw_partial
                )
            return {
                name: percentile_interval(values)
                for name, values in samples.items()
            }

        with patch(
            "experiment_core.analysis.evaluation.oriented_score",
            side_effect=AssertionError(
                "bootstrap must use precomputed scalar arrays"
            ),
        ):
            indexed = _metric_bootstrap(
                precomputed,
                human_indices,
                attacked_indices,
                raw_threshold,
                clipped_threshold,
                0.05,
                repetitions,
                seed,
            )
        serial = serial_bootstrap(attacked_indices)
        self.assertEqual(set(indexed), set(serial))
        for name in indexed:
            np.testing.assert_allclose(
                indexed[name], serial[name], rtol=0.0, atol=1e-12
            )

        for aggregation, score_fn, threshold in (
            ("raw", raw_fn, raw_threshold),
            ("clipped", clipped_fn, clipped_threshold),
        ):
            direct_h = [score_fn(rows[index]) for index in human_indices]
            direct_m = [score_fn(rows[index]) for index in attacked_indices]
            cached_h = precomputed.values(aggregation, human_indices)
            cached_m = precomputed.values(aggregation, attacked_indices)
            direct_points = (
                actual_fpr(direct_h, threshold),
                tpr(direct_m, threshold),
                auroc(direct_h, direct_m),
                partial_auroc(direct_h, direct_m, 0.05),
            )
            cached_points = (
                actual_fpr(cached_h, threshold),
                tpr(cached_m, threshold),
                auroc(cached_h, cached_m),
                partial_auroc(cached_h, cached_m, 0.05),
            )
            np.testing.assert_allclose(
                cached_points, direct_points, rtol=0.0, atol=1e-12
            )

        def serial_robustness(aggregation, threshold):
            score_fn = raw_fn if aggregation == "raw" else clipped_fn
            grouped_human = {}
            grouped_ratios = []
            for index in human_indices:
                row = rows[index]
                grouped_human.setdefault(str(row["sample_id"]), []).append(row)
            for ratio_indices in (clean_indices, attacked_indices):
                grouped = {}
                for index in ratio_indices:
                    row = rows[index]
                    grouped.setdefault(str(row["sample_id"]), []).append(row)
                grouped_ratios.append(grouped)
            ids = sorted(
                set(grouped_human)
                | set(grouped_ratios[0])
                | set(grouped_ratios[1])
            )
            rng = random.Random(seed)
            values = []
            for _ in range(repetitions):
                sampled = [rng.choice(ids) for _ in ids]
                curve = [
                    tpr(
                        [
                            score_fn(row)
                            for source_id in sampled
                            for row in grouped.get(source_id, [])
                        ],
                        threshold,
                    )
                    for grouped in grouped_ratios
                ]
                values.append(robustness_auc([0.0, 0.5], curve))
            return percentile_interval(values)

        for aggregation, threshold in (
            ("raw", raw_threshold),
            ("clipped", clipped_threshold),
        ):
            with patch(
                "experiment_core.analysis.evaluation.oriented_score",
                side_effect=AssertionError(
                    "robustness bootstrap must use precomputed scalar arrays"
                ),
            ):
                indexed_robustness = _robustness_bootstrap(
                    precomputed,
                    human_indices,
                    [clean_indices, attacked_indices],
                    [0.0, 0.5],
                    aggregation,
                    threshold,
                    repetitions,
                    seed,
                )
            np.testing.assert_allclose(
                indexed_robustness,
                serial_robustness(aggregation, threshold),
                rtol=0.0,
                atol=1e-12,
            )

    def test_independent_protocol_outputs_all_detectors(self) -> None:
        base = load_config("configs/smoke.json")
        base["evaluation"]["bootstrap_repetitions"] = 4
        config = resolved_run_config(
            base, "xsum", "Qwen/Qwen2.5-0.5B", "test-run"
        )
        target_rows = protocol_rows(config)
        timing_events = []

        with tempfile.TemporaryDirectory() as temporary:
            target_path = Path(temporary) / "target.jsonl"
            binoculars_path = Path(temporary) / "binoculars.jsonl"
            output = Path(temporary) / "metrics.csv"
            repeated_output = Path(temporary) / "metrics-repeated.csv"
            for row in target_rows:
                append_jsonl(target_path, row)
                append_jsonl(binoculars_path, binoculars_row(row))
            metrics = evaluate(
                target_path,
                output,
                config,
                binoculars_path,
                timing_callback=timing_events.append,
            )
            first_csv = output.read_bytes()
            repeated_metrics = evaluate(
                target_path,
                repeated_output,
                config,
                binoculars_path,
                timing_callback=None,
            )
            repeated_csv = repeated_output.read_bytes()

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
        self.assertEqual(metrics, repeated_metrics)
        self.assertEqual(first_csv, repeated_csv)
        configured = config["scoring"]["detectors"]
        expected_order = [
            (
                detector,
                mode,
                ratio,
                target_fpr,
                aggregation,
                "primary_frozen_mixture",
            )
            for detector in configured
            for mode in ("random", "tail")
            for ratio in (0.0, 0.1, 0.5)
            for target_fpr in (0.01, 0.05)
            for aggregation in ("raw", "clipped")
        ]
        observed_order = [
            (
                row["detector"],
                row["contamination_mode"],
                row["requested_contamination_ratio"],
                row["target_fpr"],
                row["aggregation"],
                row["analysis"],
            )
            for row in metrics
        ]
        self.assertEqual(observed_order, expected_order)
        self.assertTrue(
            all(
                row["run_id"] == "test-run"
                and row["dataset"] == "xsum"
                and row["model"] == "Qwen/Qwen2.5-0.5B"
                and row["generation_seeds"] == "[101, 202, 303]"
                and row["bootstrap_seed"] == 481516
                for row in metrics
            )
        )
        self.assertEqual(
            [event["detector"] for event in timing_events],
            configured,
        )
        self.assertTrue(
            all(
                event["analysis"] == "primary_frozen_mixture"
                and event["row_count"] == len(target_rows)
                and event["raw_score_count"] == len(target_rows)
                and event["clipped_score_count"] == len(target_rows)
                for event in timing_events
            )
        )


if __name__ == "__main__":
    unittest.main()
