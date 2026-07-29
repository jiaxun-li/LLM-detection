from __future__ import annotations

import math
import unittest

import numpy as np

from llm_detection.scoring import (
    binoculars_score,
    numpy_cross_entropy,
    numpy_exact_token_features,
    single_model_doc_scores,
)
from llm_detection.generation import length_bucketed


class ExactScoringTests(unittest.TestCase):
    def test_length_buckets_are_bounded_and_lossless(self) -> None:
        items = list(range(17))
        batches = list(length_bucketed(items, lambda value: value, 3, 4))
        self.assertEqual(sorted(item for batch in batches for item in batch), items)
        self.assertTrue(all(1 <= len(batch) <= 3 for batch in batches))
        self.assertTrue(
            all(
                max(batch) // 4 == min(batch) // 4
                for batch in batches
            )
        )

    def test_exact_rank_entropy_and_log_probability(self) -> None:
        logits = np.array(
            [
                [0.0, 2.0, 1.0, 2.0],
                [-1.0, 0.0, 3.0, 1.0],
            ]
        )
        targets = np.array([1, 3])
        features = numpy_exact_token_features(logits, targets)
        # First target ties for the maximum and therefore has competition rank 1.
        self.assertEqual(features["rank"].tolist(), [1, 2])
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        expected_entropy = -(probabilities * np.log(probabilities)).sum(axis=-1)
        np.testing.assert_allclose(features["entropy"], expected_entropy, rtol=1e-12)
        np.testing.assert_allclose(
            features["logp"],
            np.log(probabilities[np.arange(2), targets]),
            rtol=1e-12,
        )

    def test_single_forward_feature_set_includes_lrr(self) -> None:
        features = {
            "logp": [-2.0, -1.0],
            "rank": [3, 1],
            "log_rank": [math.log(3), 0.0],
            "entropy": [1.5, 0.5],
        }
        scores = single_model_doc_scores(features)
        self.assertEqual(
            set(scores),
            {
                "log_likelihood",
                "rank",
                "log_rank",
                "lrr",
                "entropy",
                "entropy_gap",
            },
        )
        self.assertTrue(math.isfinite(scores["lrr"]))

    def test_binoculars_roles_and_formula(self) -> None:
        observer_logits = np.array([[2.0, 0.0], [0.0, 2.0]])
        performer_logits = np.array([[1.0, 0.0], [0.0, 1.0]])
        xent = numpy_cross_entropy(observer_logits, performer_logits)
        target_ids = np.array([0, 1])
        perf = numpy_exact_token_features(performer_logits, target_ids)
        nll = -perf["logp"]
        value = binoculars_score(float(nll.mean()), float(xent.mean()))
        expected = math.exp(float(nll.mean())) / math.exp(float(xent.mean()))
        self.assertAlmostEqual(value, expected)


if __name__ == "__main__":
    unittest.main()
