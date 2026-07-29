from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from llm_detection.scoring import (
    binoculars_score,
    exact_token_features,
    numpy_cross_entropy,
    numpy_exact_token_features,
    single_model_doc_scores,
    TargetModelScorer,
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

    def test_compact_top_k_features_and_margins(self) -> None:
        logits = np.array(
            [
                [0.0, 3.0, 1.0, 2.0],
                [4.0, 0.0, 2.0, 1.0],
            ]
        )
        targets = np.array([2, 0])
        features = numpy_exact_token_features(logits, targets, saved_top_k=3)
        self.assertEqual(
            features["top_k_token_ids"].tolist(),
            [[1, 3, 2], [0, 2, 3]],
        )
        np.testing.assert_allclose(
            features["top1_top2_logprob_margin"], [1.0, 2.0]
        )
        np.testing.assert_allclose(
            features["target_top1_logprob_margin"], [-2.0, 0.0]
        )
        np.testing.assert_allclose(
            features["top_k_logprobs"][:, 0]
            + features["target_top1_logprob_margin"],
            features["logp"],
        )

    def test_torch_chunked_features_match_numpy_reference(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        logits = np.array(
            [
                [0.0, 3.0, 1.0, 2.0],
                [4.0, 0.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )
        targets = np.array([2, 0], dtype=np.int64)
        expected = numpy_exact_token_features(logits, targets, saved_top_k=3)
        actual = exact_token_features(
            torch.tensor(logits),
            torch.tensor(targets),
            vocab_chunk_size=2,
            saved_top_k=3,
        )
        for name in (
            "logp",
            "rank",
            "log_rank",
            "entropy",
            "top_k_logprobs",
            "top1_top2_logprob_margin",
            "target_top1_logprob_margin",
        ):
            np.testing.assert_allclose(
                actual[name].cpu().numpy(),
                expected[name],
                rtol=1e-6,
                atol=1e-6,
            )
        np.testing.assert_array_equal(
            actual["top_k_token_ids"].cpu().numpy(),
            expected["top_k_token_ids"],
        )

    def test_optional_mean_pooled_final_hidden_state(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class FakeTokenizer:
            pad_token_id = 0

            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return {"prompt": [1, 2], "text": [3, 4, 5]}[text]

        class FakeModel:
            def __call__(
                self,
                input_ids,
                attention_mask,
                output_hidden_states,
                return_dict,
            ):
                del attention_mask
                self.requested_hidden_states = output_hidden_states
                self.requested_return_dict = return_dict
                batch, sequence = input_ids.shape
                logits = (
                    torch.arange(12, dtype=torch.float32)
                    .view(1, 1, 12)
                    .expand(batch, sequence, 12)
                )
                hidden = input_ids.float().unsqueeze(-1) + torch.arange(
                    4, dtype=torch.float32
                )
                return SimpleNamespace(
                    logits=logits,
                    hidden_states=(hidden * 0.0, hidden),
                )

        scorer = TargetModelScorer.__new__(TargetModelScorer)
        scorer.config = {
            "max_tokens": 16,
            "vocab_chunk_size": 4,
            "saved_top_k": 10,
            "save_mean_pooled_final_hidden_state": True,
        }
        scorer.model_id = "fake/model"
        scorer.resolved_revision = "fake-model-commit"
        scorer.resolved_tokenizer_revision = "fake-tokenizer-commit"
        scorer.tokenizer = FakeTokenizer()
        scorer.model = FakeModel()
        scorer.device = torch.device("cpu")

        scored = scorer.score_batch(
            [{"sample_id": "sample", "prompt": "prompt", "text": "text"}]
        )[0]
        self.assertTrue(scorer.model.requested_hidden_states)
        self.assertTrue(scorer.model.requested_return_dict)
        self.assertEqual(
            scored["scoring_feature_schema"], "target-token-features-v2"
        )
        self.assertEqual(
            scored["document_features"]["pooling_token_count"], 3
        )
        self.assertEqual(scored["document_features"]["hidden_size"], 4)
        np.testing.assert_allclose(
            scored["document_features"]["mean_pooled_final_hidden_state"],
            [3.0, 4.0, 5.0, 6.0],
        )
        self.assertEqual(
            len(scored["token_features"]["top_k_token_ids"]), 3
        )
        self.assertTrue(
            all(
                len(values) == 10
                for values in scored["token_features"]["top_k_token_ids"]
            )
        )

    def test_resume_rejects_incompatible_feature_schema(self) -> None:
        scorer = TargetModelScorer.__new__(TargetModelScorer)
        scorer.config = {
            "saved_top_k": 10,
            "save_mean_pooled_final_hidden_state": False,
        }
        scorer.resolved_revision = "model-commit"
        scorer.resolved_tokenizer_revision = "tokenizer-commit"
        row = {
            "scoring_feature_schema": "target-token-features-v2",
            "scoring_model_revision": "model-commit",
            "scoring_tokenizer_revision": "tokenizer-commit",
            "token_features": {"top_k_token_ids": [[index for index in range(10)]]},
        }
        scorer.validate_existing_row(row)
        row["scoring_feature_schema"] = "legacy"
        with self.assertRaisesRegex(ValueError, "choose a new run ID"):
            scorer.validate_existing_row(row)

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
