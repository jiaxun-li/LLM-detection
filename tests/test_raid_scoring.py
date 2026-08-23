from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from RAID.raid_scoring import (
    BINOCULARS_FEATURE_SCHEMA,
    FALCON_FEATURE_SCHEMA,
    RAIDBinocularsScorer,
    RAIDFalconScorer,
    encode_scored_window,
    raid_row_key,
    score_raid_jsonl,
)
from RAID.raid_data import raid_row_key as prepared_raid_row_key
from llm_detection.scoring import numpy_cross_entropy, numpy_exact_token_features


class FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 99
    eos_token_id = 98

    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.calls = []
        self.init_kwargs = {"_commit_hash": "fake-tokenizer-revision"}

    def encode(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return list(self.mapping.get(text, [1, 2, 3]))


class FakeModel:
    def __init__(self, logits, revision):
        import torch

        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.config = SimpleNamespace(_commit_hash=revision)

    def __call__(self, input_ids, attention_mask):
        del attention_mask
        batch, sequence = input_ids.shape
        logits = self.logits[:sequence].unsqueeze(0).expand(batch, -1, -1).clone()
        return SimpleNamespace(logits=logits)


def prepared_row(condition="none"):
    return {
        "source_id": "source-1",
        "record_type": "machine",
        "selected_generation_id": "generation-7",
        "condition": condition,
        "split": "test",
        "text": "machine text",
        "realized_contamination_rate": 0.125,
    }


class FakeStreamingScorer:
    def token_count(self, row):
        return len(row["text"])

    def score_batch(self, rows):
        return [
            {
                **row,
                "score_row_key": list(raid_row_key(row)),
                "score_row_schema": "fake",
                "num_scored_tokens": 1,
            }
            for row in rows
        ]

    def validate_existing_row(self, row):
        if row.get("score_row_schema") != "fake":
            raise ValueError("schema mismatch")
        if tuple(row["score_row_key"]) != raid_row_key(row):
            raise ValueError("key mismatch")


class RAIDScoringTests(unittest.TestCase):
    def test_output_only_window_ignores_prompt_and_records_right_truncation(self):
        tokenizer = FakeTokenizer({"response": [10, 20, 30, 40, 50]})
        row = {**prepared_row(), "text": "response", "prompt": "must be ignored"}
        window = encode_scored_window(tokenizer, row, max_tokens=4)
        self.assertEqual(window.input_ids, [10, 20, 30, 40])
        self.assertEqual(window.scored_token_ids, [20, 30, 40])
        self.assertEqual(window.full_scored_token_ids, [20, 30, 40, 50])
        self.assertEqual(window.original_num_input_tokens, 5)
        self.assertEqual(window.truncated_token_count, 1)
        self.assertFalse(window.boundary_token_added)
        self.assertEqual(tokenizer.calls, [("response", {"verbose": False})])

    def test_one_token_window_adds_audited_boundary_and_scores_text_token(self):
        tokenizer = FakeTokenizer({"one": [42]})
        window = encode_scored_window(tokenizer, {**prepared_row(), "text": "one"})
        self.assertEqual(window.input_ids, [99, 42])
        self.assertEqual(window.scored_token_ids, [42])
        self.assertEqual(window.full_scored_token_ids, [42])
        self.assertTrue(window.boundary_token_added)
        self.assertEqual(window.truncated_token_count, 0)

    def test_stable_key_distinguishes_attacks_and_honors_explicit_key(self):
        self.assertNotEqual(
            raid_row_key(prepared_row("none")), raid_row_key(prepared_row("synonym"))
        )
        self.assertEqual(raid_row_key({"row_key": ["a", 2]}), ("a", "2"))
        prepared = {
            "source_id": "s",
            "label": "llm",
            "base_generation_id": "g",
            "attack": "synonym",
        }
        self.assertEqual(raid_row_key(prepared), prepared_raid_row_key(prepared))

    def test_falcon_fake_forward_writes_all_six_arrays_and_metadata(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        logits = np.array(
            [[0.0, 3.0, 1.0, 2.0], [4.0, 0.0, 2.0, 1.0], [1.0, 2.0, 3.0, 0.0]],
            dtype=np.float32,
        )
        tokenizer = FakeTokenizer({"machine text": [3, 2, 0]})
        scorer = RAIDFalconScorer(
            {
                "revision": "requested",
                "tokenizer_revision": "requested-tokenizer",
                "dtype": "float32",
                "max_tokens": 512,
                "vocab_chunk_size": 2,
            },
            tokenizer=tokenizer,
            model=FakeModel(logits, "falcon-revision"),
            device="cpu",
        )
        result = scorer.score_batch([prepared_row()])[0]
        self.assertEqual(result["realized_contamination_rate"], 0.125)
        self.assertEqual(result["scoring_feature_schema"], FALCON_FEATURE_SCHEMA)
        self.assertEqual(result["scoring_context_policy"], "raid_output_only_512")
        self.assertEqual(result["scoring_max_tokens"], 512)
        self.assertEqual(result["token_features"]["scored_token_ids"], [2, 0])
        self.assertEqual(result["token_features"]["full_scored_token_ids"], [2, 0])
        self.assertEqual(
            set(result["doc_scores"]),
            {"log_likelihood", "rank", "log_rank", "lrr", "entropy", "entropy_gap"},
        )
        expected = numpy_exact_token_features(logits[:2], np.array([2, 0]))
        np.testing.assert_allclose(result["token_features"]["logp"], expected["logp"])
        np.testing.assert_allclose(
            result["token_features"]["entropy_gap"],
            -expected["logp"] - expected["entropy"],
        )
        scorer.validate_existing_row(result)

    def test_binoculars_fake_forward_uses_roles_formula_and_local_gap(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        observer_logits = np.array(
            [[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 3.0]], dtype=np.float32
        )
        performer_logits = np.array(
            [[2.0, 0.0, 1.0], [1.0, 3.0, 0.0], [0.0, 1.0, 2.0]], dtype=np.float32
        )
        tokenizer = FakeTokenizer({"machine text": [2, 1, 2]})
        scorer = RAIDBinocularsScorer(
            {
                "observer_revision": "requested-observer",
                "performer_revision": "requested-performer",
                "tokenizer_revision": "requested-tokenizer",
                "dtype": "float32",
                "vocab_chunk_size": 2,
            },
            tokenizer=tokenizer,
            observer=FakeModel(observer_logits, "observer-revision"),
            performer=FakeModel(performer_logits, "performer-revision"),
            observer_device="cpu",
            performer_device="cpu",
        )
        result = scorer.score_batch([prepared_row()])[0]
        self.assertEqual(result["scoring_feature_schema"], BINOCULARS_FEATURE_SCHEMA)
        self.assertEqual(result["binoculars_observer_model"], "tiiuae/falcon-7b")
        self.assertEqual(
            result["binoculars_performer_model"], "tiiuae/falcon-7b-instruct"
        )
        self.assertEqual(result["token_features"]["scored_token_ids"], [1, 2])
        xent = numpy_cross_entropy(observer_logits[:2], performer_logits[:2])
        performer = numpy_exact_token_features(performer_logits[:2], np.array([1, 2]))
        gap = -performer["logp"] - xent
        np.testing.assert_allclose(
            result["token_features"]["local_gap"], gap, rtol=1e-6
        )
        self.assertAlmostEqual(
            result["doc_scores"]["binoculars"], math.exp(float(gap.mean())), places=6
        )
        scorer.validate_existing_row(result)

    def test_append_safe_resume_preserves_rows_and_rejects_schema_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "data.jsonl"
            output_path = root / "scores.jsonl"
            rows = [prepared_row("none"), prepared_row("synonym")]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            config = {"batch_size": 2, "microbatch_size": 1, "length_bucket_width": 64}
            scorer = FakeStreamingScorer()
            self.assertEqual(score_raid_jsonl(input_path, output_path, scorer, config), 2)
            self.assertEqual(score_raid_jsonl(input_path, output_path, scorer, config), 2)
            self.assertEqual(len(output_path.read_text(encoding="utf-8").splitlines()), 2)
            saved = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            saved[0]["score_row_schema"] = "old"
            output_path.write_text(
                "".join(json.dumps(row) + "\n" for row in saved), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                score_raid_jsonl(input_path, output_path, scorer, config)

    def test_resume_rejects_changed_falcon_revision(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        logits = np.zeros((3, 4), dtype=np.float32)
        tokenizer = FakeTokenizer({"machine text": [1, 2]})
        first = RAIDFalconScorer(
            {"revision": "a", "dtype": "float32"},
            tokenizer=tokenizer,
            model=FakeModel(logits, "revision-a"),
            device="cpu",
        )
        row = first.score_batch([prepared_row()])[0]
        second = RAIDFalconScorer(
            {"revision": "b", "dtype": "float32"},
            tokenizer=tokenizer,
            model=FakeModel(logits, "revision-b"),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "different model provenance"):
            second.validate_existing_row(row)


if __name__ == "__main__":
    unittest.main()
