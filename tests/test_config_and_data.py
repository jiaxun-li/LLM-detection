from __future__ import annotations

import json
import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_detection.config import load_config, resolved_run_config, total_examples
from llm_detection.data import (
    assign_splits_and_generation_seeds,
    expected_rows_per_source,
    random_token_contamination,
    rank_tail_candidates_once,
    tail_token_contamination,
    validate_disjoint_splits,
)
from llm_detection.pipeline import select_source_manifest


class ConfigAndDataTests(unittest.TestCase):
    def test_paper_configuration_records_scientific_defaults(self) -> None:
        config = load_config("configs/paper.json")
        self.assertEqual(total_examples(config), 3000)
        self.assertEqual(config["datasets"]["xsum"]["split"], "validation")
        self.assertEqual(config["datasets"]["squad"]["split"], "train")
        self.assertTrue(config["datasets"]["squad"]["pack_short_documents"])
        self.assertGreaterEqual(
            config["datasets"]["squad"]["minimum_source_words"],
            config["generation"]["prompt_tokens"]
            + config["generation"]["continuation_tokens"],
        )
        self.assertEqual(config["datasets"]["writingprompts"]["split"], "validation")
        self.assertEqual(config["splits"]["clipping_tuning"], 500)
        self.assertEqual(config["splits"]["calibration"], 500)
        self.assertEqual(config["splits"]["test"], 2000)
        self.assertEqual(config["generation"]["seeds"], [101, 202, 303])
        self.assertEqual(config["generation"]["prompt_tokens"], 30)
        self.assertEqual(config["generation"]["continuation_tokens"], 220)
        self.assertEqual(config["generation"]["temperature"], 0.8)
        self.assertEqual(config["generation"]["top_p"], 0.95)
        self.assertEqual(config["contamination"]["random_draws"], 3)
        self.assertEqual(
            config["contamination"]["ratios"],
            [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        )
        self.assertIn("lrr", config["scoring"]["detectors"])
        self.assertEqual(config["scoring"]["saved_top_k"], 10)
        self.assertFalse(
            config["scoring"]["save_mean_pooled_final_hidden_state"]
        )

    def test_splits_are_deterministic_disjoint_and_seeds_divide_total(self) -> None:
        sources = [
            {"source_id": f"id-{index}", "source_text": f"text {index}"}
            for index in range(30)
        ]
        counts = {"clipping_tuning": 5, "calibration": 5, "test": 11}
        first = assign_splits_and_generation_seeds(sources, counts, [1, 2, 3], 99)
        second = assign_splits_and_generation_seeds(
            list(reversed(sources)), counts, [1, 2, 3], 99
        )
        self.assertEqual(first, second)
        validate_disjoint_splits(first, counts)
        self.assertEqual(len(first), sum(counts.values()))
        self.assertEqual({row["generation_seed"] for row in first}, {1, 2, 3})
        self.assertEqual(len({row["sample_id"] for row in first}), len(first))

    def test_ratio_zero_is_not_duplicated_by_modes(self) -> None:
        self.assertEqual(expected_rows_per_source([0.0, 0.1, 0.5], 3), 10)
        # 2 clean + 2 positive ratios * (3 random draws + 1 tail)

    def test_token_budget_contamination_is_exact_and_fixed_length(self) -> None:
        original = list(range(100))
        spans = [list(range(1000, 1030)), list(range(2000, 2050))]
        mixed, metadata = random_token_contamination(original, spans, 0.2, 123)
        self.assertEqual(len(mixed), len(original))
        self.assertEqual(metadata["human_token_count"], 20)
        self.assertEqual(metadata["replaced_token_count"], 20)
        self.assertAlmostEqual(metadata["realized_contamination_ratio"], 0.2)
        changed = sum(first != second for first, second in zip(original, mixed))
        self.assertEqual(changed, 20)

    def test_tail_candidates_are_scored_once_and_order_reused(self) -> None:
        spans = [[10, 11], [20, 21, 22], [30]]
        calls = []

        def score(span):
            calls.append(tuple(span))
            return {10: 1.0, 20: 3.0, 30: 2.0}[span[0]]

        ranked = rank_tail_candidates_once(spans, score)
        self.assertEqual(len(calls), len(spans))
        self.assertEqual([item.candidate_id for item in ranked], [1, 2, 0])
        original = list(range(20))
        low, low_meta = tail_token_contamination(original, ranked, 0.1)
        high, high_meta = tail_token_contamination(original, ranked, 0.5)
        self.assertEqual(len(calls), len(spans))
        self.assertEqual(len(low), 20)
        self.assertEqual(len(high), 20)
        self.assertEqual(
            low_meta["tail_candidate_ids"],
            high_meta["tail_candidate_ids"][: len(low_meta["tail_candidate_ids"])],
        )
        self.assertEqual(low_meta["human_token_count"], 2)
        self.assertEqual(high_meta["human_token_count"], 10)

    def test_squad_short_contexts_are_packed_without_reuse(self) -> None:
        config = copy.deepcopy(load_config("configs/smoke.json"))
        config["splits"].update(
            {"clipping_tuning": 1, "calibration": 1, "test": 2}
        )
        fake_rows = [
            {
                "id": f"question-{index}",
                "context": " ".join(
                    [f"unique{index}"] + [f"word{offset}" for offset in range(99)]
                ),
            }
            for index in range(12)
        ]
        datasets_module = types.ModuleType("datasets")
        datasets_module.load_dataset = lambda **_kwargs: iter(fake_rows)
        hub_module = types.ModuleType("huggingface_hub")

        class FakeInfo:
            sha = "resolved-dataset-commit"

        class FakeApi:
            def dataset_info(self, *_args, **_kwargs):
                return FakeInfo()

        hub_module.HfApi = FakeApi
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            sys.modules,
            {"datasets": datasets_module, "huggingface_hub": hub_module},
        ):
            path = Path(temporary) / "sources.jsonl"
            select_source_manifest(config, "squad", path)
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(len(rows), 4)
        components = [
            component
            for row in rows
            for component in row["source_component_ids"]
        ]
        self.assertEqual(len(components), len(set(components)))
        self.assertTrue(
            all(row["dataset_resolved_revision"] == "resolved-dataset-commit" for row in rows)
        )
        self.assertTrue(all(len(row["source_text"].split()) >= 260 for row in rows))


if __name__ == "__main__":
    unittest.main()
