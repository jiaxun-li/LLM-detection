from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
from pathlib import Path

from llm_detection.data import (
    apply_token_replacement_plan,
    random_token_contamination,
    random_token_replacement_plan,
)
from llm_detection.splice_audit import (
    AUDIT_CONDITIONS,
    evaluate_splice_audit,
    select_test_base_rows,
    sentence_aligned_splice,
)


class WordTokenizer:
    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def _tokens(self, text: str) -> list[tuple[str, int, int]]:
        return [
            (match.group(0), match.start(), match.end())
            for match in re.finditer(r"\S+", text)
        ]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        result = []
        for token, _start, _end in self._tokens(text):
            if token not in self.token_to_id:
                token_id = len(self.token_to_id) + 1
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
            result.append(self.token_to_id[token])
        return result

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(self.id_to_token[int(value)] for value in token_ids)

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        del add_special_tokens
        tokens = self._tokens(text)
        result = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (start, end) for _token, start, end in tokens
            ]
        return result


class SpliceAuditTests(unittest.TestCase):
    def test_frozen_config_and_delta_wrapper_match_bounded_protocol(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "splice_artifact_audit_granite_xsum.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(config["sample_count"], 500)
        self.assertEqual(config["alternative_generation_seed"], 404)
        self.assertEqual(config["ratios"], [0.1, 0.3, 0.5])
        self.assertEqual(tuple(config["conditions"]), AUDIT_CONDITIONS)
        self.assertEqual(config["bootstrap_repetitions"], 2000)
        wrapper = (
            root / "scripts" / "delta_splice_artifact_audit.sbatch"
        ).read_text(encoding="utf-8")
        for required in (
            "#SBATCH --gpus-per-node=2",
            "#SBATCH --time=06:00:00",
            'assert torch.version.cuda == "12.8"',
            "runs must resolve under /work/hdd",
            "results must resolve under /work/hdd",
            "run_splice_artifact_audit.py",
        ):
            self.assertIn(required, wrapper)

    def test_exported_plan_exactly_matches_primary_random_contamination(self) -> None:
        original = list(range(100))
        spans = [list(range(1000, 1017)), list(range(2000, 2031))]
        chunks, windows = random_token_replacement_plan(100, spans, 0.3, 1729)
        planned = apply_token_replacement_plan(original, chunks, windows)
        primary, metadata = random_token_contamination(
            original, spans, 0.3, 1729
        )
        self.assertEqual(planned, primary)
        self.assertEqual(sum(len(chunk) for chunk in chunks), 30)
        self.assertEqual(metadata["replaced_token_count"], 30)
        occupied = [position for start, end in windows for position in range(start, end)]
        self.assertEqual(len(occupied), len(set(occupied)))

    def test_sentence_pair_uses_same_recipient_positions_and_whole_donors(self) -> None:
        tokenizer = WordTokenizer()
        recipient = "One two three. Four five six seven. Eight nine ten."
        human = "Human donor sentence. Another human donor sentence here."
        llm = "Machine donor sentence. Another machine donor sentence here."
        human_text, human_meta = sentence_aligned_splice(
            tokenizer, recipient, human, 0.3, 11, 21
        )
        llm_text, llm_meta = sentence_aligned_splice(
            tokenizer,
            recipient,
            llm,
            0.3,
            11,
            31,
            human_meta["recipient_sentence_indices"],
        )
        self.assertEqual(
            human_meta["recipient_sentence_indices"],
            llm_meta["recipient_sentence_indices"],
        )
        self.assertIn("Human donor sentence.", human_text)
        self.assertIn("Machine donor sentence.", llm_text)
        self.assertTrue(human_meta["splice_boundary_token_indices"])
        self.assertTrue(llm_meta["splice_boundary_token_indices"])

    def test_test_source_selection_is_order_independent(self) -> None:
        rows = [
            {"sample_id": f"id-{index}", "split": "test"}
            for index in range(20)
        ] + [{"sample_id": "calibration", "split": "calibration"}]
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.jsonl"
            second = Path(temporary) / "second.jsonl"
            first.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            second.write_text(
                "".join(json.dumps(row) + "\n" for row in reversed(rows)),
                encoding="utf-8",
            )
            selected_first = select_test_base_rows(first, 7, 99)
            selected_second = select_test_base_rows(second, 7, 99)
        self.assertEqual(selected_first, selected_second)
        self.assertTrue(all(row["split"] == "test" for row in selected_first))

    def test_frozen_evaluator_writes_paired_artifact_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "source_metrics.csv"
            fieldnames = [
                "split",
                "analysis",
                "detector",
                "aggregation",
                "target_fpr",
                "direction",
                "clipping_specification",
                "calibration_threshold",
                "actual_fpr",
                "requested_contamination_ratio",
                "tpr",
            ]
            with metrics.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for aggregation in ("raw", "clipped"):
                    for target_fpr in (0.01, 0.05):
                        writer.writerow(
                            {
                                "split": "test",
                                "analysis": "primary_frozen_mixture",
                                "detector": "log_likelihood",
                                "aggregation": aggregation,
                                "target_fpr": target_fpr,
                                "direction": 1,
                                "clipping_specification": (
                                    "{}"
                                    if aggregation == "raw"
                                    else json.dumps({"lower": -4.0})
                                ),
                                "calibration_threshold": -2.0,
                                "actual_fpr": target_fpr,
                                "requested_contamination_ratio": 0.0,
                                "tpr": 1.0,
                            }
                        )
            scores = root / "scores.jsonl"
            rows = []
            for source_index in range(8):
                sample_id = f"source-{source_index}"
                clean_logp = -1.0 - 0.08 * source_index
                rows.append(
                    {
                        "sample_id": sample_id,
                        "contamination_mode": "none",
                        "audit_condition": "none",
                        "requested_contamination_ratio": 0.0,
                        "splice_boundary_token_indices": [],
                        "token_features": {
                            "logp": [clean_logp] * 8,
                            "rank": [1.0] * 8,
                            "log_rank": [0.0] * 8,
                            "entropy": [1.0] * 8,
                        },
                        "doc_scores": {"log_likelihood": clean_logp},
                    }
                )
                for condition in AUDIT_CONDITIONS:
                    donor_penalty = 1.0 if condition.endswith("human") else 0.5
                    placement_penalty = 0.2 if condition.startswith("token") else 0.0
                    for ratio in (0.1, 0.3, 0.5):
                        logp = clean_logp - ratio * (
                            donor_penalty + placement_penalty
                        )
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "contamination_mode": condition,
                                "audit_condition": condition,
                                "requested_contamination_ratio": ratio,
                                "splice_boundary_token_indices": [2, 5],
                                "token_features": {
                                    "logp": [logp] * 8,
                                    "rank": [1.0] * 8,
                                    "log_rank": [0.0] * 8,
                                    "entropy": [1.0] * 8,
                                },
                                "doc_scores": {"log_likelihood": logp},
                            }
                        )
            scores.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "output"
            summary = evaluate_splice_audit(
                scores,
                metrics,
                output,
                [0.1, 0.3, 0.5],
                bootstrap_repetitions=20,
                bootstrap_seed=7,
                boundary_radius=1,
            )
            with (output / "artifact_share.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                artifact_rows = list(csv.DictReader(handle))
        self.assertEqual(summary["audit_status"], "complete")
        self.assertEqual(summary["metric_rows"], 64)
        self.assertEqual(len(artifact_rows), 8)


if __name__ == "__main__":
    unittest.main()
