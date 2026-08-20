import json
import tempfile
import unittest
from pathlib import Path

from Beemo.beemo_data import (
    VARIANTS,
    normalize_record,
    parse_edit_versions,
    prepare_from_records,
    word_edit_ratio,
)
from Beemo.beemo_evaluation import evaluate_beemo
from Beemo.beemo_pipeline import load_beemo_config, validate_artifacts


def synthetic_record(index: int) -> dict:
    original = f"Machine answer {index} is concise and regular."
    return {
        "id": index,
        "category": "Open QA" if index % 2 else "Generation",
        "model": "generator/a" if index % 2 else "generator/b",
        "prompt_id": f"prompt-{index}",
        "prompt": f"Explain item {index}.",
        "model_output": original,
        "human_output": f"A person explains item {index} with varied phrasing.",
        "human_edits": f"Machine answer {index} is concise, natural, and corrected.",
        "llama-3.1-70b_edits": [
            {"P1": f"Llama one edits answer {index}."},
            {"P2": f"Llama two revises answer {index}."},
            {"P3": f"Llama three rewrites answer {index}."},
        ],
        "gpt-4o_edits": json.dumps(
            [
                {"P1": f"GPT one edits answer {index}."},
                {"P2": f"GPT two revises answer {index}."},
                {"P3": f"GPT three rewrites answer {index}."},
            ]
        ),
    }


def small_config() -> dict:
    return {
        "protocol_name": "beemo-test",
        "result_label": "DEBUG",
        "dataset": {
            "id": "toloka/beemo",
            "revision": "test",
            "expected_records": 12,
        },
        "splits": {
            "clipping_tuning": 3,
            "calibration": 4,
            "test": 5,
            "selection_seed": 17,
        },
        "scoring_models": [
            {"key": "test_a", "id": "reference/test-a", "enabled": True},
            {"key": "test_b", "id": "reference/test-b", "enabled": True},
        ],
        "scoring": {"detectors": ["log_likelihood"]},
        "evaluation": {
            "target_fprs": [0.01, 0.05],
            "partial_auroc_max_fpr": 0.05,
            "bootstrap_repetitions": 5,
            "bootstrap_seed": 19,
            "clipping_quantiles": [0.8, 0.9],
            "primary_condition": "expert",
            "reported_conditions": [
                "original",
                "expert",
                "llama_p1",
                "llama_p2",
                "llama_p3",
                "gpt_p1",
                "gpt_p2",
                "gpt_p3",
                "llama_all",
                "gpt_all",
            ],
        },
    }


class BeemoDataTests(unittest.TestCase):
    def test_frozen_config_and_delta_wrapper(self):
        root = Path(__file__).resolve().parents[1]
        config = load_beemo_config(root / "Beemo" / "config.json")
        self.assertEqual(config["splits"]["clipping_tuning"], 437)
        self.assertEqual(config["splits"]["calibration"], 875)
        self.assertEqual(config["splits"]["test"], 875)
        self.assertEqual(
            [model["key"] for model in config["scoring_models"] if model["enabled"]],
            ["gpt2_xl", "opt_1_3b", "falcon_7b", "qwen2_7b"],
        )
        granite = next(
            model for model in config["scoring_models"] if model["key"] == "granite_3_3_8b"
        )
        self.assertFalse(granite["enabled"])
        self.assertEqual(config["scoring"]["context_policy"], "detectllm_output_only")
        self.assertIsNone(config["scoring"]["max_tokens"])
        self.assertEqual(config["binoculars"]["max_tokens"], 512)
        wrapper = (root / "Beemo" / "delta_beemo.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --gpus-per-node=2", wrapper)
        self.assertIn("runs must resolve under /work/hdd", wrapper)
        self.assertIn('torch.version.cuda == "12.8"', wrapper)
        self.assertIn("INCLUDE_GRANITE", wrapper)
        self.assertIn("--include-granite", wrapper)

    def test_edit_parser_accepts_released_string_shape(self):
        parsed = parse_edit_versions(
            "[{'P1': 'one'}, {'P2': 'two'}, {'P3': 'three'}]", "edits"
        )
        self.assertEqual(parsed, {"P1": "one", "P2": "two", "P3": "three"})

    def test_normalization_and_edit_ratio(self):
        normalized = normalize_record(synthetic_record(1))
        self.assertEqual(normalized["llama_p3"], "Llama three rewrites answer 1.")
        self.assertEqual(word_edit_ratio("same words", "same words"), 0.0)
        self.assertGreater(word_edit_ratio("same words", "different text"), 0.0)

    def test_prepare_keeps_nine_variants_in_one_group_split(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            summary = prepare_from_records(
                [synthetic_record(index) for index in range(12)], path, config
            )
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(summary["rows"], 12 * len(VARIANTS))
        by_id = {}
        for row in rows:
            by_id.setdefault(row["sample_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in by_id.values()))
        self.assertTrue(all(row["prompt"] == row["original_prompt"] for row in rows))
        self.assertTrue(all(row["prompt_response_separator"] == "" for row in rows))
        self.assertEqual(
            {row["target_model"] for row in rows}, {"beemo-mixed-source-generators"}
        )


class BeemoEvaluationTests(unittest.TestCase):
    def test_end_to_end_synthetic_evaluation(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "data.jsonl"
            prepare_from_records(
                [synthetic_record(index) for index in range(12)], data_path, config
            )
            prepared = [
                json.loads(line)
                for line in data_path.read_text(encoding="utf-8").splitlines()
            ]
            means = {
                "human": -4.0,
                "original": -1.0,
                "expert": -2.0,
                "llama_p1": -1.6,
                "llama_p2": -1.7,
                "llama_p3": -1.8,
                "gpt_p1": -1.4,
                "gpt_p2": -1.5,
                "gpt_p3": -1.6,
            }
            score_paths = {}
            score_dir = root / "target_scores"
            score_dir.mkdir()
            for model in config["scoring_models"]:
                scored = []
                for row in prepared:
                    value = means[row["beemo_variant"]]
                    features = [value - 0.1, value, value + 0.1]
                    scored.append(
                        {
                            **row,
                            "scorer_key": model["key"],
                            "scoring_model": model["id"],
                            "scoring_model_revision": "test",
                            "scoring_tokenizer_revision": "test",
                            "scoring_context_policy": "detectllm_output_only",
                            "token_features": {"logp": features},
                            "doc_scores": {
                                "log_likelihood": sum(features) / len(features)
                            },
                        }
                    )
                score_path = score_dir / f"{model['key']}.jsonl"
                score_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in scored),
                    encoding="utf-8",
                )
                score_paths[model["key"]] = score_path
            results_dir = root / "results"
            summary = evaluate_beemo(
                score_paths, results_dir, config, bootstrap_repetitions=5
            )
            metrics = (results_dir / "metrics.csv").read_text(encoding="utf-8")
            frozen = json.loads(
                (results_dir / "frozen_specs.json").read_text(encoding="utf-8")
            )
            for marker in ("prepare.complete.json", "score.complete.json"):
                (root / marker).write_text("{}", encoding="utf-8")
            for marker in ("evaluate.complete.json", "plot.complete.json"):
                (results_dir / marker).write_text("{}", encoding="utf-8")
            for model in config["scoring_models"]:
                (results_dir / f"beemo_tpr_{model['key']}_2x7.png").write_bytes(b"png")
            validation = validate_artifacts(
                root,
                results_dir,
                config,
                skip_binoculars=True,
                include_granite=False,
            )
        self.assertEqual(summary["metrics_rows"], 80)
        self.assertEqual(summary["scorers"], ["test_a", "test_b"])
        self.assertEqual(set(frozen["scorers"]), {"test_a", "test_b"})
        self.assertEqual(validation["target_score_rows"], {"test_a": 108, "test_b": 108})
        self.assertIn("condition_minus_original_tpr", metrics)


if __name__ == "__main__":
    unittest.main()
