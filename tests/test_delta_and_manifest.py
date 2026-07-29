from __future__ import annotations

import unittest
from pathlib import Path

from llm_detection.config import load_config, resolved_run_config
from llm_detection.runtime import build_manifest


class DeltaAndManifestTests(unittest.TestCase):
    def test_delta_partitions_and_matrix_are_static_validated(self) -> None:
        job = Path("scripts/delta_experiment.sbatch").read_text(encoding="utf-8")
        submit = Path("scripts/submit_delta_matrix.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=gpuA100x4", job)
        self.assertIn("gpuA100x4)", submit)
        self.assertIn("gpuH200x8)", submit)
        for model in (
            "meta-llama/Llama-3.1-8B",
            "mistralai/Mistral-Small-24B-Base-2501",
            "KoboldAI/GPT-NeoX-20B-Erebus",
            "Qwen/Qwen2.5-72B",
        ):
            self.assertIn(model, submit)

    def test_manifest_contains_required_provenance_groups(self) -> None:
        base = load_config("configs/smoke.json")
        config = resolved_run_config(
            base, "xsum", "Qwen/Qwen2.5-0.5B", "manifest-test"
        )
        manifest = build_manifest(
            config,
            ".",
            {"source": "source.jsonl"},
            {"data": "data.jsonl"},
            "source.jsonl",
        )
        for key in (
            "run_id",
            "creation_time_utc",
            "git_commit",
            "dirty_worktree",
            "dataset",
            "source_sample_manifest",
            "target_model",
            "target_model_revision",
            "target_tokenizer_revision",
            "generation",
            "random_seeds",
            "split_sizes",
            "contamination",
            "scoring",
            "software_versions",
            "accelerator",
            "host",
            "inputs",
            "outputs",
            "completion_status",
        ):
            self.assertIn(key, manifest)


if __name__ == "__main__":
    unittest.main()
