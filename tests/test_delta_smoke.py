from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiment_core.infrastructure.config import load_config, total_examples
from experiment_core.infrastructure.io import iter_jsonl
from experiment_core.validation.smoke_validation import DEBUG_LABEL, EXPECTED_DETECTORS


class DeltaSmokeTests(unittest.TestCase):
    def test_dedicated_config_is_exactly_the_bounded_smoke_scope(self) -> None:
        config = load_config("configs/delta_smoke_qwen_0_5b.json")
        self.assertEqual(config["models"]["primary"], ["Qwen/Qwen2.5-0.5B"])
        self.assertEqual(total_examples(config), 12)
        self.assertEqual(
            {
                name: config["splits"][name]
                for name in ("clipping_tuning", "calibration", "test")
            },
            {"clipping_tuning": 4, "calibration": 4, "test": 4},
        )
        self.assertEqual(config["generation"]["seeds"], [101])
        self.assertEqual(config["generation"]["prompt_tokens"], 30)
        self.assertEqual(config["generation"]["continuation_tokens"], 64)
        self.assertEqual(config["generation"]["temperature"], 0.8)
        self.assertEqual(config["generation"]["top_p"], 0.95)
        self.assertEqual(config["contamination"]["ratios"], [0.0, 0.2, 0.5])
        self.assertEqual(config["contamination"]["random_draws"], 1)
        self.assertEqual(set(config["scoring"]["detectors"]), EXPECTED_DETECTORS)
        self.assertNotIn("binoculars", config["scoring"]["detectors"])
        self.assertEqual(config["scoring"]["saved_top_k"], 10)
        self.assertTrue(
            config["scoring"]["save_mean_pooled_final_hidden_state"]
        )
        self.assertTrue(config["debug_only"])
        self.assertEqual(config["result_label"], DEBUG_LABEL)

    def test_slurm_job_and_wrapper_have_static_safety_guards(self) -> None:
        job = Path("tools/delta/delta_smoke_qwen_0_5b.sbatch").read_text(
            encoding="utf-8"
        )
        wrapper = Path("tools/delta/submit_delta_smoke.sh").read_text(
            encoding="utf-8"
        )
        setup = Path("tools/delta/setup_delta_env.sh").read_text(encoding="utf-8")
        for directive in (
            "#SBATCH --partition=gpuA100x4",
            "#SBATCH --gpus-per-node=1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=32G",
            "#SBATCH --time=00:45:00",
        ):
            self.assertIn(directive, job)
        self.assertNotIn("#SBATCH --account=", job)
        self.assertNotIn("#SBATCH --exclusive", job)
        self.assertIn("--skip-binoculars", job)
        self.assertEqual(job.count("srun python -u run_experiment.py"), 2)
        self.assertIn("--compare-baseline", job)
        self.assertIn("--write-completion-marker", job)
        self.assertIn("torch.cuda.is_available()", job)
        self.assertIn('torch.version.cuda == "12.8"', job)
        self.assertIn('PYTHON_MODULE="${PYTHON_MODULE:-miniforge3-python}"', job)
        self.assertIn("nvidia-smi", job)
        self.assertIn('account="${ACCOUNT:-}"', wrapper)
        self.assertIn("--account=", wrapper)
        self.assertIn("Refusing to overwrite completed smoke run", wrapper)
        self.assertIn('"torch==2.11.0"', setup)
        self.assertIn("https://download.pytorch.org/whl/cu128", setup)
        self.assertLess(
            setup.index('"torch==2.11.0"'),
            setup.index("pip install -r requirements.txt"),
        )

    def test_no_network_synthetic_two_pass_smoke(self) -> None:
        with tempfile.TemporaryDirectory(prefix="delta-smoke-test-") as temp:
            output = Path(temp) / "run"
            process = subprocess.run(
                [
                    sys.executable,
                    "tools/validation/synthetic_delta_smoke.py",
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("SYNTHETIC DELTA SMOKE DRY RUN: PASS", process.stdout)

            report = json.loads(
                (output / "validation_report.json").read_text(encoding="utf-8")
            )
            marker = json.loads(
                (output / "smoke_validation.complete.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["resume_baseline_compared"])
            self.assertEqual(marker["status"], "complete")
            self.assertTrue(marker["resume_validated"])
            self.assertEqual(len(list(iter_jsonl(output / "selected_sources.jsonl"))), 12)
            self.assertEqual(
                len(list(iter_jsonl(output / "base_generations.jsonl"))), 12
            )
            self.assertEqual(
                len(list(iter_jsonl(output / "tail_candidate_cache.jsonl"))), 12
            )
            self.assertEqual(len(list(iter_jsonl(output / "clean_data.jsonl"))), 24)
            self.assertEqual(
                len(list(iter_jsonl(output / "contaminated_data.jsonl"))), 48
            )
            self.assertEqual(len(list(iter_jsonl(output / "data.jsonl"))), 72)
            self.assertEqual(
                len(list(iter_jsonl(output / "target_scores.jsonl"))), 72
            )
            with (output / "metrics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 144)


if __name__ == "__main__":
    unittest.main()
