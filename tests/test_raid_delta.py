from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RaidDeltaContractTests(unittest.TestCase):
    def test_frozen_config_matches_scientific_protocol(self):
        config = json.loads((ROOT / "RAID" / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["selection"]["split_fractions"], {
            "clipping_tuning": 0.4,
            "calibration": 0.2,
            "test": 0.4,
        })
        self.assertEqual(config["evaluation"]["target_fpr"], 0.05)
        self.assertEqual(config["evaluation"]["bootstrap_repetitions"], 2000)
        self.assertEqual(config["evaluation"]["attack_weight"], 0.8)
        self.assertEqual(config["evaluation"]["clean_weight"], 0.2)
        self.assertEqual(config["models"]["falcon"]["id"], "tiiuae/falcon-7b")
        self.assertEqual(config["models"]["binoculars"]["observer"], "tiiuae/falcon-7b")
        self.assertEqual(
            config["models"]["binoculars"]["performer"],
            "tiiuae/falcon-7b-instruct",
        )
        self.assertEqual(config["scoring"]["max_tokens"], 512)

    def test_delta_wrapper_has_storage_gpu_and_resume_guards(self):
        wrapper = (ROOT / "RAID" / "delta_raid.sbatch").read_text(encoding="utf-8")
        submit = (ROOT / "RAID" / "submit_raid.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=gpuA100x4", wrapper)
        self.assertIn("#SBATCH --gpus-per-node=2", wrapper)
        self.assertNotIn("#SBATCH --gpus-per-node=4", wrapper)
        self.assertIn("/projects/bhuc/${USER}/venvs/delta-smoke", wrapper)
        self.assertIn('torch.version.cuda == "12.8"', wrapper)
        self.assertIn("rapidfuzz", wrapper)
        self.assertIn("RAID_DATA_PATH", wrapper)
        self.assertIn("/work/hdd/*", wrapper)
        self.assertIn('RAID_STAGE="${RAID_STAGE:-all}"', wrapper)
        self.assertIn('--gpus-per-node=2', submit)
        self.assertIn("bhuc-delta-gpu", submit)

    def test_sharded_delta_graph_separates_cpu_and_gpu_stages(self):
        prepare = (ROOT / "RAID" / "delta_raid_prepare.sbatch").read_text(encoding="utf-8")
        falcon = (ROOT / "RAID" / "delta_raid_falcon_shard.sbatch").read_text(encoding="utf-8")
        binoculars = (ROOT / "RAID" / "delta_raid_binoculars_shard.sbatch").read_text(encoding="utf-8")
        finalize = (ROOT / "RAID" / "delta_raid_finalize.sbatch").read_text(encoding="utf-8")
        submit = (ROOT / "RAID" / "submit_raid.sh").read_text(encoding="utf-8")

        self.assertIn("#SBATCH --partition=cpu", prepare)
        self.assertNotIn("--gpus-per-node", prepare)
        self.assertIn("--stage prepare", prepare)
        self.assertIn("--index-cache-dir", prepare)
        self.assertIn("#SBATCH --gpus-per-node=1", falcon)
        self.assertIn("--scorer falcon", falcon)
        self.assertIn("#SBATCH --gpus-per-node=2", binoculars)
        self.assertIn("--scorer binoculars", binoculars)
        self.assertIn("#SBATCH --partition=cpu", finalize)
        self.assertIn("--stage all", finalize)
        self.assertIn('--array="${ARRAY_RANGE}"', submit)
        self.assertIn('afterok:${PREP_JOB_ID}', submit)
        self.assertIn('afterok:${FALCON_JOB_ID}:${BINOCULARS_JOB_ID}', submit)
        self.assertIn('CPU_ACCOUNT="${CPU_ACCOUNT:-${GPU_ACCOUNT%-gpu}-cpu}"', submit)
        self.assertIn('--account="${GPU_ACCOUNT}"', submit)
        self.assertIn('--account="${CPU_ACCOUNT}"', submit)
        self.assertIn("last_workflow.env", submit)

    def test_delta_guide_requires_smoke_validation_before_full_run(self):
        guide = (ROOT / "RAID" / "DELTA_GUIDE.md").read_text(encoding="utf-8")
        self.assertIn("Mandatory four-shard smoke gate", guide)
        self.assertIn("LIMIT_SOURCES=64", guide)
        self.assertIn("BOOTSTRAP_REPETITIONS=100", guide)
        self.assertIn("RAID/validate_raid.py", guide)
        self.assertIn("2,000 bootstrap", guide)
        self.assertIn("NUM_SHARDS=4", guide)
        self.assertIn("CPU-only preparation", guide)


if __name__ == "__main__":
    unittest.main()
