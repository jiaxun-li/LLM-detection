#!/usr/bin/env python3
"""Run mandatory CPU/CUDA regressions inside the two-GPU revision gate."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require_success(result):
    if not result.wasSuccessful() or result.skipped or result.testsRun == 0:
        raise RuntimeError("Gate regressions must pass with no skipped tests")


def main():
    import sqlite3
    import torch
    from llm_detection.detector_revision import IMPLEMENTATION_VERSION
    from tests import test_detector_revision

    print("Python:", sys.executable, flush=True)
    print("SQLite:", sqlite3.sqlite_version, flush=True)
    print("Torch:", torch.__version__, flush=True)
    print("Implementation:", IMPLEMENTATION_VERSION, flush=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("The combined RAID gate requires two visible CUDA GPUs")
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}", flush=True)

    loader = unittest.TestLoader()
    if loader.loadTestsFromTestCase(test_detector_revision.CudaReplayTests).countTestCases() < 2:
        raise RuntimeError("Mandatory CUDA regression tests are missing")
    result = unittest.TextTestRunner(verbosity=2).run(
        loader.loadTestsFromModule(test_detector_revision))
    require_success(result)
    print("Gate CPU/CUDA regressions: PASS; real-model gate follows", flush=True)


if __name__ == "__main__":
    main()
