from pathlib import Path
from types import SimpleNamespace
import unittest

from tools.validation.check_detector_revision_gate import require_success

ROOT = Path(__file__).resolve().parents[1]


class GateLauncherTests(unittest.TestCase):
    def test_only_executed_successful_unskipped_tests_pass(self):
        require_success(SimpleNamespace(wasSuccessful=lambda: True, skipped=[], testsRun=2))
        for ok, skipped, count in ((False, [], 2), (True, [("cuda", "missing")], 2),
                                   (True, [], 0)):
            with self.subTest(ok=ok, skipped=skipped, count=count):
                with self.assertRaises(RuntimeError):
                    require_success(SimpleNamespace(wasSuccessful=lambda: ok,
                                                    skipped=skipped, testsRun=count))

    def test_wrapper_initializes_environment_and_checks_reference_before_tests(self):
        wrapper = (ROOT / "tools/delta/delta_detector_revision.sbatch").read_text()
        runner = "srun python -u tools/validation/check_detector_revision_gate.py"
        self.assertLess(wrapper.index("module reset"), wrapper.index("module load miniforge3-python"))
        self.assertLess(wrapper.index("module load miniforge3-python"), wrapper.index("/bin/activate"))
        self.assertLess(wrapper.index("/bin/activate"), wrapper.index(runner))
        self.assertLess(wrapper.index('! -r "$REFERENCE_METRICS"'), wrapper.index(runner))
        self.assertLess(wrapper.index(runner), wrapper.index('srun python -u RAID/revise_detectors.py'))
        self.assertIn('primary|gate|score|evaluate)', wrapper)


if __name__ == "__main__":
    unittest.main()
