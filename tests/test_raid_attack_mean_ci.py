import unittest
import numpy as np
from tools.reanalysis.raid_attack_mean_ci import joint_bootstrap


class JointAttackBootstrapTests(unittest.TestCase):
    def test_no_change_and_constant_gain(self):
        hits = np.zeros((10, 11, 2, 3), dtype=int)
        hits[:, :, 1, 1:] = 1
        point, ci, samples = joint_bootstrap(hits, ["a"]*5+["b"]*5, 100)
        np.testing.assert_array_equal(ci[:, 0], 0)
        np.testing.assert_array_equal(ci[:, 1], 1)
        self.assertEqual(point[1, 1], 1)

    def test_attack_correlation_preserved(self):
        hits = np.zeros((20, 11, 1, 3), dtype=int)
        hits[:10, :, 0, 1:] = 1
        _, _, samples = joint_bootstrap(hits, ["a"]*20, 1000, 7)
        # Eleven perfectly correlated variants behave as 20 clusters, not 220 independent rows.
        self.assertGreater(samples[:, 0, 0].std(), .09)
        np.testing.assert_array_equal(samples[:, :, 0], samples[:, :, 1])
        np.testing.assert_array_equal(samples, joint_bootstrap(hits, ["a"]*20, 1000, 7)[2])

    def test_stratum_sizes_fixed(self):
        hits = np.zeros((10, 11, 1, 3), dtype=int)
        hits[:3, :, :, 1:] = 1
        _, ci, _ = joint_bootstrap(hits, ["a"]*3+["b"]*7, 100)
        np.testing.assert_allclose(ci, .3)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            joint_bootstrap(np.zeros((3, 10, 1, 3)), ["a"]*3)


if __name__ == "__main__":
    unittest.main()
