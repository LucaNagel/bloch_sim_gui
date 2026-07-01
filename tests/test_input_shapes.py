import unittest
import numpy as np

from blochsimulator import BlochSimulator, TissueParameters


class InputShapeTests(unittest.TestCase):
    def setUp(self):
        self.tissue = TissueParameters.gray_matter(3.0)
        self.dt = 1e-5
        self.ntime = 32
        self.time = np.arange(self.ntime) * self.dt

    def test_column_b1_and_positions_1d(self):
        """B1 column vectors and 1D position arrays should be accepted."""
        sim = BlochSimulator(use_parallel=False)
        b1 = np.zeros((self.ntime, 1), dtype=complex)
        b1[0, 0] = 0.01
        gradients = np.zeros((self.ntime, 3))
        positions = np.array([0.0, 0.0, 0.0])  # 1D position
        frequencies = np.array([0.0])

        result = sim.simulate(
            (b1, gradients, self.time),
            self.tissue,
            positions=positions,
            frequencies=frequencies,
            mode=0,
        )
        self.assertIn("mx", result)
        self.assertEqual(result["mx"].shape[0], positions.reshape(1, 3).shape[0])

    def test_gradient_1d_is_padded(self):
        """1D gradient arrays should be padded to 3 columns without error."""
        sim = BlochSimulator(use_parallel=False)
        b1 = np.zeros(self.ntime, dtype=complex)
        gradients = np.zeros(self.ntime)  # 1D gradient input
        positions = np.array([[0.0, 0.0, 0.0]])
        frequencies = np.array([0.0])

        result = sim.simulate(
            (b1, gradients, self.time),
            self.tissue,
            positions=positions,
            frequencies=frequencies,
            mode=0,
        )
        self.assertIn("mz", result)
        self.assertEqual(result["mz"].shape, (positions.shape[0], frequencies.shape[0]))

    def test_scalar_initial_magnetization_is_normalized(self):
        """Scalar Mz input should reach the compiled backend as a 3xN state."""
        sim = BlochSimulator(use_parallel=False)
        result = sim.simulate(
            (
                np.zeros(self.ntime, dtype=complex),
                np.zeros((self.ntime, 3)),
                self.time,
            ),
            self.tissue,
            initial_magnetization=0.5,
            mode=0,
        )
        self.assertGreater(float(result["mz"][0, 0]), 0.5)
        self.assertLess(float(result["mz"][0, 0]), 1.0)

    def test_vector_initial_magnetization_is_normalized(self):
        """A length-three vector should be accepted for every simulated spin."""
        sim = BlochSimulator(use_parallel=False)
        result = sim.simulate(
            (
                np.zeros(self.ntime, dtype=complex),
                np.zeros((self.ntime, 3)),
                self.time,
            ),
            self.tissue,
            frequencies=np.array([-10.0, 10.0]),
            initial_magnetization=np.array([0.0, 0.0, 0.5]),
            mode=0,
        )
        self.assertEqual(result["mz"].shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
