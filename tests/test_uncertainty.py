"""Unit tests for uncertainty quantification and sensitivity analysis."""

import unittest
import numpy as np

from src.ml.models import LinearRidgeSurrogate
from src.uncertainty.sensitivity import MonteCarloSensitivity


class TestUncertaintyAnalysis(unittest.TestCase):
    """Test Monte Carlo perturbation sensitivity analysis."""

    def test_monte_carlo_sensitivity(self):
        # Fit a simple surrogate
        rng = np.random.RandomState(42)
        X = rng.uniform(1.0, 10.0, size=(100, 3))
        y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + 1.0 * X[:, 2]

        model = LinearRidgeSurrogate(alpha=1.0)
        model.fit(X, y)

        mc = MonteCarloSensitivity(surrogate=model, seed=42)
        nominal_x = np.array([5.0, 5.0, 5.0])

        results = mc.analyze_perturbations(
            nominal_features=nominal_x,
            noise_levels=[0.05, 0.15],
            num_mc_samples=200,
        )

        self.assertIn("nominal_prediction", results)
        self.assertIn("noise_5%", results["noise_experiments"])
        self.assertIn("noise_15%", results["noise_experiments"])

        # Check that 15% noise causes higher standard deviation than 5% noise
        std_5 = results["noise_experiments"]["noise_5%"]["std_prediction"]
        std_15 = results["noise_experiments"]["noise_15%"]["std_prediction"]
        self.assertGreater(std_15, std_5)

        # Check confidence interval bounds
        p5 = results["noise_experiments"]["noise_5%"]["percentile_5"]
        p95 = results["noise_experiments"]["noise_5%"]["percentile_95"]
        self.assertLess(p5, p95)


if __name__ == "__main__":
    unittest.main()
