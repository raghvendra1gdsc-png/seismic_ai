"""Unit tests for seismic structural optimization and closed-loop physics verification."""

import unittest
import numpy as np

from src.structural.building import ShearBuilding
from src.earthquake.database import GroundMotionDatabase
from src.features.extractor import extract_features
from src.ml.models import LinearRidgeSurrogate
from src.optimization.problem import SeismicOptimizationProblem
from src.optimization.optimizer import DifferentialEvolutionOptimizer
from src.optimization.verifier import ClosedLoopVerifier


class TestOptimizationPipeline(unittest.TestCase):
    """Test constrained optimization and closed-loop verification."""

    def setUp(self):
        self.db = GroundMotionDatabase()
        self.record = self.db.get_record("El_Centro_1940_NS")

        # Mock surrogate fitted on a small sample
        bldg_base = ShearBuilding.from_uniform(3, 100000.0, 150000000.0, 3.5)
        feat_dict = extract_features(bldg_base, self.record)
        self.feat_cols = list(feat_dict.keys())

        # Fit a simple monotonic surrogate: drift decreases as stiffness increases
        rng = np.random.RandomState(42)
        X = rng.uniform(0.5, 2.0, size=(60, len(self.feat_cols)))
        # Target drift is inversely proportional to base stiffness (feature index 11)
        y = 0.015 / (X[:, 11] + 0.1) + rng.normal(0, 0.0005, size=60)

        self.surrogate = LinearRidgeSurrogate(alpha=1.0)
        self.surrogate.fit(X, y)

    def test_optimization_problem_evaluation(self):
        problem = SeismicOptimizationProblem(
            num_storeys=3,
            floor_mass_kg=100000.0,
            storey_height_m=3.5,
            design_earthquake=self.record,
            surrogate_model=self.surrogate,
            feature_columns=self.feat_cols,
            max_allowable_idr=0.012,
        )

        k_valid = np.array([2e8, 1.8e8, 1.5e8])
        res = problem.evaluate_surrogate(k_valid)
        self.assertIn("loss", res)
        self.assertIn("predicted_pidr", res)
        self.assertIn("total_stiffness", res)

    def test_optimizer_and_closed_loop_verification(self):
        problem = SeismicOptimizationProblem(
            num_storeys=3,
            floor_mass_kg=80000.0,
            storey_height_m=3.2,
            design_earthquake=self.record,
            surrogate_model=self.surrogate,
            feature_columns=self.feat_cols,
            max_allowable_idr=0.015,
            k_bounds=(8e7, 3e8),
        )

        optimizer = DifferentialEvolutionOptimizer(
            problem=problem,
            pop_size=15,
            max_generations=10,
            seed=42,
        )
        opt_res = optimizer.optimize()

        self.assertIn("optimal_stiffnesses", opt_res)
        self.assertEqual(len(opt_res["optimal_stiffnesses"]), 3)
        self.assertLessEqual(opt_res["history"][-1], opt_res["history"][0])

        # Closed-loop verification
        verif = ClosedLoopVerifier.verify_design(problem, opt_res["optimal_stiffnesses"])
        self.assertIn("physics_true_pidr", verif)
        self.assertIn("relative_pidr_error_percent", verif)
        self.assertIn("is_physics_verified_safe", verif)
        self.assertGreater(verif["physics_true_pidr"], 0.0)


if __name__ == "__main__":
    unittest.main()
