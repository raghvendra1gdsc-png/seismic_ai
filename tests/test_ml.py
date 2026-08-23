"""Unit tests for machine learning surrogate models and training pipeline."""

import unittest
import os
import tempfile
import numpy as np

from src.ml.models import (
    StandardScaler,
    LinearRidgeSurrogate,
    RandomForestSurrogate,
    GradientBoostingSurrogate,
    NeuralSurrogate,
)
from src.ml.evaluate import compute_regression_metrics, benchmark_speedup


class TestMLSurrogates(unittest.TestCase):
    """Test scaler, surrogate architectures, and metric calculators."""

    def setUp(self):
        # Synthetic nonlinear physics-like function:
        # y = 2.0 * x0 + 1.5 * (x1^2) / (x2 + 1) + noise
        rng = np.random.RandomState(42)
        n = 150
        self.X = rng.uniform(1.0, 5.0, size=(n, 4))
        self.y = 2.0 * self.X[:, 0] + 1.5 * (self.X[:, 1] ** 2) / (self.X[:, 2] + 1.0) + rng.normal(0, 0.05, size=n)

        # Train/Test split
        self.X_train, self.X_test = self.X[:110], self.X[110:]
        self.y_train, self.y_test = self.y[:110], self.y[110:]

    def test_standard_scaler(self):
        scaler = StandardScaler()
        x_norm = scaler.fit_transform(self.X_train)
        np.testing.assert_allclose(np.mean(x_norm, axis=0), 0.0, atol=1e-7)
        np.testing.assert_allclose(np.std(x_norm, axis=0), 1.0, atol=1e-7)

        # Inverse transform
        x_rec = scaler.inverse_transform(x_norm)
        np.testing.assert_allclose(x_rec, self.X_train, atol=1e-7)

    def test_linear_ridge_surrogate(self):
        model = LinearRidgeSurrogate(alpha=1.0)
        model.fit(self.X_train, self.y_train)
        r2 = model.score(self.X_test, self.y_test)
        self.assertGreater(r2, 0.85)

    def test_random_forest_surrogate(self):
        model = RandomForestSurrogate(n_estimators=30, max_depth=6, seed=42)
        model.fit(self.X_train, self.y_train)
        r2 = model.score(self.X_test, self.y_test)
        self.assertGreater(r2, 0.70)

    def test_gradient_boosting_surrogate(self):
        model = GradientBoostingSurrogate(n_estimators=35, learning_rate=0.1, max_depth=4, seed=42)
        model.fit(self.X_train, self.y_train)
        r2 = model.score(self.X_test, self.y_test)
        self.assertGreater(r2, 0.75)

    def test_neural_surrogate(self):
        model = NeuralSurrogate(hidden_layers=(32, 16), learning_rate=0.01, max_epochs=80, seed=42)
        model.fit(self.X_train, self.y_train)
        r2 = model.score(self.X_test, self.y_test)
        self.assertGreater(r2, 0.80)

    def test_model_serialization(self):
        model = LinearRidgeSurrogate(alpha=1.0)
        model.fit(self.X_train, self.y_train)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model.save(tmp_path)
            loaded_model = LinearRidgeSurrogate.load(tmp_path)
            y_pred_orig = model.predict(self.X_test)
            y_pred_load = loaded_model.predict(self.X_test)
            np.testing.assert_allclose(y_pred_orig, y_pred_load, atol=1e-9)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_metrics_and_speedup(self):
        metrics = compute_regression_metrics(self.y_test, self.y_test * 1.01)
        self.assertGreater(metrics["r2"], 0.95)
        self.assertLess(metrics["mape_percent"], 2.0)

        model = LinearRidgeSurrogate(alpha=1.0)
        model.fit(self.X_train, self.y_train)
        bench = benchmark_speedup(model, self.X_test, physics_time_per_sim_s=0.05)
        self.assertGreater(bench["speedup_factor"], 10.0)


if __name__ == "__main__":
    unittest.main()
