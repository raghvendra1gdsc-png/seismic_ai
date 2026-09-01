"""Unit tests for Physics-Informed Neural Network (PINN) surrogate."""

import pytest
import numpy as np
import tempfile
import os

from src.ml.pinn import PhysicsInformedSurrogate


def test_pinn_surrogate_training_and_prediction():
    """Test PINN training, prediction, and physics residual loss calculation."""
    np.random.seed(42)
    N = 100
    n_features = 29

    # Synthetic features
    X = np.random.uniform(0.5, 5.0, size=(N, n_features))
    # Target: PIDR proxy
    y = 0.005 * (X[:, 0] / X[:, 1]) + np.random.normal(0, 1e-4, size=N)

    feature_names = [f"feat_{i}" for i in range(n_features)]
    feature_names[24] = "Sd_T1_m"
    feature_names[2] = "total_height_m"

    pinn = PhysicsInformedSurrogate(
        hidden_layer_sizes=(32, 16),
        lambda_physics=0.2,
        max_iter=100,
    )

    pinn.fit(X, y, feature_names=feature_names, target_name="target_max_pidr")
    assert pinn.is_fitted

    preds = pinn.predict(X[:5])
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))

    phys_res = pinn.compute_residual_loss(X[:10], feature_names=feature_names, target_name="target_max_pidr")
    assert np.isfinite(phys_res)
    assert phys_res >= 0.0

    # Serialization test
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "pinn_test.pkl")
        pinn.save(model_path)
        loaded_pinn = PhysicsInformedSurrogate.load(model_path)
        loaded_preds = loaded_pinn.predict(X[:5])
        np.testing.assert_allclose(preds, loaded_preds, rtol=1e-5)
