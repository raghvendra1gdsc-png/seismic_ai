"""Unit tests for Conformal Prediction and Sobol' Global Sensitivity Analysis."""

import pytest
import numpy as np

from src.uncertainty.conformal import ConformalPredictor, ConformalInterval
from src.uncertainty.sobol import SobolSensitivityAnalyzer, SobolAnalysisResult


def test_conformal_prediction():
    """Test distribution-free conformal prediction calibration and test coverage."""
    np.random.seed(42)
    N_cal = 100
    N_test = 200

    # Calibration true vs predicted
    y_cal = np.random.uniform(0.005, 0.030, size=N_cal)
    y_cal_pred = y_cal + np.random.normal(0, 0.002, size=N_cal)

    cp = ConformalPredictor(alpha=0.10)  # 90% confidence
    q = cp.calibrate(y_cal, y_cal_pred)
    assert q > 0.0

    # Test predictions
    y_test = np.random.uniform(0.005, 0.030, size=N_test)
    y_test_pred = y_test + np.random.normal(0, 0.002, size=N_test)

    eval_res = cp.evaluate_empirical_coverage(y_test, y_test_pred)
    assert eval_res["nominal_confidence_level"] == 0.90
    assert eval_res["empirical_test_coverage"] >= 0.82  # Empirical coverage close to 90%
    assert eval_res["is_valid_coverage"]

    # Point prediction interval
    interval = cp.predict_interval(0.015, target_name="PIDR")
    assert isinstance(interval, ConformalInterval)
    assert interval.lower_bound < interval.point_prediction < interval.upper_bound


def test_sobol_global_sensitivity_analysis():
    """Test Sobol' sensitivity analysis on Ishigami-type non-linear structural benchmark."""
    param_names = ["PGA_g", "T1_s", "StiffnessTaper", "DampingRatio"]
    param_bounds = [(0.1, 0.8), (0.3, 2.0), (0.5, 1.0), (0.02, 0.08)]

    analyzer = SobolSensitivityAnalyzer(
        parameter_names=param_names,
        parameter_bounds=param_bounds,
        num_samples_N=128,
    )

    # Model function where PGA and T1 are dominant
    def model_fn(X: np.ndarray) -> np.ndarray:
        pga = X[:, 0]
        t1 = X[:, 1]
        taper = X[:, 2]
        damp = X[:, 3]
        # Peak drift proxy: strongly dependent on PGA and T1, weakly on taper and damping
        return 0.02 * (pga * t1) / (np.sqrt(damp) * taper)

    res = analyzer.analyze(model_fn, target_name="PIDR")

    assert isinstance(res, SobolAnalysisResult)
    assert len(res.parameter_sensitivities) == 4
    # All S_i and S_Ti must be non-negative
    for s in res.parameter_sensitivities:
        assert s.first_order_index_Si >= 0.0
        assert s.total_effect_index_STi >= s.first_order_index_Si - 0.05

    # Dominant parameters (PGA_g and T1_s) should be in top ranking
    assert "PGA_g" in res.ranking_by_importance[:2] or "T1_s" in res.ranking_by_importance[:2]
