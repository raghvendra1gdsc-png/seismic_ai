"""Evaluation metrics and computational speedup benchmark for surrogate models."""

from typing import Dict, Any, Tuple
import time
import numpy as np

from src.ml.models import BaseSurrogate


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute standard regression performance metrics (R^2, RMSE, MAE, MAPE).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values from physics simulation.
    y_pred : np.ndarray
        Predicted values from surrogate model.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'r2', 'rmse', 'mae', 'mape'.
    """
    yt = np.asarray(y_true, dtype=np.float64).flatten()
    yp = np.asarray(y_pred, dtype=np.float64).flatten()

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else (1.0 if ss_res < 1e-12 else 0.0)

    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))

    # Mask out values near zero for stable MAPE
    non_zero = np.abs(yt) > 1e-6
    if np.any(non_zero):
        mape = float(np.mean(np.abs((yt[non_zero] - yp[non_zero]) / yt[non_zero])) * 100.0)
    else:
        mape = 0.0

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape_percent": mape,
    }


def benchmark_speedup(
    surrogate: BaseSurrogate,
    X_sample: np.ndarray,
    physics_time_per_sim_s: float = 0.045,
) -> Dict[str, float]:
    """Benchmark computational acceleration of the surrogate vs the physics solver.

    Parameters
    ----------
    surrogate : BaseSurrogate
        Trained surrogate model.
    X_sample : np.ndarray
        Feature matrix (e.g. 1000 samples).
    physics_time_per_sim_s : float, default=0.045
        Average wall-clock time for one full time-history dynamic simulation.

    Returns
    -------
    Dict[str, float]
        Inference time, equivalent physics time, and speedup factor.
    """
    n_samples = len(X_sample)
    
    # Warmup
    _ = surrogate.predict(X_sample[:min(10, n_samples)])

    t0 = time.perf_counter()
    _ = surrogate.predict(X_sample)
    t1 = time.perf_counter()

    surrogate_total_s = max(t1 - t0, 1e-6)
    surrogate_per_sample_s = surrogate_total_s / n_samples
    physics_total_s = physics_time_per_sim_s * n_samples
    speedup = physics_total_s / surrogate_total_s

    return {
        "num_samples": float(n_samples),
        "surrogate_total_time_s": surrogate_total_s,
        "surrogate_per_sample_ms": surrogate_per_sample_s * 1e3,
        "physics_equivalent_time_s": physics_total_s,
        "speedup_factor": speedup,
    }
