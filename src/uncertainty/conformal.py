"""Distribution-Free Split Conformal Prediction for Structural Response Quantification.

Provides non-parametric, finite-sample statistical coverage guarantees:
    P(Y_test in [y_hat(x) - q_alpha, y_hat(x) + q_alpha]) >= 1 - alpha

References:
- Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
- Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class ConformalInterval:
    """Predicted point estimate with conformal prediction coverage interval."""
    point_prediction: float
    lower_bound: float
    upper_bound: float
    interval_half_width: float
    nominal_confidence_level: float  # e.g. 0.90 (90% coverage)
    target_metric: str


class ConformalPredictor:
    """Split Conformal Prediction Calibrator for ML Surrogates.

    Parameters
    ----------
    alpha : float, default=0.10
        Significance level (alpha=0.10 corresponds to 90% confidence coverage).
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = float(alpha)
        self.confidence_level = 1.0 - self.alpha
        self.calibrated_quantile: Optional[float] = None
        self.calibration_sample_size: int = 0

    def calibrate(
        self,
        y_true_cal: np.ndarray,
        y_pred_cal: np.ndarray,
    ) -> float:
        """Compute the empirical non-conformity quantile q_hat on held-out calibration set."""
        y_true = np.asarray(y_true_cal, dtype=np.float64).flatten()
        y_pred = np.asarray(y_pred_cal, dtype=np.float64).flatten()
        n = len(y_true)
        if n < 10:
            raise ValueError(f"Calibration set too small: n={n}, need at least 10 samples.")

        # Absolute residual non-conformity scores R_i = |y_i - y_hat_i|
        scores = np.abs(y_true - y_pred)

        # Finite-sample adjusted quantile level: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, np.ceil((n + 1) * self.confidence_level) / n)
        self.calibrated_quantile = float(np.quantile(scores, level, method="higher"))
        self.calibration_sample_size = n
        return self.calibrated_quantile

    def predict_interval(
        self,
        y_pred: Union[float, np.ndarray],
        target_name: str = "PIDR",
    ) -> Union[ConformalInterval, List[ConformalInterval]]:
        """Construct calibrated conformal prediction intervals."""
        if self.calibrated_quantile is None:
            raise RuntimeError("ConformalPredictor must be calibrated before predict_interval().")

        q = self.calibrated_quantile
        if np.isscalar(y_pred):
            y_val = float(y_pred)
            return ConformalInterval(
                point_prediction=round(y_val, 5),
                lower_bound=round(max(0.0, y_val - q), 5),
                upper_bound=round(y_val + q, 5),
                interval_half_width=round(q, 5),
                nominal_confidence_level=self.confidence_level,
                target_metric=target_name,
            )

        y_arr = np.asarray(y_pred, dtype=np.float64).flatten()
        intervals = []
        for y_val in y_arr:
            intervals.append(
                ConformalInterval(
                    point_prediction=round(float(y_val), 5),
                    lower_bound=round(max(0.0, float(y_val) - q), 5),
                    upper_bound=round(float(y_val) + q, 5),
                    interval_half_width=round(q, 5),
                    nominal_confidence_level=self.confidence_level,
                    target_metric=target_name,
                )
            )
        return intervals

    def evaluate_empirical_coverage(
        self,
        y_true_test: np.ndarray,
        y_pred_test: np.ndarray,
    ) -> Dict[str, float]:
        """Verify empirical test coverage fraction: fraction where y_true is in [lower, upper]."""
        y_true = np.asarray(y_true_test, dtype=np.float64).flatten()
        y_pred = np.asarray(y_pred_test, dtype=np.float64).flatten()
        q = self.calibrated_quantile

        lower = np.maximum(0.0, y_pred - q)
        upper = y_pred + q

        covered = (y_true >= lower) & (y_true <= upper)
        empirical_coverage = float(np.mean(covered))
        avg_width = float(np.mean(upper - lower))

        return {
            "nominal_confidence_level": self.confidence_level,
            "empirical_test_coverage": round(empirical_coverage, 4),
            "is_valid_coverage": empirical_coverage >= (self.confidence_level - 0.05),
            "average_interval_width": round(avg_width, 5),
            "calibrated_quantile_q": round(q, 5),
        }
