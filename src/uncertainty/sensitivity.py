"""Monte Carlo parameter sensitivity and uncertainty quantification."""

from typing import Dict, Any, List, Optional
import numpy as np

from src.ml.models import BaseSurrogate


class MonteCarloSensitivity:
    """Evaluates surrogate model output uncertainty under noisy input features."""

    def __init__(self, surrogate: BaseSurrogate, seed: int = 42) -> None:
        self.surrogate = surrogate
        self.rng = np.random.RandomState(seed)

    def analyze_perturbations(
        self,
        nominal_features: np.ndarray,
        noise_levels: List[float] = [0.02, 0.05, 0.10, 0.15, 0.20],
        num_mc_samples: int = 400,
    ) -> Dict[str, Any]:
        """Perform Monte Carlo noise injection across various structural uncertainty levels.

        Parameters
        ----------
        nominal_features : np.ndarray
            1D nominal feature array of length D.
        noise_levels : List[float]
            List of Gaussian noise standard deviations (e.g. 0.05 for +/- 5%).
        num_mc_samples : int, default=400
            Number of Monte Carlo realizations per noise level.

        Returns
        -------
        Dict[str, Any]
            Statistics (mean, std, 5th and 95th percentiles, coefficient of variation).
        """
        x_nom = np.asarray(nominal_features, dtype=np.float64).flatten()
        nom_pred = float(self.surrogate.predict(x_nom.reshape(1, -1))[0])

        results = {
            "nominal_prediction": nom_pred,
            "noise_experiments": {},
        }

        for sigma in noise_levels:
            # Multiplicative noise: x_noisy = x_nom * (1 + N(0, sigma^2))
            noise_matrix = self.rng.normal(loc=0.0, scale=sigma, size=(num_mc_samples, len(x_nom)))
            x_noisy = x_nom[np.newaxis, :] * (1.0 + noise_matrix)

            preds = self.surrogate.predict(x_noisy)
            mean_pred = float(np.mean(preds))
            std_pred = float(np.std(preds))
            p5 = float(np.percentile(preds, 5))
            p95 = float(np.percentile(preds, 95))
            cov = float(std_pred / mean_pred) if mean_pred != 0 else 0.0

            results["noise_experiments"][f"noise_{int(sigma*100)}%"] = {
                "noise_sigma": sigma,
                "mean_prediction": mean_pred,
                "std_prediction": std_pred,
                "percentile_5": p5,
                "percentile_95": p95,
                "coeff_of_variation": cov,
            }

        return results
