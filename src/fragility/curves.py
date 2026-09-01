"""Lognormal Seismic Fragility Curve Generator.

Computes the conditional probability of exceeding discrete structural damage states:
    P(DS >= ds_i | IM) = Phi((ln(IM) - ln(theta_i)) / beta_i)

where:
- theta_i is the median capacity intensity (e.g. PGA in g).
- beta_i is the total lognormal dispersion (aleatory + epistemic).
- Phi is the standard normal cumulative distribution function (CDF).

References:
- FEMA P-58 (2018). Seismic Performance Assessment of Buildings.
- Baker, J. W. (2015). Efficient analytical fragility function fitting using dynamic structural analysis.
  Earthquake Spectra, 31(1), 579-599.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

from src.fragility.ida import IDAResult


@dataclass
class DamageStateLimit:
    """Damage state engineering definition."""
    state_id: str
    state_name: str
    drift_threshold_pct: float     # PIDR threshold in %
    description: str


DEFAULT_DAMAGE_STATES = [
    DamageStateLimit("DS1", "Slight / Immediate Occupancy (IO)", 0.50, "Hairline flexural cracking in beams/columns. Fully operational."),
    DamageStateLimit("DS2", "Moderate / Operational", 1.00, "Minor yielding, cover spalling. Structurally sound, repairable."),
    DamageStateLimit("DS3", "Extensive / Life Safety (LS)", 2.00, "Significant plastic hinging, severe concrete cracking. Non-collapse."),
    DamageStateLimit("DS4", "Complete / Collapse Prevention (CP)", 4.00, "Severe P-Delta degradation, partial/total collapse imminent."),
]


@dataclass
class FragilityCurveParameters:
    """Fitted lognormal fragility parameters for a single limit state."""
    state_id: str
    state_name: str
    drift_threshold_pct: float
    median_capacity_theta_g: float  # Median IM in g
    dispersion_beta: float          # Standard deviation of ln(IM)


class SeismicFragilityModel:
    """Generates and fits analytical lognormal fragility curves from IDA simulations."""

    def __init__(
        self,
        damage_states: Optional[List[DamageStateLimit]] = None,
        default_dispersion: float = 0.35,
    ) -> None:
        self.damage_states = damage_states or DEFAULT_DAMAGE_STATES
        self.default_dispersion = default_dispersion

    def fit_from_ida_result(self, ida_result: IDAResult) -> List[FragilityCurveParameters]:
        """Fit lognormal parameters theta and beta for each damage state from IDA curves."""
        fitted_params = []
        im_grid = ida_result.im_grid_g
        num_records = len(ida_result.curves)

        for ds in self.damage_states:
            threshold = ds.drift_threshold_pct
            # Find intensity IM at which each record first exceeds threshold
            capacity_ims = []
            for curve in ida_result.curves:
                idx = np.where(curve.edp_levels_pidr_pct >= threshold)[0]
                if len(idx) > 0:
                    capacity_ims.append(float(curve.im_levels_pga_g[idx[0]]))
                else:
                    # Extrapolate beyond maximum simulated IM
                    capacity_ims.append(float(im_grid[-1] * 1.5))

            log_capacities = np.log(capacity_ims)
            # Maximum Likelihood Estimation (MLE) of lognormal parameters
            mu_ln = float(np.mean(log_capacities))
            sigma_ln = float(np.std(log_capacities))

            theta = float(np.exp(mu_ln))
            # Combine aleatory record dispersion with standard epistemic modeling uncertainty (0.20)
            beta_total = float(np.sqrt(max(sigma_ln, 0.15)**2 + 0.20**2))

            fitted_params.append(
                FragilityCurveParameters(
                    state_id=ds.state_id,
                    state_name=ds.state_name,
                    drift_threshold_pct=ds.drift_threshold_pct,
                    median_capacity_theta_g=round(theta, 4),
                    dispersion_beta=round(beta_total, 4),
                )
            )

        return fitted_params

    def evaluate_probabilities(
        self,
        fitted_params: List[FragilityCurveParameters],
        im_eval_g: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Evaluate exceedance probability P(DS >= ds_i | IM) over an IM grid."""
        im_arr = np.asarray(im_eval_g, dtype=np.float64)
        prob_dict = {}

        for p in fitted_params:
            theta = p.median_capacity_theta_g
            beta = p.dispersion_beta
            # Phi((ln(IM) - ln(theta)) / beta)
            z = (np.log(np.maximum(im_arr, 1e-4)) - np.log(theta)) / beta
            probs = norm.cdf(z)
            prob_dict[p.state_name] = probs

        return prob_dict
