"""Park-Ang Damage Index Evaluator for Real-Time Structural Health Monitoring (SHM).

Implements the classical Park & Ang (1985) cumulative damage index:
    DI = (u_m / u_u) + (beta_pa / (Q_y * u_u)) * E_H

Categorizes damage states:
- DI < 0.20 : Slight Damage (Immediate Occupancy / Serviceable)
- 0.20 <= DI < 0.50 : Moderate Damage (Repairable, Minor Cracking)
- 0.50 <= DI < 1.00 : Severe Damage (Irreparable, Extensive Yielding)
- DI >= 1.00 : Collapse Limit State (Structural Failure)

References:
- Park, Y. J., & Ang, A. H. S. (1985). Mechanistic seismic damage model for reinforced concrete.
  J. Struct. Eng., ASCE, 111(4), 722-739.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class DamageEvaluationResult:
    """Detailed structural damage state evaluation."""
    park_ang_damage_index: float
    damage_state: str
    is_safe_for_occupancy: bool
    is_repairable: bool
    ductility_demand_ratio: float
    normalized_hysteretic_energy: float
    description: str


class ParkAngDamageEvaluator:
    """Evaluates Park-Ang seismic damage index for structural components and storeys.

    Parameters
    ----------
    beta_pa : float, default=0.08
        Non-negative cyclic degradation parameter (typically 0.05 to 0.15 for RC/Steel).
    ultimate_ductility_capacity : float, default=6.0
        Maximum displacement ductility capacity mu_u = u_u / u_y.
    """

    def __init__(
        self,
        beta_pa: float = 0.08,
        ultimate_ductility_capacity: float = 6.0,
    ) -> None:
        self.beta_pa = float(beta_pa)
        self.mu_u = float(ultimate_ductility_capacity)

    def evaluate_storey(
        self,
        max_drift_m: float,
        yield_disp_m: float,
        yield_force_n: float,
        hysteretic_energy_j: float,
    ) -> DamageEvaluationResult:
        """Compute Park-Ang Damage Index for a single storey."""
        u_m = max(abs(float(max_drift_m)), 1e-6)
        u_y = max(float(yield_disp_m), 1e-6)
        u_u = self.mu_u * u_y
        q_y = max(float(yield_force_n), 1e-3)
        e_h = max(float(hysteretic_energy_j), 0.0)

        # Ductility component: u_m / u_u
        term_ductility = u_m / u_u

        # Hysteretic energy component: beta * E_H / (Q_y * u_u)
        term_energy = (self.beta_pa * e_h) / (q_y * u_u)

        di = float(term_ductility + term_energy)

        # Classify damage state
        if di < 0.20:
            state = "Slight / Immediate Occupancy"
            safe = True
            repairable = True
            desc = "Structure exhibits minor elastic/micro-cracking. Fully safe for immediate re-occupancy."
        elif di < 0.50:
            state = "Moderate / Operational"
            safe = True
            repairable = True
            desc = "Moderate yielding observed. Structure is structurally stable and repairable."
        elif di < 1.00:
            state = "Severe / Life Safety Limit"
            safe = False
            repairable = False
            desc = "Severe plastic degradation and permanent residual drift. Evacuation required."
        else:
            state = "Collapse Limit State"
            safe = False
            repairable = False
            desc = "Partial or total structural collapse threshold exceeded."

        return DamageEvaluationResult(
            park_ang_damage_index=round(di, 4),
            damage_state=state,
            is_safe_for_occupancy=safe,
            is_repairable=repairable,
            ductility_demand_ratio=round(u_m / u_y, 3),
            normalized_hysteretic_energy=round(term_energy, 4),
            description=desc,
        )

    def evaluate_building_envelope(
        self,
        storey_drifts_m: List[float],
        storey_yield_disps_m: List[float],
        storey_yield_forces_n: List[float],
        storey_hysteretic_energies_j: List[float],
    ) -> Dict[str, Any]:
        """Compute global building damage index and per-storey breakdown."""
        n = len(storey_drifts_m)
        storey_results = []
        di_list = []

        for i in range(n):
            res = self.evaluate_storey(
                max_drift_m=storey_drifts_m[i],
                yield_disp_m=storey_yield_disps_m[i],
                yield_force_n=storey_yield_forces_n[i],
                hysteretic_energy_j=storey_hysteretic_energies_j[i],
            )
            storey_results.append(res)
            di_list.append(res.park_ang_damage_index)

        # Global building DI is weighted by storey energy dissipation or max storey DI
        max_di = float(np.max(di_list))
        avg_di = float(np.mean(di_list))

        return {
            "max_storey_damage_index": max_di,
            "average_building_damage_index": avg_di,
            "critical_storey_index": int(np.argmax(di_list)) + 1,
            "global_safety_status": "SAFE" if max_di < 0.50 else "UNSAFE",
            "storey_evaluations": [
                {
                    "storey": i + 1,
                    "damage_index": r.park_ang_damage_index,
                    "damage_state": r.damage_state,
                    "is_safe": r.is_safe_for_occupancy,
                }
                for i, r in enumerate(storey_results)
            ],
        }
