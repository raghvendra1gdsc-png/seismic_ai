"""Closed-loop mechanics verification for surrogate-optimized structural designs."""

from typing import Dict, Any, Sequence
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.damping import RayleighDamping
from src.dynamics.solver import NewmarkSolver
from src.earthquake.record import GroundMotionRecord
from src.optimization.problem import SeismicOptimizationProblem


class ClosedLoopVerifier:
    """Verifies surrogate-optimized structural designs against the Phase 1 physics solver.

    Guarantees scientific honesty by checking whether the surrogate-optimized configuration
    truly satisfies structural performance and drift safety limits in the actual physical domain.
    """

    @staticmethod
    def verify_design(
        problem: SeismicOptimizationProblem,
        optimal_stiffnesses: Sequence[float],
    ) -> Dict[str, Any]:
        """Re-simulate optimal building using the physics-based Newmark integrator."""
        k_opt = np.asarray(optimal_stiffnesses, dtype=np.float64).flatten()
        masses = np.full(problem.num_storeys, problem.floor_mass_kg)
        masses[-1] = masses[-1] * 0.8  # lighter roof

        # 1. Physics solver simulation
        bldg_true = ShearBuilding(
            masses=masses,
            stiffnesses=k_opt,
            heights=problem.storey_height_m,
            name="OptimizedBuilding_PhysicsVerified",
        )

        damping = RayleighDamping.from_uniform_ratio(bldg_true, zeta=problem.damping_ratio)
        solver = NewmarkSolver.average_acceleration(bldg_true, damping)

        physics_response = solver.solve(
            ground_acceleration=problem.design_earthquake.acceleration,
            dt=problem.design_earthquake.dt,
        )

        true_pidr = physics_response.max_drift_ratio
        true_base_shear = physics_response.peak_base_shear
        true_roof_disp = physics_response.max_roof_displacement

        # 2. Surrogate evaluation
        surrogate_eval = problem.evaluate_surrogate(k_opt)
        surrogate_pidr = surrogate_eval["predicted_pidr"]

        # 3. Discrepancy analysis
        abs_err = abs(true_pidr - surrogate_pidr)
        rel_err_percent = (abs_err / true_pidr) * 100.0 if true_pidr > 0 else 0.0
        is_physically_safe = bool(true_pidr <= problem.max_allowable_idr)

        return {
            "num_storeys": problem.num_storeys,
            "optimal_stiffnesses_N_m": k_opt.tolist(),
            "total_stiffness_N_m": float(np.sum(k_opt)),
            "allowable_pidr_limit": problem.max_allowable_idr,
            "surrogate_predicted_pidr": surrogate_pidr,
            "physics_true_pidr": true_pidr,
            "absolute_pidr_error": abs_err,
            "relative_pidr_error_percent": rel_err_percent,
            "physics_peak_base_shear_kN": true_base_shear / 1e3,
            "physics_max_roof_disp_mm": true_roof_disp * 1e3,
            "is_surrogate_feasible": surrogate_eval["is_feasible"],
            "is_physics_verified_safe": is_physically_safe,
            "verification_status": "PASSED" if is_physically_safe and rel_err_percent < 10.0 else "WARNING",
        }
