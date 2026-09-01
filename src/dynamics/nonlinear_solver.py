"""Nonlinear Inelastic Structural Dynamics Solver with Newton-Raphson Iteration.

Solves the coupled MDOF nonlinear equations of motion:
    M * u_ddot(t) + C * u_dot(t) + F_s(u(t), z(t)) = -M * r * a_g(t)

where F_s is the non-linear hysteretic restoring force vector (Bouc-Wen model).
"""

from typing import Optional, Union, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.damping import RayleighDamping
from src.dynamics.hysteretic import BoucWenStoreyHysteresis


@dataclass
class InelasticDynamicResponse:
    """Complete response container for nonlinear inelastic time-history analysis."""

    time: np.ndarray
    displacement: np.ndarray          # Shape (N, N_t)
    velocity: np.ndarray              # Shape (N, N_t)
    acceleration: np.ndarray          # Shape (N, N_t)
    interstorey_drifts: np.ndarray    # Shape (N, N_t)
    restoring_forces: np.ndarray      # Shape (N, N_t)
    hysteretic_z: np.ndarray          # Shape (N, N_t)
    hysteretic_energies: np.ndarray   # Shape (N, N_t)
    max_pidr: float
    residual_drift_ratio: float       # RIDR (Permanent residual drift at t_end)
    peak_base_shear: float
    max_roof_displacement: float
    total_energy_dissipated_joules: float
    storey_ductilities: List[float]   # mu = Delta_max / u_yield

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_pidr": self.max_pidr,
            "max_pidr_pct": self.max_pidr * 100.0,
            "residual_drift_ratio": self.residual_drift_ratio,
            "residual_drift_ratio_pct": self.residual_drift_ratio * 100.0,
            "peak_base_shear_n": self.peak_base_shear,
            "peak_base_shear_kn": self.peak_base_shear / 1e3,
            "max_roof_displacement_m": self.max_roof_displacement,
            "total_hysteretic_energy_j": self.total_energy_dissipated_joules,
            "storey_ductilities": self.storey_ductilities,
        }


class NonlinearInelasticSolver:
    """MDOF Nonlinear Step-by-Step Newmark Newton-Raphson Dynamic Solver.

    Parameters
    ----------
    building : ShearBuilding
        Building structural model.
    damping : RayleighDamping | np.ndarray
        Viscous damping matrix C.
    yield_drift_ratio : float, default=0.005
        Yield drift ratio (u_y / h_i) defining the onset of inelastic yielding (0.5% standard RC/Steel).
    post_yield_ratio : float, default=0.05
        Post-yield stiffness ratio alpha = K_post / K_0.
    n_exp : float, default=2.0
        Bouc-Wen yield smoothness parameter.
    delta_nu : float, default=0.01
        Strength degradation parameter.
    delta_eta : float, default=0.01
        Stiffness degradation parameter.
    max_iter : int, default=30
        Maximum Newton-Raphson equilibrium iterations per step.
    tol : float, default=1e-5
        Newton-Raphson convergence tolerance on unbalance force norm.
    """

    def __init__(
        self,
        building: ShearBuilding,
        damping: Union[RayleighDamping, np.ndarray],
        yield_drift_ratio: float = 0.005,
        post_yield_ratio: float = 0.05,
        n_exp: float = 2.0,
        delta_nu: float = 0.01,
        delta_eta: float = 0.01,
        max_iter: int = 30,
        tol: float = 1e-5,
    ) -> None:
        self.building = building
        self.num_storeys: int = building.num_storeys

        if isinstance(damping, RayleighDamping):
            self.damping_matrix = damping.damping_matrix
        elif isinstance(damping, np.ndarray):
            self.damping_matrix = np.asarray(damping, dtype=np.float64)
        else:
            raise TypeError(f"Expected RayleighDamping or np.ndarray, got {type(damping).__name__}")

        self.m_mat = building.mass_matrix
        self.r_vec = building.influence_vector
        self.max_iter = max_iter
        self.tol = tol

        # Instantiate a Bouc-Wen hysteretic element for each storey
        self.hysteretic_elements: List[BoucWenStoreyHysteresis] = []
        for i in range(self.num_storeys):
            k_i = float(building.stiffnesses[i])
            h_i = float(building.heights[i])
            u_y = yield_drift_ratio * h_i
            elem = BoucWenStoreyHysteresis(
                k0=k_i,
                yield_disp=u_y,
                alpha=post_yield_ratio,
                n_exp=n_exp,
                delta_nu=delta_nu,
                delta_eta=delta_eta,
            )
            self.hysteretic_elements.append(elem)

    def _assemble_restoring_force_vector(
        self,
        drifts_iter: np.ndarray,
        drifts_prev: np.ndarray,
        v_drifts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble global restoring force vector F_s using trial hysteretic states."""
        n = self.num_storeys
        f_s = np.zeros(n, dtype=np.float64)
        storey_forces = np.zeros(n, dtype=np.float64)
        z_trial_vec = np.zeros(n, dtype=np.float64)

        for i in range(n):
            elem = self.hysteretic_elements[i]
            dzdu = elem.dz_du(v_drifts[i], elem.z)
            delta_drift = drifts_iter[i] - drifts_prev[i]
            z_trial = float(np.clip(elem.z + dzdu * delta_drift, -1.0, 1.0))
            z_trial_vec[i] = z_trial
            storey_forces[i] = elem.restoring_force(drifts_iter[i], z_val=z_trial)

        # Global equilibrium: F_s_i = V_i - V_{i+1}
        for i in range(n):
            v_i = storey_forces[i]
            v_next = storey_forces[i + 1] if i + 1 < n else 0.0
            f_s[i] = v_i - v_next

        return f_s, storey_forces

    def _assemble_tangent_stiffness_matrix(
        self,
        drifts_iter: np.ndarray,
        v_drifts: np.ndarray,
        z_trial_vec: np.ndarray,
    ) -> np.ndarray:
        """Assemble global tangent stiffness matrix K_t."""
        n = self.num_storeys
        k_t = np.zeros((n, n), dtype=np.float64)
        k_storey_t = np.zeros(n, dtype=np.float64)

        for i in range(n):
            k_storey_t[i] = self.hysteretic_elements[i].tangent_stiffness(
                drifts_iter[i], v_drifts[i], z_val=z_trial_vec[i]
            )

        for i in range(n):
            k_i = k_storey_t[i]
            k_next = k_storey_t[i + 1] if i + 1 < n else 0.0
            k_t[i, i] = k_i + k_next
            if i + 1 < n:
                k_t[i, i + 1] = -k_storey_t[i + 1]
                k_t[i + 1, i] = -k_storey_t[i + 1]

        return k_t

    def solve(
        self,
        ground_acceleration: np.ndarray,
        dt: float,
    ) -> InelasticDynamicResponse:
        """Solve nonlinear step-by-step dynamic response using Newton-Raphson Newmark integration."""
        ag = np.asarray(ground_acceleration, dtype=np.float64).flatten()
        nt = ag.size
        n = self.num_storeys
        t_vec = np.arange(nt) * dt

        # Reset all hysteretic elements
        for elem in self.hysteretic_elements:
            elem.reset()

        # Newmark average acceleration parameters (gamma=0.5, beta=0.25)
        gamma = 0.5
        beta = 0.25
        a0 = 1.0 / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)
        a6 = dt * (1.0 - gamma)
        a7 = gamma * dt

        # Output histories
        u_hist = np.zeros((n, nt), dtype=np.float64)
        v_hist = np.zeros((n, nt), dtype=np.float64)
        a_hist = np.zeros((n, nt), dtype=np.float64)
        drift_hist = np.zeros((n, nt), dtype=np.float64)
        force_hist = np.zeros((n, nt), dtype=np.float64)
        z_hist = np.zeros((n, nt), dtype=np.float64)
        e_hist = np.zeros((n, nt), dtype=np.float64)

        u_curr = np.zeros(n, dtype=np.float64)
        v_curr = np.zeros(n, dtype=np.float64)
        a_curr = -self.r_vec * ag[0]
        a_hist[:, 0] = a_curr

        drifts_prev = np.zeros(n, dtype=np.float64)

        for step_idx in range(nt - 1):
            ag_next = ag[step_idx + 1]
            p_next = -self.m_mat @ self.r_vec * ag_next

            u_iter = u_curr.copy()
            v_iter = v_curr.copy()
            a_iter = a_curr.copy()

            for iter_idx in range(self.max_iter):
                drifts_iter = np.zeros(n, dtype=np.float64)
                drifts_iter[0] = u_iter[0]
                if n > 1:
                    drifts_iter[1:] = u_iter[1:] - u_iter[:-1]

                v_drifts_iter = np.zeros(n, dtype=np.float64)
                v_drifts_iter[0] = v_iter[0]
                if n > 1:
                    v_drifts_iter[1:] = v_iter[1:] - v_iter[:-1]

                f_s_iter, _ = self._assemble_restoring_force_vector(drifts_iter, drifts_prev, v_drifts_iter)

                # Dynamic residual
                residual = p_next - self.m_mat @ a_iter - self.damping_matrix @ v_iter - f_s_iter
                res_norm = np.linalg.norm(residual)

                if res_norm < self.tol or iter_idx == self.max_iter - 1:
                    break

                z_trial_v = np.array([self.hysteretic_elements[i].z for i in range(n)])
                k_t_iter = self._assemble_tangent_stiffness_matrix(drifts_iter, v_drifts_iter, z_trial_v)
                k_eff = k_t_iter + a0 * self.m_mat + a1 * self.damping_matrix

                delta_u = np.linalg.solve(k_eff, residual)
                u_iter += delta_u

                a_iter = a0 * (u_iter - u_curr) - a2 * v_curr - a3 * a_curr
                v_iter = v_curr + a6 * a_curr + a7 * a_iter

            # Advance Bouc-Wen state variables at converged step
            drifts_final = np.zeros(n, dtype=np.float64)
            drifts_final[0] = u_iter[0]
            if n > 1:
                drifts_final[1:] = u_iter[1:] - u_iter[:-1]

            v_drifts_final = np.zeros(n, dtype=np.float64)
            v_drifts_final[0] = v_iter[0]
            if n > 1:
                v_drifts_final[1:] = v_iter[1:] - v_iter[:-1]

            for elem_idx in range(n):
                f_i, z_i = self.hysteretic_elements[elem_idx].step(
                    drift_next=drifts_final[elem_idx],
                    dt=dt,
                    velocity=v_drifts_final[elem_idx],
                )
                force_hist[elem_idx, step_idx + 1] = f_i
                z_hist[elem_idx, step_idx + 1] = z_i
                e_hist[elem_idx, step_idx + 1] = self.hysteretic_elements[elem_idx].hysteretic_energy

            drifts_prev = drifts_final.copy()
            u_curr = u_iter
            v_curr = v_iter
            a_curr = a_iter

            u_hist[:, step_idx + 1] = u_curr
            v_hist[:, step_idx + 1] = v_curr
            a_hist[:, step_idx + 1] = a_curr
            drift_hist[:, step_idx + 1] = drifts_final

        # Compute peak EDPs
        h_col = self.building.heights[:, np.newaxis]
        drift_ratios = np.abs(drift_hist / h_col)
        max_pidr = float(np.max(drift_ratios))

        residual_drift_ratio = float(np.max(np.abs(drift_hist[:, -1] / self.building.heights)))

        ductilities = []
        for idx in range(n):
            u_y = self.hysteretic_elements[idx].yield_disp
            mu_i = float(np.max(np.abs(drift_hist[idx, :])) / u_y)
            ductilities.append(round(mu_i, 3))

        total_e = float(np.sum(e_hist[:, -1]))
        peak_v_base = float(np.max(np.abs(force_hist[0, :])))
        max_roof_disp = float(np.max(np.abs(u_hist[-1, :])))

        return InelasticDynamicResponse(
            time=t_vec,
            displacement=u_hist,
            velocity=v_hist,
            acceleration=a_hist,
            interstorey_drifts=drift_hist,
            restoring_forces=force_hist,
            hysteretic_z=z_hist,
            hysteretic_energies=e_hist,
            max_pidr=max_pidr,
            residual_drift_ratio=residual_drift_ratio,
            peak_base_shear=peak_v_base,
            max_roof_displacement=max_roof_disp,
            total_energy_dissipated_joules=total_e,
            storey_ductilities=ductilities,
        )
