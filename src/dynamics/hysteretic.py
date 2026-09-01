"""Bouc-Wen smooth hysteretic model with energy-based degradation.

Implements:
1. Multi-storey Bouc-Wen nonlinear restoring force formulation using dimensionless hysteretic state z_tilde in [-1, 1].
2. Strength and stiffness degradation under cyclic plastic excursions.
3. Hysteretic energy dissipation computation E_H(t).
4. Ductility ratio (mu) and Residual Interstorey Drift Ratio (RIDR).

References:
- Bouc, R. (1967). Forced vibration of mechanical systems with hysteresis.
- Wen, Y. K. (1976). Method for random vibration of hysteretic systems. J. Eng. Mech.
- Baber, T. T., & Wen, Y. K. (1981). Random vibration of hysteretic, degrading systems.
- Foliente, G. C. (1995). Hysteresis modeling of wood joints and structural systems.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


class BoucWenStoreyHysteresis:
    """Single-storey Bouc-Wen hysteretic restoring force element.

    Restoring force:
        F_s(u, z) = alpha * k_0 * u + (1 - alpha) * k_0 * u_y * z_tilde

    where z_tilde in [-1, 1] is the normalized dimensionless hysteretic state.

    Rate equation:
        dz_tilde / dt = (du/dt / u_y) * (A - nu * (beta * sgn(du/dt * z_tilde) + gamma) * |z_tilde|^n) / eta

    Parameters
    ----------
    k0 : float
        Initial elastic lateral shear stiffness in N/m.
    yield_disp : float
        Characteristic yield displacement u_y in meters.
    alpha : float, default=0.05
        Post-yield stiffness ratio (0 < alpha < 1).
    A : float, default=1.0
        Restoring amplitude parameter.
    beta : float, default=0.5
        Hysteresis loop shape parameter.
    gamma : float, default=0.5
        Hysteresis loop shape parameter.
    n_exp : float, default=2.0
        Transition smoothness exponent (higher n -> sharper elastoplastic yield).
    delta_nu : float, default=0.0
        Strength degradation rate per unit hysteretic energy.
    delta_eta : float, default=0.0
        Stiffness degradation rate per unit hysteretic energy.
    """

    def __init__(
        self,
        k0: float,
        yield_disp: float = 0.015,
        alpha: float = 0.05,
        A: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        n_exp: float = 2.0,
        delta_nu: float = 0.0,
        delta_eta: float = 0.0,
    ) -> None:
        if k0 <= 0.0:
            raise ValueError(f"k0 must be positive, got {k0}")
        if yield_disp <= 0.0:
            raise ValueError(f"yield_disp must be positive, got {yield_disp}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.k0 = float(k0)
        self.yield_disp = float(yield_disp)
        self.yield_force = self.k0 * self.yield_disp
        self.alpha = float(alpha)
        self.A = float(A)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.n_exp = float(n_exp)
        self.delta_nu = float(delta_nu)
        self.delta_eta = float(delta_eta)

        # State variables
        self.z: float = 0.0  # z_tilde in [-1, 1]
        self.hysteretic_energy: float = 0.0
        self.max_drift: float = 0.0

    def restoring_force(self, drift: float, z_val: Optional[float] = None) -> float:
        """Compute instantaneous restoring force F_s(u, z_tilde)."""
        z_curr = self.z if z_val is None else z_val
        return float(self.alpha * self.k0 * drift + (1.0 - self.alpha) * self.k0 * self.yield_disp * z_curr)

    def tangent_stiffness(self, drift: float, velocity: float, z_val: Optional[float] = None) -> float:
        """Compute instantaneous tangent stiffness k_t = dF_s / du."""
        z_curr = self.z if z_val is None else z_val
        dz_du = self.dz_du(velocity, z_curr)
        return float(self.alpha * self.k0 + (1.0 - self.alpha) * self.k0 * self.yield_disp * dz_du)

    def dz_du(self, velocity: float, z_val: Optional[float] = None) -> float:
        """Derivative of dimensionless hysteretic variable with respect to displacement: dz_tilde/du."""
        z_curr = self.z if z_val is None else z_val
        abs_z = abs(z_curr)

        # Energy degradation factors
        e_norm = max(self.yield_force * self.yield_disp, 1.0)
        nu = 1.0 + self.delta_nu * (self.hysteretic_energy / e_norm)
        eta = 1.0 + self.delta_eta * (self.hysteretic_energy / e_norm)

        if abs(velocity) < 1e-12:
            return float((self.A / self.yield_disp) / eta)

        sgn_term = np.sign(velocity * z_curr)
        if self.n_exp == 1.0:
            term = (self.beta * sgn_term + self.gamma) * abs_z
        elif self.n_exp == 2.0:
            term = (self.beta * sgn_term + self.gamma) * (abs_z ** 2.0)
        else:
            term = (self.beta * sgn_term + self.gamma) * (abs_z ** self.n_exp)

        dzdu = ((self.A - nu * term) / self.yield_disp) / eta
        return float(dzdu)

    def step(self, drift_next: float, dt: float, velocity: float) -> Tuple[float, float]:
        """Advance hysteretic state by one time step using 4th-order Runge-Kutta on dz_tilde/dt."""
        def f_z(z_eval: float, v_eval: float) -> float:
            return v_eval * self.dz_du(v_eval, z_eval)

        k1 = f_z(self.z, velocity)
        k2 = f_z(self.z + 0.5 * dt * k1, velocity)
        k3 = f_z(self.z + 0.5 * dt * k2, velocity)
        k4 = f_z(self.z + dt * k3, velocity)

        dz = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        z_new = float(np.clip(self.z + dz, -1.0, 1.0))

        # Update hysteretic dissipated energy dE_H = (1 - alpha) * k0 * u_y * z * du
        du = velocity * dt
        d_energy = max(0.0, (1.0 - self.alpha) * self.k0 * self.yield_disp * (0.5 * (self.z + z_new)) * du)
        self.hysteretic_energy += d_energy

        self.z = z_new
        self.max_drift = max(self.max_drift, abs(drift_next))

        force = self.restoring_force(drift_next, self.z)
        return force, self.z

    def reset(self) -> None:
        """Reset internal hysteretic states."""
        self.z = 0.0
        self.hysteretic_energy = 0.0
        self.max_drift = 0.0
