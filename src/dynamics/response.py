"""Time-history dynamic response container and post-processing.

Encapsulates complete kinematic and kinetic response time-series for multi-storey
buildings, computing derived quantities such as interstorey drifts, drift ratios,
storey shears, base shears, and peak engineering demand parameters (EDPs).
"""

from typing import Dict, Any, Optional
import numpy as np

from src.structural.building import ShearBuilding


class DynamicResponse:
    """Complete dynamic time-history response container for an MDOF building.

    Stores relative and absolute time histories and computes derived engineering
    demand parameters (EDPs).

    Parameters
    ----------
    building : ShearBuilding
        The structural model.
    time : np.ndarray
        Time vector of length N_t in seconds.
    displacement : np.ndarray
        Relative floor displacements array of shape (N, N_t) in meters.
    velocity : np.ndarray
        Relative floor velocities array of shape (N, N_t) in m/s.
    acceleration : np.ndarray
        Relative floor accelerations array of shape (N, N_t) in m/s^2.
    ground_acceleration : Optional[np.ndarray], optional
        Ground acceleration array of length N_t in m/s^2. If None, assumed zero.
    """

    def __init__(
        self,
        building: ShearBuilding,
        time: np.ndarray,
        displacement: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        ground_acceleration: Optional[np.ndarray] = None,
    ) -> None:
        self.building = building
        self.time = np.asarray(time, dtype=np.float64).flatten()
        self.num_steps: int = self.time.size

        if self.num_steps < 2:
            raise ValueError(f"Time array must contain at least 2 points, got {self.num_steps}")

        self.dt: float = float(self.time[1] - self.time[0])
        n = building.num_storeys

        # Validate shapes: (N, N_t)
        self.displacement = np.asarray(displacement, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.acceleration = np.asarray(acceleration, dtype=np.float64)

        if self.displacement.shape != (n, self.num_steps):
            raise ValueError(
                f"Displacement shape mismatch: expected ({n}, {self.num_steps}), got {self.displacement.shape}"
            )
        if self.velocity.shape != (n, self.num_steps):
            raise ValueError(
                f"Velocity shape mismatch: expected ({n}, {self.num_steps}), got {self.velocity.shape}"
            )
        if self.acceleration.shape != (n, self.num_steps):
            raise ValueError(
                f"Acceleration shape mismatch: expected ({n}, {self.num_steps}), got {self.acceleration.shape}"
            )

        if ground_acceleration is not None:
            g_acc = np.asarray(ground_acceleration, dtype=np.float64).flatten()
            if g_acc.size != self.num_steps:
                raise ValueError(
                    f"Ground acceleration length ({g_acc.size}) must match time steps ({self.num_steps})."
                )
            self.ground_acceleration = g_acc
        else:
            self.ground_acceleration = np.zeros(self.num_steps, dtype=np.float64)

        # Compute absolute/total accelerations: u_total_ddot = u_ddot + r * a_g
        self.total_acceleration = self.acceleration + self.ground_acceleration[np.newaxis, :]

        # Compute derived kinetic and kinematic quantities
        self._drift = self._compute_drifts()
        self._drift_ratio = self._compute_drift_ratios()
        self._storey_shear = self._compute_storey_shears()

    def _compute_drifts(self) -> np.ndarray:
        """Compute interstorey displacement drift: Delta_i(t) = u_i(t) - u_{i-1}(t)."""
        n, nt = self.displacement.shape
        drifts = np.zeros((n, nt), dtype=np.float64)
        drifts[0, :] = self.displacement[0, :]
        if n > 1:
            drifts[1:, :] = self.displacement[1:, :] - self.displacement[:-1, :]
        return drifts

    def _compute_drift_ratios(self) -> np.ndarray:
        """Compute interstorey drift ratio (IDR): theta_i(t) = Delta_i(t) / h_i."""
        h_col = self.building.heights[:, np.newaxis]
        return self._drift / h_col

    def _compute_storey_shears(self) -> np.ndarray:
        """Compute storey shear forces: V_i(t) = k_i * Delta_i(t)."""
        k_col = self.building.stiffnesses[:, np.newaxis]
        return k_col * self._drift

    @property
    def interstorey_drift(self) -> np.ndarray:
        """Interstorey drift histories (N x N_t) in meters."""
        return self._drift.copy()

    @property
    def interstorey_drift_ratio(self) -> np.ndarray:
        """Interstorey drift ratio histories (N x N_t) (dimensionless / radians)."""
        return self._drift_ratio.copy()

    @property
    def storey_shear_force(self) -> np.ndarray:
        """Storey shear force histories (N x N_t) in Newtons."""
        return self._storey_shear.copy()

    @property
    def base_shear(self) -> np.ndarray:
        """Base shear force history V_b(t) = V_1(t) in Newtons."""
        return self._storey_shear[0, :].copy()

    @property
    def peak_floor_displacements(self) -> np.ndarray:
        """Peak floor displacements (PFD) for each floor [PFD_1, ..., PFD_N] in meters."""
        return np.max(np.abs(self.displacement), axis=1)

    @property
    def peak_floor_accelerations(self) -> np.ndarray:
        """Peak total floor accelerations (PFA) for each floor [PFA_1, ..., PFA_N] in m/s^2."""
        return np.max(np.abs(self.total_acceleration), axis=1)

    @property
    def peak_interstorey_drifts(self) -> np.ndarray:
        """Peak interstorey drifts for each storey [PID_1, ..., PID_N] in meters."""
        return np.max(np.abs(self._drift), axis=1)

    @property
    def peak_interstorey_drift_ratios(self) -> np.ndarray:
        """Peak interstorey drift ratios (PIDR) for each storey [PIDR_1, ..., PIDR_N]."""
        return np.max(np.abs(self._drift_ratio), axis=1)

    @property
    def peak_base_shear(self) -> float:
        """Maximum absolute base shear force in Newtons."""
        return float(np.max(np.abs(self.base_shear)))

    @property
    def max_roof_displacement(self) -> float:
        """Maximum absolute roof displacement in meters."""
        return float(np.max(np.abs(self.displacement[-1, :])))

    @property
    def max_drift_ratio(self) -> float:
        """Maximum absolute interstorey drift ratio across all storeys and time steps."""
        return float(np.max(np.abs(self._drift_ratio)))

    def summary(self) -> Dict[str, Any]:
        """Summary dictionary of peak response quantities."""
        return {
            "num_steps": self.num_steps,
            "duration_s": float(self.time[-1]),
            "dt_s": self.dt,
            "max_roof_displacement_m": self.max_roof_displacement,
            "max_roof_displacement_mm": self.max_roof_displacement * 1e3,
            "max_interstorey_drift_ratio": self.max_drift_ratio,
            "peak_base_shear_N": self.peak_base_shear,
            "peak_base_shear_kN": self.peak_base_shear / 1e3,
            "peak_floor_displacements_m": self.peak_floor_displacements.tolist(),
            "peak_floor_accelerations_m_s2": self.peak_floor_accelerations.tolist(),
            "peak_interstorey_drift_ratios": self.peak_interstorey_drift_ratios.tolist(),
        }

    def __repr__(self) -> str:
        return (
            f"DynamicResponse(duration={self.time[-1]:.2f}s, steps={self.num_steps}, "
            f"max_roof_disp={self.max_roof_displacement*1e3:.2f}mm, "
            f"max_IDR={self.max_drift_ratio*100:.3f}%, "
            f"peak_base_shear={self.peak_base_shear/1e3:.2f}kN)"
        )
