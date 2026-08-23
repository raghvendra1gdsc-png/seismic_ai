"""Elastic response spectra generation for earthquake ground motions.

Computes Pseudo-Spectral Acceleration (Sa), Pseudo-Spectral Velocity (Sv),
and Relative Displacement Spectrum (Sd) for specified damping ratios (zeta).
"""

from typing import Sequence, Optional, Union, Dict, Any
import numpy as np

from src.earthquake.record import GroundMotionRecord, GRAVITY


class ResponseSpectrum:
    """Elastic response spectrum computer for an earthquake record.

    Parameters
    ----------
    record : GroundMotionRecord
        Input ground motion accelerogram.
    damping_ratio : float, default=0.05
        Viscous damping ratio zeta (5% = 0.05).
    periods : Optional[Sequence[float] | np.ndarray], optional
        Array of natural periods T in seconds. Defaults to standard 100-point
        log-spaced array from 0.02s to 4.0s.
    """

    def __init__(
        self,
        record: GroundMotionRecord,
        damping_ratio: float = 0.05,
        periods: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> None:
        if not isinstance(record, GroundMotionRecord):
            raise TypeError(f"Expected GroundMotionRecord instance, got {type(record).__name__}")
        if damping_ratio < 0.0:
            raise ValueError(f"Damping ratio must be non-negative, got {damping_ratio}")

        self.record = record
        self.damping_ratio: float = float(damping_ratio)

        if periods is None:
            # Standard structural engineering period grid: 0.02s to 4.0s (100 points)
            t_short = np.linspace(0.02, 0.2, 20)
            t_mid = np.linspace(0.22, 2.0, 60)
            t_long = np.linspace(2.1, 4.0, 20)
            self.periods = np.unique(np.concatenate([t_short, t_mid, t_long]))
        else:
            p_arr = np.asarray(periods, dtype=np.float64).flatten()
            if np.any(p_arr <= 0.0):
                raise ValueError("All spectrum periods must be strictly positive (> 0).")
            self.periods = np.sort(p_arr)

        self.num_periods: int = self.periods.size

        # Compute spectra
        self.sd, self.sv, self.sa = self._compute_spectra()
        self.sa_g = self.sa / GRAVITY

    def _compute_spectra(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Sd, Sv, and Sa across all periods using step-by-step Newmark-beta SDOF integration."""
        ag = self.record.acceleration
        dt = self.record.dt
        nt = ag.size
        zeta = self.damping_ratio

        # Integration constants (Average Acceleration: gamma=0.5, beta=0.25)
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

        omegas = 2.0 * np.pi / self.periods
        m = 1.0
        k = omegas**2
        c = 2.0 * zeta * omegas

        k_hat = k + a0 * m + a1 * c
        inv_k_hat = 1.0 / k_hat

        # Vectorized states for all periods simultaneously
        u_curr = np.zeros(self.num_periods, dtype=np.float64)
        v_curr = np.zeros(self.num_periods, dtype=np.float64)
        a_curr = np.full(self.num_periods, -ag[0], dtype=np.float64)
        u_max = np.zeros(self.num_periods, dtype=np.float64)

        for i in range(nt - 1):
            ag_next = ag[i + 1]
            p_hat = (
                -ag_next
                + (a0 * u_curr + a2 * v_curr + a3 * a_curr)
                + c * (a1 * u_curr + a4 * v_curr + a5 * a_curr)
            )

            u_next = p_hat * inv_k_hat
            a_next = a0 * (u_next - u_curr) - a2 * v_curr - a3 * a_curr
            v_next = v_curr + a6 * a_curr + a7 * a_next

            np.maximum(u_max, np.abs(u_next), out=u_max)

            u_curr = u_next
            v_curr = v_next
            a_curr = a_next

        sd_arr = u_max
        sv_arr = omegas * sd_arr          # Pseudo-velocity Sv = wn * Sd
        sa_arr = (omegas**2) * sd_arr     # Pseudo-acceleration Sa = wn^2 * Sd

        return sd_arr, sv_arr, sa_arr

    def get_sa(self, period: float) -> float:
        """Interpolate pseudo-spectral acceleration Sa (in m/s^2) at arbitrary period T."""
        if period <= 0.0:
            raise ValueError(f"Period must be positive, got {period}")
        return float(np.interp(period, self.periods, self.sa))

    def get_sa_g(self, period: float) -> float:
        """Interpolate pseudo-spectral acceleration Sa/g (dimensionless) at arbitrary period T."""
        return self.get_sa(period) / GRAVITY

    def get_sd(self, period: float) -> float:
        """Interpolate spectral displacement Sd (in meters) at arbitrary period T."""
        if period <= 0.0:
            raise ValueError(f"Period must be positive, got {period}")
        return float(np.interp(period, self.periods, self.sd))

    def to_dict(self) -> Dict[str, Any]:
        """Summary dictionary."""
        return {
            "record_name": self.record.name,
            "damping_ratio": self.damping_ratio,
            "periods": self.periods.tolist(),
            "sa_m_s2": self.sa.tolist(),
            "sa_g": self.sa_g.tolist(),
            "sd_m": self.sd.tolist(),
            "sv_m_s": self.sv.tolist(),
            "pga_g": self.record.pga_g,
        }

    def __repr__(self) -> str:
        return (
            f"ResponseSpectrum(record='{self.record.name}', zeta={self.damping_ratio*100:.1f}%, "
            f"PGA={self.record.pga_g:.3f}g, max_Sa={np.max(self.sa_g):.3f}g)"
        )
