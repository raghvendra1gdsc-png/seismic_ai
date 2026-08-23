"""Ground motion record container and seismic intensity measures extraction.

Encapsulates earthquake ground acceleration time histories and computes standard
engineering intensity measures (IMs): PGA, PGV, PGD, Arias Intensity, Significant
Duration (D_5-95), Mean Period (Tm), and Predominant Period (Tp).
"""

from typing import Dict, Any, Optional
import numpy as np

GRAVITY = 9.80665  # m/s^2


class GroundMotionRecord:
    """Seismic ground motion record container.

    Parameters
    ----------
    name : str
        Earthquake / record identifier (e.g., 'El_Centro_1940_NS').
    dt : float
        Sampling time step in seconds.
    acceleration : np.ndarray | Sequence[float]
        Ground acceleration time history in m/s^2.
    metadata : Optional[Dict[str, Any]], optional
        Descriptive metadata (year, station, magnitude Mw, distance, etc.).
    """

    def __init__(
        self,
        name: str,
        dt: float,
        acceleration: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = str(name)
        self.dt = float(dt)

        if self.dt <= 0.0:
            raise ValueError(f"Sampling time step dt must be positive, got {self.dt}")

        acc_arr = np.asarray(acceleration, dtype=np.float64).flatten()
        if acc_arr.size < 2:
            raise ValueError(f"Acceleration array must have at least 2 points, got {acc_arr.size}")

        self._acceleration = acc_arr
        self.num_points: int = acc_arr.size
        self._time = np.arange(self.num_points, dtype=np.float64) * self.dt
        self.duration: float = float(self._time[-1])
        self.metadata: Dict[str, Any] = metadata or {}

        # Cached kinematic and intensity quantities
        self._velocity: Optional[np.ndarray] = None
        self._displacement: Optional[np.ndarray] = None

    @property
    def time(self) -> np.ndarray:
        """Time vector in seconds."""
        return self._time.copy()

    @property
    def acceleration(self) -> np.ndarray:
        """Ground acceleration history in m/s^2."""
        return self._acceleration.copy()

    @property
    def acceleration_g(self) -> np.ndarray:
        """Ground acceleration history in units of g."""
        return self._acceleration / GRAVITY

    @property
    def pga(self) -> float:
        """Peak Ground Acceleration (PGA) in m/s^2."""
        return float(np.max(np.abs(self._acceleration)))

    @property
    def pga_g(self) -> float:
        """Peak Ground Acceleration (PGA) in units of g."""
        return self.pga / GRAVITY

    @property
    def velocity(self) -> np.ndarray:
        """Ground velocity history in m/s (integrated from acceleration)."""
        if self._velocity is None:
            # Cumulative trapezoidal integration
            v = np.zeros(self.num_points, dtype=np.float64)
            v[1:] = np.cumsum(0.5 * (self._acceleration[:-1] + self._acceleration[1:]) * self.dt)
            self._velocity = v
        return self._velocity.copy()

    @property
    def pgv(self) -> float:
        """Peak Ground Velocity (PGV) in m/s."""
        return float(np.max(np.abs(self.velocity)))

    @property
    def displacement(self) -> np.ndarray:
        """Ground displacement history in meters (integrated from velocity)."""
        if self._displacement is None:
            v = self.velocity
            d = np.zeros(self.num_points, dtype=np.float64)
            d[1:] = np.cumsum(0.5 * (v[:-1] + v[1:]) * self.dt)
            self._displacement = d
        return self._displacement.copy()

    @property
    def pgd(self) -> float:
        """Peak Ground Displacement (PGD) in meters."""
        return float(np.max(np.abs(self.displacement)))

    @property
    def arias_intensity(self) -> float:
        """Arias Intensity Ia = (pi / (2*g)) * integral(a(t)^2 dt) in m/s."""
        a_sq = self._acceleration**2
        integral_a2 = float(np.sum(0.5 * (a_sq[:-1] + a_sq[1:]) * self.dt))
        return (np.pi / (2.0 * GRAVITY)) * integral_a2

    @property
    def husid_vector(self) -> np.ndarray:
        """Normalized cumulative Husid vector: H(t) in [0, 1]."""
        a_sq = self._acceleration**2
        cum_energy = np.zeros(self.num_points, dtype=np.float64)
        cum_energy[1:] = np.cumsum(0.5 * (a_sq[:-1] + a_sq[1:]) * self.dt)
        total_energy = cum_energy[-1]
        if total_energy > 0:
            return cum_energy / total_energy
        return np.zeros(self.num_points, dtype=np.float64)

    def significant_duration(self, p1: float = 0.05, p2: float = 0.95) -> float:
        """Compute Significant Duration D_{p1-p2} based on the Husid energy buildup.

        Parameters
        ----------
        p1 : float, default=0.05
            Lower percentile (e.g. 5% = 0.05).
        p2 : float, default=0.95
            Upper percentile (e.g. 95% = 0.95).

        Returns
        -------
        float
            Duration in seconds (t_p2 - t_p1).
        """
        husid = self.husid_vector
        if np.all(husid == 0.0):
            return 0.0

        idx1 = int(np.searchsorted(husid, p1))
        idx2 = int(np.searchsorted(husid, p2))
        idx1 = min(idx1, self.num_points - 1)
        idx2 = min(idx2, self.num_points - 1)
        return float(self._time[idx2] - self._time[idx1])

    @property
    def mean_period(self) -> float:
        """Mean Period Tm (Rathje et al., 1998) in seconds.

        Tm = sum(C_i^2 / f_i) / sum(C_i^2) for frequencies between 0.25 Hz and 20 Hz,
        where C_i are Fourier amplitude coefficients.
        """
        # FFT
        n_fft = 2 ** int(np.ceil(np.log2(self.num_points)))
        freqs = np.fft.rfftfreq(n_fft, d=self.dt)
        fft_vals = np.abs(np.fft.rfft(self._acceleration, n=n_fft)) * self.dt

        # Filter between 0.25 Hz and 20.0 Hz
        valid = (freqs >= 0.25) & (freqs <= 20.0)
        if not np.any(valid):
            return 0.0

        f_band = freqs[valid]
        c_band = fft_vals[valid]

        denom = np.sum(c_band**2)
        if denom == 0.0:
            return 0.0

        numer = np.sum((c_band**2) / f_band)
        return float(numer / denom)

    @property
    def predominant_period(self) -> float:
        """Predominant Period Tp corresponding to peak Fourier amplitude in seconds."""
        n_fft = 2 ** int(np.ceil(np.log2(self.num_points)))
        freqs = np.fft.rfftfreq(n_fft, d=self.dt)
        fft_vals = np.abs(np.fft.rfft(self._acceleration, n=n_fft)) * self.dt

        valid = freqs > 0.05
        if not np.any(valid):
            return 0.0

        max_idx = np.argmax(fft_vals[valid])
        peak_f = freqs[valid][max_idx]
        return float(1.0 / peak_f) if peak_f > 0 else 0.0

    def intensity_measures(self) -> Dict[str, float]:
        """Compute complete dictionary of scalar ground motion intensity measures."""
        return {
            "pga_m_s2": self.pga,
            "pga_g": self.pga_g,
            "pgv_m_s": self.pgv,
            "pgd_m": self.pgd,
            "arias_intensity_m_s": self.arias_intensity,
            "significant_duration_5_95_s": self.significant_duration(0.05, 0.95),
            "mean_period_s": self.mean_period,
            "predominant_period_s": self.predominant_period,
            "duration_s": self.duration,
            "dt_s": self.dt,
        }

    def scale_to_pga(self, target_pga_g: float) -> "GroundMotionRecord":
        """Return a new GroundMotionRecord linearly scaled to the target PGA in g."""
        current_pga_g = self.pga_g
        if current_pga_g <= 0.0:
            raise ValueError("Cannot scale a zero-acceleration record.")
        scale_factor = (target_pga_g * GRAVITY) / self.pga
        scaled_acc = self._acceleration * scale_factor
        new_meta = self.metadata.copy()
        new_meta["scaled_from"] = self.name
        new_meta["scale_factor"] = scale_factor
        new_meta["target_pga_g"] = target_pga_g
        return GroundMotionRecord(
            name=f"{self.name}_scaled_{target_pga_g:.2f}g",
            dt=self.dt,
            acceleration=scaled_acc,
            metadata=new_meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Summary dictionary."""
        d = {
            "name": self.name,
            "num_points": self.num_points,
            "metadata": self.metadata,
        }
        d.update(self.intensity_measures())
        return d

    def __repr__(self) -> str:
        return (
            f"GroundMotionRecord('{self.name}', duration={self.duration:.2f}s, "
            f"PGA={self.pga_g:.3f}g, PGV={self.pgv*100:.1f}cm/s, Ia={self.arias_intensity:.2f}m/s)"
        )
