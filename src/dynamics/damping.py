"""Rayleigh proportional damping formulation.

Computes mass- and stiffness-proportional damping coefficients (alpha, beta)
and assembles the classical damping matrix C = alpha * M + beta * K.
"""

from typing import Optional, Tuple, Union, Dict, Any
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.modal import ModalAnalysis


class RayleighDamping:
    """Rayleigh proportional damping model: C = alpha * M + beta * K.

    The damping ratio in mode n with circular frequency omega_n is:
        zeta_n = alpha / (2 * omega_n) + beta * omega_n / 2

    Parameters
    ----------
    building : ShearBuilding
        The structural building model providing mass (M) and stiffness (K) matrices.
    alpha : float
        Mass-proportional damping coefficient in s^-1.
    beta : float
        Stiffness-proportional damping coefficient in s.

    Raises
    ------
    ValueError
        If alpha or beta are negative, or if building is invalid.
    """

    def __init__(
        self,
        building: ShearBuilding,
        alpha: float,
        beta: float,
    ) -> None:
        if not isinstance(building, ShearBuilding):
            raise TypeError(f"Expected ShearBuilding instance, got {type(building).__name__}")

        if alpha < 0.0:
            raise ValueError(f"Rayleigh alpha must be non-negative (>= 0), got {alpha}")
        if beta < 0.0:
            raise ValueError(f"Rayleigh beta must be non-negative (>= 0), got {beta}")

        self._building = building
        self.alpha: float = float(alpha)
        self.beta: float = float(beta)

        # Assemble damping matrix C = alpha * M + beta * K
        self._damping_matrix = self.alpha * building.mass_matrix + self.beta * building.stiffness_matrix

    @classmethod
    def from_coefficients(
        cls,
        building: ShearBuilding,
        alpha: float,
        beta: float,
    ) -> "RayleighDamping":
        """Construct Rayleigh damping directly from alpha and beta coefficients."""
        return cls(building=building, alpha=alpha, beta=beta)

    @classmethod
    def from_frequencies(
        cls,
        building: ShearBuilding,
        zeta_i: float,
        zeta_j: float,
        omega_i: float,
        omega_j: float,
    ) -> "RayleighDamping":
        """Construct Rayleigh damping from two target damping ratios and frequencies.

        Parameters
        ----------
        building : ShearBuilding
            Structural building model.
        zeta_i : float
            Target damping ratio at frequency omega_i (e.g. 0.05 for 5%).
        zeta_j : float
            Target damping ratio at frequency omega_j.
        omega_i : float
            First target circular frequency in rad/s (omega_i < omega_j).
        omega_j : float
            Second target circular frequency in rad/s.

        Returns
        -------
        RayleighDamping
            Configured Rayleigh damping instance.
        """
        if zeta_i < 0.0 or zeta_j < 0.0:
            raise ValueError(f"Damping ratios must be non-negative. Got zeta_i={zeta_i}, zeta_j={zeta_j}")
        if omega_i <= 0.0 or omega_j <= 0.0:
            raise ValueError(f"Target frequencies must be strictly positive. Got omega_i={omega_i}, omega_j={omega_j}")
        if np.isclose(omega_i, omega_j):
            raise ValueError(f"Target frequencies omega_i ({omega_i}) and omega_j ({omega_j}) cannot be identical.")

        # Ensure omega_i < omega_j for clarity
        if omega_i > omega_j:
            omega_i, omega_j = omega_j, omega_i
            zeta_i, zeta_j = zeta_j, zeta_i

        denom = omega_j**2 - omega_i**2
        alpha = 2.0 * omega_i * omega_j * (zeta_i * omega_j - zeta_j * omega_i) / denom
        beta = 2.0 * (zeta_j * omega_j - zeta_i * omega_i) / denom

        if alpha < 0.0 or beta < 0.0:
            # Handle edge cases where target damping ratios result in negative coefficients
            raise ValueError(
                f"Specified (zeta_i={zeta_i}, zeta_j={zeta_j}) at (omega_i={omega_i:.3f}, omega_j={omega_j:.3f}) "
                f"produced negative Rayleigh coefficient(s): alpha={alpha:.4e}, beta={beta:.4e}."
            )

        return cls(building=building, alpha=alpha, beta=beta)

    @classmethod
    def from_modes(
        cls,
        building: ShearBuilding,
        zeta_i: float,
        zeta_j: float,
        mode_i: int = 1,
        mode_j: int = 2,
    ) -> "RayleighDamping":
        """Construct Rayleigh damping by matching target damping ratios at two modes.

        Parameters
        ----------
        building : ShearBuilding
            Structural building model.
        zeta_i : float
            Target damping ratio at mode i.
        zeta_j : float
            Target damping ratio at mode j.
        mode_i : int, default=1
            First mode index (1-based index: 1 <= mode_i <= N).
        mode_j : int, default=2
            Second mode index (1-based index: 1 <= mode_j <= N, mode_i != mode_j).
        """
        n_storeys = building.num_storeys
        if n_storeys == 1:
            # For 1-DOF system: alpha = 2 * zeta * omega, beta = 0 (or stiffness-only)
            modal = ModalAnalysis(building)
            w1 = modal.circular_frequencies[0]
            alpha = 2.0 * zeta_i * w1
            return cls(building=building, alpha=alpha, beta=0.0)

        if not (1 <= mode_i <= n_storeys) or not (1 <= mode_j <= n_storeys):
            raise ValueError(
                f"Mode indices must be between 1 and {n_storeys}. Got mode_i={mode_i}, mode_j={mode_j}."
            )
        if mode_i == mode_j:
            raise ValueError(f"mode_i and mode_j must be distinct, got {mode_i}.")

        modal = ModalAnalysis(building)
        w_i = float(modal.circular_frequencies[mode_i - 1])
        w_j = float(modal.circular_frequencies[mode_j - 1])

        return cls.from_frequencies(
            building=building,
            zeta_i=zeta_i,
            zeta_j=zeta_j,
            omega_i=w_i,
            omega_j=w_j,
        )

    @classmethod
    def from_uniform_ratio(
        cls,
        building: ShearBuilding,
        zeta: float = 0.05,
        mode_i: int = 1,
        mode_j: Optional[int] = None,
    ) -> "RayleighDamping":
        """Construct Rayleigh damping with constant target damping ratio zeta across reference modes.

        Parameters
        ----------
        building : ShearBuilding
            Structural building model.
        zeta : float, default=0.05
            Target damping ratio (e.g. 5% = 0.05).
        mode_i : int, default=1
            First reference mode.
        mode_j : int, optional
            Second reference mode. Defaults to min(3, N) or 2 if N >= 2.
        """
        n = building.num_storeys
        if n == 1:
            return cls.from_modes(building, zeta_i=zeta, zeta_j=zeta, mode_i=1, mode_j=1)

        if mode_j is None:
            mode_j = min(3, n) if n >= 3 else 2

        return cls.from_modes(building, zeta_i=zeta, zeta_j=zeta, mode_i=mode_i, mode_j=mode_j)

    @property
    def damping_matrix(self) -> np.ndarray:
        """Classical damping matrix C (N x N) in N*s/m."""
        return self._damping_matrix.copy()

    def damping_ratio_at_frequency(self, omega: float) -> float:
        """Compute damping ratio zeta for a given circular frequency omega in rad/s."""
        if omega <= 0.0:
            raise ValueError(f"Frequency must be strictly positive, got {omega}")
        return float(self.alpha / (2.0 * omega) + self.beta * omega / 2.0)

    def modal_damping_ratios(self, modal: Optional[ModalAnalysis] = None) -> np.ndarray:
        """Compute damping ratios for all natural modes of the building."""
        if modal is None:
            modal = ModalAnalysis(self._building)
        return np.array(
            [self.damping_ratio_at_frequency(w) for w in modal.circular_frequencies],
            dtype=np.float64,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Summary of damping parameters."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "modal_damping_ratios": self.modal_damping_ratios().tolist(),
        }

    def __repr__(self) -> str:
        return f"RayleighDamping(alpha={self.alpha:.4e}, beta={self.beta:.4e})"
