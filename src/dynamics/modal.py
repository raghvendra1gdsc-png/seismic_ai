"""Modal analysis and eigenvalue solver for MDOF structural systems.

Computes natural frequencies, periods, mode shapes, modal participation factors,
and modal effective masses for linear elastic multi-storey shear buildings.
"""

from typing import Literal, Optional, Tuple, Dict, Any
import numpy as np

from src.structural.building import ShearBuilding

try:
    from scipy.linalg import eigh as scipy_eigh
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class ModalAnalysis:
    """Modal property extractor for MDOF shear buildings.

    Solves the generalized eigenvalue problem:
        K * phi_n = omega_n^2 * M * phi_n

    Parameters
    ----------
    building : ShearBuilding
        The structural building model containing M and K matrices.

    Attributes
    ----------
    num_modes : int
        Number of extracted vibration modes (N).
    circular_frequencies : np.ndarray
        Undamped natural circular frequencies [omega_1, ..., omega_N] in rad/s,
        sorted in ascending order.
    cyclic_frequencies : np.ndarray
        Undamped natural cyclic frequencies [f_1, ..., f_N] in Hz.
    periods : np.ndarray
        Natural periods [T_1, ..., T_N] in seconds.
    fundamental_period : float
        First mode period T_1 in seconds.
    """

    def __init__(self, building: ShearBuilding) -> None:
        if not isinstance(building, ShearBuilding):
            raise TypeError(f"Expected ShearBuilding instance, got {type(building).__name__}")

        self._building = building
        self._m_mat = building.mass_matrix
        self._k_mat = building.stiffness_matrix
        self._r_vec = building.influence_vector
        self.num_modes: int = building.num_storeys

        # Solve eigenvalue problem
        self._eigenvalues, self._mass_norm_phi = self._solve_eigenvalues()

        # Compute frequencies and periods
        # Numerical guard: ensure positive eigenvalues
        clipped_eigs = np.clip(self._eigenvalues, a_min=1e-12, a_max=None)
        self.circular_frequencies: np.ndarray = np.sqrt(clipped_eigs)
        self.cyclic_frequencies: np.ndarray = self.circular_frequencies / (2.0 * np.pi)
        self.periods: np.ndarray = 2.0 * np.pi / self.circular_frequencies
        self.fundamental_period: float = float(self.periods[0])

        # Modal participation and effective masses
        self._participation_factors, self._effective_masses = self._compute_modal_participation()

    def _solve_eigenvalues(self) -> Tuple[np.ndarray, np.ndarray]:
        """Solve K * phi = omega^2 * M * phi and return sorted (eigvals, mass_norm_phi)."""
        m_diag = np.diag(self._m_mat)
        
        if _HAS_SCIPY:
            # Generalized symmetric eigenvalue solver
            eigvals, eigvecs = scipy_eigh(self._k_mat, self._m_mat)
        else:
            # Standard symmetric transformation: K_tilde = M^{-1/2} K M^{-1/2}
            # M is diagonal and positive definite
            m_inv_sqrt = np.diag(1.0 / np.sqrt(m_diag))
            k_tilde = m_inv_sqrt @ self._k_mat @ m_inv_sqrt
            k_tilde = 0.5 * (k_tilde + k_tilde.T)  # Ensure exact symmetry

            eigvals, psi = np.linalg.eigh(k_tilde)
            # phi = M^{-1/2} * psi
            eigvecs = m_inv_sqrt @ psi

        # Sort in ascending order of frequency
        sort_idx = np.argsort(eigvals)
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, sort_idx]

        # Enforce mass-orthonormality: phi_n^T * M * phi_n = 1
        for n in range(self.num_modes):
            phi_n = eigvecs[:, n]
            mn = float(phi_n.T @ self._m_mat @ phi_n)
            phi_n = phi_n / np.sqrt(mn)

            # Consistent sign convention: top floor positive
            if phi_n[-1] < 0:
                phi_n = -phi_n
            eigvecs[:, n] = phi_n

        return eigvals, eigvecs

    def _compute_modal_participation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute modal participation factors and modal effective masses."""
        gammas = np.zeros(self.num_modes, dtype=np.float64)
        eff_masses = np.zeros(self.num_modes, dtype=np.float64)

        for n in range(self.num_modes):
            phi_n = self._mass_norm_phi[:, n]
            # For mass-normalized phi_n: L_n = phi_n^T * M * r, M_n = 1.0
            l_n = float(phi_n.T @ self._m_mat @ self._r_vec)
            gammas[n] = l_n
            eff_masses[n] = l_n**2

        return gammas, eff_masses

    def mode_shapes(
        self,
        normalization: Literal["mass", "roof", "max"] = "mass",
    ) -> np.ndarray:
        """Get the mode shapes matrix Phi (N x N) under desired normalization.

        Parameters
        ----------
        normalization : {"mass", "roof", "max"}, default="mass"
            - "mass": phi_n^T * M * phi_n = 1.0 (orthonormal)
            - "roof": phi_{N, n} = 1.0 (roof displacement is unity)
            - "max": max(|phi_{i, n}|) = 1.0 (maximum component is unity)

        Returns
        -------
        np.ndarray
            Matrix Phi (N x N), where column n corresponds to mode n.
        """
        phi = self._mass_norm_phi.copy()

        if normalization == "mass":
            return phi
        elif normalization == "roof":
            for n in range(self.num_modes):
                roof_val = phi[-1, n]
                if abs(roof_val) > 1e-14:
                    phi[:, n] = phi[:, n] / roof_val
            return phi
        elif normalization == "max":
            for n in range(self.num_modes):
                max_abs_idx = np.argmax(np.abs(phi[:, n]))
                max_val = phi[max_abs_idx, n]
                if abs(max_val) > 1e-14:
                    phi[:, n] = phi[:, n] / max_val
            return phi
        else:
            raise ValueError(
                f"Unknown normalization '{normalization}'. Must be 'mass', 'roof', or 'max'."
            )

    @property
    def participation_factors(self) -> np.ndarray:
        """Modal participation factors [Gamma_1, ..., Gamma_N] (mass-normalized)."""
        return self._participation_factors.copy()

    @property
    def effective_masses(self) -> np.ndarray:
        """Modal effective masses [M_1^*, ..., M_N^*] in kg."""
        return self._effective_masses.copy()

    @property
    def effective_mass_ratios(self) -> np.ndarray:
        """Modal effective mass ratios M_n^* / M_total."""
        return self._effective_masses / self._building.total_mass

    @property
    def cumulative_mass_ratios(self) -> np.ndarray:
        """Cumulative modal effective mass ratios (sums to 1.0)."""
        return np.cumsum(self.effective_mass_ratios)

    def to_dict(self) -> Dict[str, Any]:
        """Summary of modal properties."""
        return {
            "num_modes": self.num_modes,
            "circular_frequencies_rad_s": self.circular_frequencies.tolist(),
            "cyclic_frequencies_hz": self.cyclic_frequencies.tolist(),
            "periods_s": self.periods.tolist(),
            "fundamental_period_s": self.fundamental_period,
            "participation_factors": self.participation_factors.tolist(),
            "effective_masses_kg": self.effective_masses.tolist(),
            "effective_mass_ratios": self.effective_mass_ratios.tolist(),
            "cumulative_mass_ratios": self.cumulative_mass_ratios.tolist(),
        }

    def __repr__(self) -> str:
        return (
            f"ModalAnalysis(num_modes={self.num_modes}, "
            f"T1={self.fundamental_period:.4f}s, "
            f"mass_participation_mode1={self.effective_mass_ratios[0]*100:.1f}%)"
        )
