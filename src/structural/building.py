"""Multi-storey shear building model and structural matrices assembly.

Defines the ShearBuilding class which models an N-storey lumped mass shear frame
and constructs the mass (M) and stiffness (K) matrices.
"""

from typing import List, Sequence, Union, Dict, Any
import numpy as np


class ShearBuilding:
    """Multi-degree-of-freedom (MDOF) linear shear-building model.

    Assumptions:
    - Lumped mass at each floor level.
    - Floors/diaphragms are infinitely rigid in-plane and in flexure.
    - Columns deform in double curvature (pure shear frame behavior).
    - Fixed base at ground level (u_0 = 0).
    - Linear elastic behavior.
    - Single horizontal translation degree of freedom per floor.

    Parameters
    ----------
    masses : Sequence[float] | np.ndarray
        Floor masses [m_1, m_2, ..., m_N] in kg, ordered from 1st floor to roof.
    stiffnesses : Sequence[float] | np.ndarray
        Storey lateral stiffnesses [k_1, k_2, ..., k_N] in N/m, ordered from
        1st storey (ground to floor 1) to Nth storey (floor N-1 to floor N).
    heights : Sequence[float] | np.ndarray | float
        Storey heights [h_1, h_2, ..., h_N] in meters, or a single float for
        uniform storey height.
    name : str, optional
        Descriptive label for the building model.

    Raises
    ------
    ValueError
        If inputs have invalid dimensions, empty arrays, or non-positive values.
    """

    def __init__(
        self,
        masses: Union[Sequence[float], np.ndarray],
        stiffnesses: Union[Sequence[float], np.ndarray],
        heights: Union[Sequence[float], np.ndarray, float],
        name: str = "ShearBuilding",
    ) -> None:
        self.name = str(name)

        # Convert to 1D float arrays
        m_arr = np.asarray(masses, dtype=np.float64).flatten()
        k_arr = np.asarray(stiffnesses, dtype=np.float64).flatten()

        if m_arr.size == 0:
            raise ValueError("Masses array cannot be empty.")
        if k_arr.size == 0:
            raise ValueError("Stiffnesses array cannot be empty.")
        if m_arr.size != k_arr.size:
            raise ValueError(
                f"Mismatch between number of masses ({m_arr.size}) and stiffnesses ({k_arr.size})."
            )

        self.num_storeys: int = int(m_arr.size)

        if np.isscalar(heights) or isinstance(heights, (int, float)):
            h_val = float(heights)
            if h_val <= 0.0:
                raise ValueError(f"Storey height must be positive, got {h_val}")
            h_arr = np.full(self.num_storeys, h_val, dtype=np.float64)
        else:
            h_arr = np.asarray(heights, dtype=np.float64).flatten()
            if h_arr.size != self.num_storeys:
                raise ValueError(
                    f"Mismatch between number of storeys ({self.num_storeys}) and heights ({h_arr.size})."
                )

        # Validation: positive values
        if np.any(m_arr <= 0.0):
            invalid_indices = np.where(m_arr <= 0.0)[0]
            raise ValueError(
                f"All masses must be strictly positive (> 0). Invalid at indices {invalid_indices}: {m_arr[invalid_indices]}"
            )
        if np.any(k_arr <= 0.0):
            invalid_indices = np.where(k_arr <= 0.0)[0]
            raise ValueError(
                f"All stiffnesses must be strictly positive (> 0). Invalid at indices {invalid_indices}: {k_arr[invalid_indices]}"
            )
        if np.any(h_arr <= 0.0):
            invalid_indices = np.where(h_arr <= 0.0)[0]
            raise ValueError(
                f"All storey heights must be strictly positive (> 0). Invalid at indices {invalid_indices}: {h_arr[invalid_indices]}"
            )

        self._masses = m_arr
        self._stiffnesses = k_arr
        self._heights = h_arr

        # Assemble M and K matrices
        self._mass_matrix = self._assemble_mass_matrix()
        self._stiffness_matrix = self._assemble_stiffness_matrix()
        self._influence_vector = np.ones(self.num_storeys, dtype=np.float64)

    @classmethod
    def from_uniform(
        cls,
        num_storeys: int,
        storey_mass: float,
        storey_stiffness: float,
        storey_height: float,
        name: str = "UniformShearBuilding",
    ) -> "ShearBuilding":
        """Factory method to construct a building with identical storeys.

        Parameters
        ----------
        num_storeys : int
            Number of storeys (N >= 1).
        storey_mass : float
            Lumped mass per floor in kg.
        storey_stiffness : float
            Lateral shear stiffness per storey in N/m.
        storey_height : float
            Storey height in meters.
        name : str, optional
            Building name.

        Returns
        -------
        ShearBuilding
            Instantiated uniform shear building model.
        """
        if num_storeys < 1:
            raise ValueError(f"num_storeys must be >= 1, got {num_storeys}")
        masses = np.full(num_storeys, float(storey_mass))
        stiffnesses = np.full(num_storeys, float(storey_stiffness))
        return cls(masses=masses, stiffnesses=stiffnesses, heights=storey_height, name=name)

    def _assemble_mass_matrix(self) -> np.ndarray:
        """Assemble diagonal lumped mass matrix M."""
        return np.diag(self._masses)

    def _assemble_stiffness_matrix(self) -> np.ndarray:
        """Assemble tridiagonal lateral shear stiffness matrix K.

        For an N-storey building:
        K[0, 0] = k_1 + k_2 (or k_1 if N=1)
        K[i, i] = k_i + k_{i+1} for 0 < i < N-1
        K[N-1, N-1] = k_N
        K[i, i+1] = K[i+1, i] = -k_{i+1}
        """
        n = self.num_storeys
        k_mat = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            # Main diagonal: k_i + k_{i+1} (with k_{N+1} = 0)
            k_i = self._stiffnesses[i]
            k_next = self._stiffnesses[i + 1] if i + 1 < n else 0.0
            k_mat[i, i] = k_i + k_next

            # Off-diagonals: -k_{i+1}
            if i + 1 < n:
                k_mat[i, i + 1] = -self._stiffnesses[i + 1]
                k_mat[i + 1, i] = -self._stiffnesses[i + 1]

        return k_mat

    @property
    def masses(self) -> np.ndarray:
        """Floor masses [m_1, ..., m_N] in kg."""
        return self._masses.copy()

    @property
    def stiffnesses(self) -> np.ndarray:
        """Storey stiffnesses [k_1, ..., k_N] in N/m."""
        return self._stiffnesses.copy()

    @property
    def heights(self) -> np.ndarray:
        """Storey heights [h_1, ..., h_N] in meters."""
        return self._heights.copy()

    @property
    def mass_matrix(self) -> np.ndarray:
        """Lumped diagonal mass matrix M (N x N) in kg."""
        return self._mass_matrix.copy()

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Tridiagonal lateral stiffness matrix K (N x N) in N/m."""
        return self._stiffness_matrix.copy()

    @property
    def influence_vector(self) -> np.ndarray:
        """Ground motion influence vector r = [1, ..., 1]^T of length N."""
        return self._influence_vector.copy()

    @property
    def total_mass(self) -> float:
        """Total building mass in kg."""
        return float(np.sum(self._masses))

    @property
    def total_height(self) -> float:
        """Total building height in meters."""
        return float(np.sum(self._heights))

    @property
    def storey_elevations(self) -> np.ndarray:
        """Elevation of each floor diaphragm above ground in meters."""
        return np.cumsum(self._heights)

    def to_dict(self) -> Dict[str, Any]:
        """Convert building definition to a dictionary."""
        return {
            "name": self.name,
            "num_storeys": self.num_storeys,
            "masses": self._masses.tolist(),
            "stiffnesses": self._stiffnesses.tolist(),
            "heights": self._heights.tolist(),
            "total_mass": self.total_mass,
            "total_height": self.total_height,
        }

    def __repr__(self) -> str:
        return (
            f"ShearBuilding(name='{self.name}', num_storeys={self.num_storeys}, "
            f"total_mass={self.total_mass:.2e} kg, total_height={self.total_height:.2f} m)"
        )
