"""Structural property calculation utilities.

Provides functions to compute lateral stiffnesses and structural parameters
for multi-storey building models.
"""

from typing import Union


def compute_storey_shear_stiffness(
    elastic_modulus: float,
    moment_of_inertia: float,
    storey_height: float,
    num_columns: int = 1,
) -> float:
    """Compute lateral shear stiffness for a storey assuming columns in double curvature.

    For columns fixed against rotation at both ends (shear building assumption):
        k_column = 12 * E * I / h^3
        k_storey = num_columns * k_column

    Parameters
    ----------
    elastic_modulus : float
        Modulus of elasticity (E) in Pa (N/m^2).
    moment_of_inertia : float
        Combined or single column second moment of area (I) in m^4.
    storey_height : float
        Storey height (h) in meters.
    num_columns : int, default=1
        Number of identical columns contributing to lateral resistance.

    Returns
    -------
    float
        Lateral shear stiffness (k) in N/m.

    Raises
    ------
    ValueError
        If any input parameter is non-positive.
    """
    if elastic_modulus <= 0.0:
        raise ValueError(f"Elastic modulus must be positive, got {elastic_modulus}")
    if moment_of_inertia <= 0.0:
        raise ValueError(f"Moment of inertia must be positive, got {moment_of_inertia}")
    if storey_height <= 0.0:
        raise ValueError(f"Storey height must be positive, got {storey_height}")
    if num_columns < 1:
        raise ValueError(f"Number of columns must be >= 1, got {num_columns}")

    k_col = 12.0 * elastic_modulus * moment_of_inertia / (storey_height**3)
    return float(num_columns * k_col)
