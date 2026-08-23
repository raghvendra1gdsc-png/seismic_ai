"""Standards and design code specifications for structural earthquake engineering."""

from src.standards.is1893 import (
    ZONE_FACTORS,
    SOIL_TYPES,
    RESPONSE_REDUCTION_FACTORS,
    IMPORTANCE_FACTORS,
    compute_is1893_spectral_shape,
    compute_is1893_approx_period,
    analyze_is1893_equivalent_static,
    generate_3way_comparison_table,
    IS1893DesignResult,
)

__all__ = [
    "ZONE_FACTORS",
    "SOIL_TYPES",
    "RESPONSE_REDUCTION_FACTORS",
    "IMPORTANCE_FACTORS",
    "compute_is1893_spectral_shape",
    "compute_is1893_approx_period",
    "analyze_is1893_equivalent_static",
    "generate_3way_comparison_table",
    "IS1893DesignResult",
]
