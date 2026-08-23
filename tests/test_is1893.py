"""Tests for BIS IS 1893 (Part 1): 2016 seismic code implementation."""

import pytest
import numpy as np

from src.structural.building import ShearBuilding
from src.standards.is1893 import (
    compute_is1893_spectral_shape,
    compute_is1893_approx_period,
    analyze_is1893_equivalent_static,
    generate_3way_comparison_table,
    ZONE_FACTORS,
)


def test_is1893_spectral_shape():
    """Verify IS 1893:2016 spectral shapes for Type I, II, and III soils."""
    # Type II (Medium Soil)
    # T = 0.05 -> 1 + 15(0.05) = 1.75
    sa_short = compute_is1893_spectral_shape(0.05, "Type II (Medium)")
    assert np.isclose(sa_short, 1.75, atol=1e-3)

    # T = 0.30 -> Plateau = 2.50
    sa_plateau = compute_is1893_spectral_shape(0.30, "Type II (Medium)")
    assert np.isclose(sa_plateau, 2.50, atol=1e-3)

    # T = 1.0 -> 1.36 / 1.0 = 1.36
    sa_decay = compute_is1893_spectral_shape(1.0, "Type II (Medium)")
    assert np.isclose(sa_decay, 1.36, atol=1e-3)

    # Type I (Rock)
    sa_rock = compute_is1893_spectral_shape(1.0, "Type I (Rock/Hard)")
    assert np.isclose(sa_rock, 1.00, atol=1e-3)

    # Type III (Soft)
    sa_soft = compute_is1893_spectral_shape(1.0, "Type III (Soft)")
    assert np.isclose(sa_soft, 1.67, atol=1e-3)


def test_is1893_approx_period():
    """Verify empirical period calculation for RC and Steel frames."""
    # 5-storey building (17.5m)
    h = 17.5
    ta_rc = compute_is1893_approx_period(h, "RC Moment Frame")
    expected_rc = 0.075 * (h ** 0.75)
    assert np.isclose(ta_rc, expected_rc, atol=1e-4)

    ta_steel = compute_is1893_approx_period(h, "Steel Moment Frame")
    expected_steel = 0.085 * (h ** 0.75)
    assert np.isclose(ta_steel, expected_steel, atol=1e-4)


def test_is1893_equivalent_static_analysis():
    """Verify equivalent static base shear and lateral force distribution."""
    # 5-storey building: uniform 100 tonnes, 150 MN/m, 3.5m height
    bldg = ShearBuilding.from_uniform(num_storeys=5, storey_mass=100000.0, storey_stiffness=150.0e6, storey_height=3.5)

    res = analyze_is1893_equivalent_static(bldg, zone="Zone IV", soil_type="Type II (Medium)", importance_factor=1.0, response_reduction=5.0)

    assert res.zone_factor_z == 0.24
    assert res.importance_factor_i == 1.0
    assert res.response_reduction_r == 5.0
    assert res.total_seismic_weight_kn > 0.0
    assert res.design_base_shear_kn > 0.0

    # Base shear = sum of lateral forces
    assert np.isclose(np.sum(res.storey_lateral_forces_n), res.design_base_shear_n, rtol=1e-3)

    # Base shear = storey shear at base
    assert np.isclose(res.storey_shear_forces_n[0], res.design_base_shear_n, rtol=1e-3)

    # Drift ratio strictly positive
    assert res.max_interstorey_drift_ratio > 0.0


def test_3way_comparison_table():
    """Verify 3-way comparison table output structure."""
    bldg = ShearBuilding.from_uniform(3, 100000.0, 100.0e6, 3.5)
    comp = generate_3way_comparison_table(
        building=bldg,
        zone="Zone IV",
        physics_peak_base_shear_n=500000.0,
        physics_max_pidr=0.005,
        surrogate_peak_base_shear_n=490000.0,
        surrogate_max_pidr=0.0048,
    )

    assert "table_rows" in comp
    assert len(comp["table_rows"]) >= 3
    assert comp["building_name"] == bldg.name
