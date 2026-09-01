"""Unit tests for Incremental Dynamic Analysis (IDA) and Seismic Fragility Curves."""

import pytest
import numpy as np

from src.structural.building import ShearBuilding
from src.earthquake.database import GroundMotionDatabase
from src.fragility.ida import IncrementalDynamicAnalysis, IDAResult, IDACurve
from src.fragility.curves import SeismicFragilityModel, DEFAULT_DAMAGE_STATES


def test_ida_engine_and_fragility_curves():
    """Test full IDA suite execution and lognormal fragility curve fitting."""
    db = GroundMotionDatabase()
    rec1 = db.get_record("El_Centro_1940_NS")
    rec2 = db.get_record("Kobe_1995_NS")

    bldg = ShearBuilding.from_uniform(
        num_storeys=3,
        storey_mass=100000.0,
        storey_stiffness=1.5e8,
        storey_height=3.5,
    )

    ida = IncrementalDynamicAnalysis(
        building=bldg,
        im_min_g=0.10,
        im_max_g=1.00,
        num_scale_points=5,
    )

    ida_res = ida.run_suite(records=[rec1, rec2], use_surrogate=False)

    assert ida_res.record_count == 2
    assert len(ida_res.curves) == 2
    assert len(ida_res.median_50_pidr_pct) == 5
    # Monotonicity: drift should increase with intensity
    assert ida_res.median_50_pidr_pct[-1] > ida_res.median_50_pidr_pct[0]

    # Fragility fitting
    frag_model = SeismicFragilityModel()
    fitted_params = frag_model.fit_from_ida_result(ida_res)

    assert len(fitted_params) == len(DEFAULT_DAMAGE_STATES)
    # Medians should be ordered: DS1 (Slight) < DS2 (Moderate) < DS3 (Extensive) < DS4 (Collapse)
    assert fitted_params[0].median_capacity_theta_g <= fitted_params[1].median_capacity_theta_g
    assert fitted_params[1].median_capacity_theta_g <= fitted_params[2].median_capacity_theta_g

    # Evaluate probabilities
    im_test = np.array([0.1, 0.5, 1.0, 2.0])
    probs = frag_model.evaluate_probabilities(fitted_params, im_test)
    for ds_name, p_vals in probs.items():
        assert np.all(p_vals >= 0.0)
        assert np.all(p_vals <= 1.0)
        # Probability must increase monotonically with IM
        assert p_vals[-1] >= p_vals[0]
