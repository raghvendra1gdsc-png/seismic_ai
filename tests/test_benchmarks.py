"""Tests for real benchmark buildings (SAC Steel Project & IIT Campus Frames)."""

import pytest
import numpy as np

from src.structural.benchmarks import (
    get_sac_3storey_building,
    get_sac_9storey_building,
    get_iit_delhi_4storey_building,
    get_iit_roorkee_6storey_building,
    list_benchmark_buildings,
)
from src.dynamics.modal import ModalAnalysis


def test_sac_3storey_benchmark():
    """Verify SAC 3-Story LA benchmark building modal properties."""
    bldg = get_sac_3storey_building()
    assert bldg.num_storeys == 3
    assert np.isclose(bldg.total_height, 3.96 * 3, atol=1e-2)

    modal = ModalAnalysis(bldg)
    # Target fundamental period T1 ~ 1.01 s (+/- 10%)
    assert 0.85 <= modal.fundamental_period <= 1.20
    # Mode 1 mass participation should dominate (> 80%)
    assert modal.effective_mass_ratios[0] > 0.80


def test_sac_9storey_benchmark():
    """Verify SAC 9-Story LA benchmark building modal properties."""
    bldg = get_sac_9storey_building()
    assert bldg.num_storeys == 9

    modal = ModalAnalysis(bldg)
    # Target fundamental period T1 ~ 2.27 s (+/- 15%)
    assert 1.80 <= modal.fundamental_period <= 2.60
    # Period ratio T2/T1 ~ 0.33 to 0.40 (typical shear frame behavior)
    assert 0.25 <= (modal.periods[1] / modal.fundamental_period) <= 0.45


def test_iit_campus_benchmarks():
    """Verify IIT Delhi and IIT Roorkee benchmark frames."""
    delhi_4st = get_iit_delhi_4storey_building()
    assert delhi_4st.num_storeys == 4
    modal_delhi = ModalAnalysis(delhi_4st)
    assert 0.35 <= modal_delhi.fundamental_period <= 0.70

    roorkee_6st = get_iit_roorkee_6storey_building()
    assert roorkee_6st.num_storeys == 6
    modal_roorkee = ModalAnalysis(roorkee_6st)
    assert 0.50 <= modal_roorkee.fundamental_period <= 1.00


def test_list_benchmark_buildings():
    """Verify benchmark catalog listing."""
    catalog = list_benchmark_buildings()
    assert len(catalog) >= 4
    for entry in catalog:
        assert "label" in entry
        assert "fundamental_period_T1_s" in entry
        assert entry["fundamental_period_T1_s"] > 0.0
