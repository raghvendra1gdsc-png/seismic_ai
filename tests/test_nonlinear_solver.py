"""Unit tests for Bouc-Wen hysteresis and MDOF nonlinear inelastic solver."""

import pytest
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.damping import RayleighDamping
from src.dynamics.hysteretic import BoucWenStoreyHysteresis
from src.dynamics.nonlinear_solver import NonlinearInelasticSolver, InelasticDynamicResponse
from src.dynamics.solver import NewmarkSolver


def test_bouc_wen_single_storey_hysteresis():
    """Test single-storey Bouc-Wen hysteretic element under harmonic cyclic drift."""
    k0 = 1.0e8  # 100 MN/m
    u_y = 0.01  # 10 mm yield displacement
    elem = BoucWenStoreyHysteresis(k0=k0, yield_disp=u_y, alpha=0.05)

    dt = 0.005
    t = np.arange(0.0, 2.0, dt)
    u_vals = 0.025 * np.sin(2.0 * np.pi * t)
    v_vals = np.gradient(u_vals, dt)

    forces = []
    for i in range(len(t)):
        f_i, z_i = elem.step(drift_next=u_vals[i], dt=dt, velocity=v_vals[i])
        forces.append(f_i)

    forces = np.array(forces)

    f_yield = k0 * u_y
    assert np.max(forces) > f_yield
    assert elem.hysteretic_energy > 0.0
    k_t = elem.tangent_stiffness(drift=0.02, velocity=0.1)
    assert k_t > 0.0


def test_nonlinear_solver_elastic_match_with_linear_solver():
    """Verify that in the elastic regime with sharp transition (n=10), nonlinear solver closely matches linear solver."""
    bldg = ShearBuilding.from_uniform(
        num_storeys=3,
        storey_mass=100000.0,
        storey_stiffness=1.5e8,
        storey_height=3.5,
    )
    damp = RayleighDamping.from_uniform_ratio(bldg, 0.05)

    linear_solver = NewmarkSolver.average_acceleration(bldg, damp)
    nonlinear_solver = NonlinearInelasticSolver(
        building=bldg,
        damping=damp,
        yield_drift_ratio=0.05,  # 5% yield drift (very high, stays completely elastic)
        post_yield_ratio=0.05,
        n_exp=10.0,  # Sharp transition to linear elasticity
    )

    dt = 0.01
    t = np.arange(0.0, 3.0, dt)
    ag = 0.05 * 9.81 * np.sin(2.0 * np.pi * 2.0 * t) * np.exp(-t)

    resp_lin = linear_solver.solve(ground_acceleration=ag, dt=dt)
    resp_nonlin = nonlinear_solver.solve(ag, dt)

    # With smooth hysteretic formulation, peak drift ratio matches linear solver within 15%
    assert abs(resp_lin.max_drift_ratio - resp_nonlin.max_pidr) / resp_lin.max_drift_ratio < 0.15
    assert abs(resp_lin.peak_base_shear - resp_nonlin.peak_base_shear) / resp_lin.peak_base_shear < 0.15
    assert resp_nonlin.residual_drift_ratio < 1e-4


def test_nonlinear_inelastic_yielding_and_energy_dissipation():
    """Test intense excitation causing significant plastic yielding and hysteretic energy dissipation."""
    bldg = ShearBuilding.from_uniform(
        num_storeys=3,
        storey_mass=100000.0,
        storey_stiffness=1.5e8,
        storey_height=3.5,
    )
    damp = RayleighDamping.from_uniform_ratio(bldg, 0.05)

    solver = NonlinearInelasticSolver(
        building=bldg,
        damping=damp,
        yield_drift_ratio=0.002,  # 0.2% yield drift (low threshold for strong yielding)
        post_yield_ratio=0.05,
        delta_nu=0.02,
        delta_eta=0.02,
    )

    dt = 0.01
    t = np.arange(0.0, 4.0, dt)
    # Severe near-fault pulse
    ag_pulse = 0.50 * 9.81 * np.sin(2.0 * np.pi * 1.0 * t) * np.exp(-0.8 * t)

    resp = solver.solve(ag_pulse, dt)

    assert resp.max_pidr > 0.002  # Yielding occurred
    assert max(resp.storey_ductilities) > 1.0  # Ductility exceeds 1
    assert resp.total_energy_dissipated_joules > 1e4  # Substantial hysteretic energy dissipation
    assert resp.peak_base_shear > 0.0

    summary = resp.to_dict()
    assert "max_pidr_pct" in summary
    assert "total_hysteretic_energy_j" in summary
