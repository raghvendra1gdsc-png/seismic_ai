"""Unit tests for Newmark-beta time-history integration solver."""

import unittest
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.modal import ModalAnalysis
from src.dynamics.damping import RayleighDamping
from src.dynamics.solver import NewmarkSolver


class TestNewmarkSolver(unittest.TestCase):
    """Test Newmark-beta solver against analytical closed-form dynamics solutions."""

    def test_sdof_damped_free_vibration_exact(self):
        """Verify damped free vibration against exact closed-form solution."""
        m = 1000.0
        k = 40000.0
        zeta = 0.05
        u0 = 0.05  # 50 mm initial displacement
        v0 = 0.10  # 0.1 m/s initial velocity

        bldg = ShearBuilding(masses=[m], stiffnesses=[k], heights=3.0)
        damping = RayleighDamping.from_uniform_ratio(bldg, zeta=zeta)
        solver = NewmarkSolver.average_acceleration(bldg, damping)

        wn = np.sqrt(k / m)
        wd = wn * np.sqrt(1.0 - zeta**2)
        duration = 5.0
        dt = 0.002
        time = np.arange(0, duration, dt)

        # Numerical solution
        response = solver.solve(
            ground_acceleration=np.zeros_like(time),
            time=time,
            initial_displacement=[u0],
            initial_velocity=[v0],
        )

        # Exact closed-form solution:
        # u(t) = exp(-zeta*wn*t) * [ u0*cos(wd*t) + (v0 + zeta*wn*u0)/wd * sin(wd*t) ]
        a_coef = u0
        b_coef = (v0 + zeta * wn * u0) / wd
        u_exact = np.exp(-zeta * wn * time) * (a_coef * np.cos(wd * time) + b_coef * np.sin(wd * time))

        u_num = response.displacement[0, :]

        # Check relative L2 and Max error
        l2_error = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)
        max_error = np.max(np.abs(u_num - u_exact))

        self.assertLess(l2_error, 0.001)  # Less than 0.1% L2 error
        self.assertLess(max_error, 5e-4)   # Sub-millimeter peak error

    def test_sdof_harmonic_resonance_amplitude(self):
        """Verify resonant steady-state displacement matches dynamic amplification theory."""
        m = 500.0
        k = 20000.0  # wn = sqrt(40) = 6.324555 rad/s
        wn = np.sqrt(k / m)
        zeta = 0.05

        bldg = ShearBuilding(masses=[m], stiffnesses=[k], heights=3.0)
        damping = RayleighDamping.from_uniform_ratio(bldg, zeta=zeta)
        solver = NewmarkSolver.average_acceleration(bldg, damping)

        # Resonant harmonic base motion: a_g(t) = a0 * sin(wn * t)
        a0 = 1.0  # 1 m/s^2
        dt = 0.005
        # Run for 80 seconds (~80 cycles) to reach steady state
        time = np.arange(0, 80.0, dt)
        ag = a0 * np.sin(wn * time)

        response = solver.solve(ground_acceleration=ag, time=time)

        # Theoretical steady-state peak displacement:
        # Dynamic Amplification Factor at resonance: Rd = 1 / (2 * zeta) = 10
        # Static equivalent displacement: u_st = m * a0 / k = a0 / wn^2
        # Peak amplitude: u_peak = Rd * u_st = a0 / (2 * zeta * wn^2)
        u_peak_theory = a0 / (2.0 * zeta * wn**2)

        # Evaluate peak amplitude in the last 15 seconds (steady state)
        last_indices = np.where(time >= 65.0)[0]
        u_steady_peak_numerical = np.max(np.abs(response.displacement[0, last_indices]))

        relative_error = abs(u_steady_peak_numerical - u_peak_theory) / u_peak_theory
        self.assertLess(relative_error, 0.005)  # Within 0.5% of theoretical amplitude

    def test_mdof_modal_superposition_match(self):
        """Verify 3-DOF Newmark solver response matches exact modal superposition."""
        m = [1000.0, 1000.0, 800.0]
        k = [200000.0, 180000.0, 150000.0]
        bldg = ShearBuilding(masses=m, stiffnesses=k, heights=3.0)

        zeta = 0.03
        damping = RayleighDamping.from_uniform_ratio(bldg, zeta=zeta, mode_i=1, mode_j=3)
        modal = ModalAnalysis(bldg)

        solver = NewmarkSolver.average_acceleration(bldg, damping)

        # Multi-frequency base excitation
        dt = 0.005
        time = np.arange(0, 10.0, dt)
        ag = 1.5 * np.sin(2.0 * np.pi * 1.5 * time) + 0.8 * np.sin(2.0 * np.pi * 3.5 * time)

        # Direct Newmark integration
        resp_newmark = solver.solve(ground_acceleration=ag, time=time)

        # Modal superposition solution:
        # Solve each SDOF mode: q_ddot_n + 2*zeta_n*w_n*q_dot_n + w_n^2*q_n = -Gamma_n * a_g(t)
        phi_mass = modal.mode_shapes("mass")  # (N, N)
        gammas = modal.participation_factors
        modal_zetas = damping.modal_damping_ratios(modal)
        omegas = modal.circular_frequencies

        u_modal = np.zeros_like(resp_newmark.displacement)

        for n in range(bldg.num_storeys):
            wn = omegas[n]
            zn = modal_zetas[n]
            gn = gammas[n]

            # 1-DOF SDOF solver for mode n
            mode_bldg = ShearBuilding(masses=[1.0], stiffnesses=[wn**2], heights=1.0)
            mode_damp = RayleighDamping.from_coefficients(
                building=mode_bldg, alpha=2.0 * zn * wn, beta=0.0
            )
            mode_solver = NewmarkSolver.average_acceleration(mode_bldg, mode_damp)

            # Modal load: P_n(t) = -Gamma_n * a_g(t)
            p_n = -gn * ag[np.newaxis, :]
            mode_resp = mode_solver.solve(external_force=p_n, time=time)
            q_n = mode_resp.displacement[0, :]

            # Add modal contribution: u(t) += phi_n * q_n(t)
            u_modal += phi_mass[:, n:n+1] @ q_n[np.newaxis, :]

        # Compare Newmark direct integration vs modal superposition
        for floor_idx in range(bldg.num_storeys):
            u_nmk = resp_newmark.displacement[floor_idx, :]
            u_mod = u_modal[floor_idx, :]
            l2_err = np.linalg.norm(u_nmk - u_mod) / np.linalg.norm(u_mod)
            self.assertLess(l2_err, 0.005)  # Less than 0.5% difference

    def test_linear_acceleration_solver(self):
        """Verify Linear Acceleration solver produces matching results for fine dt."""
        bldg = ShearBuilding.from_uniform(3, 1000.0, 100000.0, 3.0)
        damping = RayleighDamping.from_uniform_ratio(bldg, zeta=0.05)

        solver_avg = NewmarkSolver.average_acceleration(bldg, damping)
        solver_lin = NewmarkSolver.linear_acceleration(bldg, damping)

        dt = 0.001
        time = np.arange(0, 5.0, dt)
        ag = 2.0 * np.sin(5.0 * time)

        resp_avg = solver_avg.solve(ground_acceleration=ag, time=time)
        resp_lin = solver_lin.solve(ground_acceleration=ag, time=time)

        rel_diff = np.linalg.norm(resp_avg.displacement - resp_lin.displacement) / np.linalg.norm(resp_avg.displacement)
        self.assertLess(rel_diff, 1e-3)


if __name__ == "__main__":
    unittest.main()
