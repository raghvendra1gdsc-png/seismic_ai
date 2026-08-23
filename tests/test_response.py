"""Unit tests for DynamicResponse container and derived engineering demand parameters."""

import unittest
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.damping import RayleighDamping
from src.dynamics.solver import NewmarkSolver
from src.dynamics.response import DynamicResponse


class TestDynamicResponse(unittest.TestCase):
    """Test response post-processing, drifts, shears, and dynamic equilibrium."""

    def setUp(self):
        self.heights = [3.5, 3.0, 3.0]
        self.masses = [1000.0, 1000.0, 800.0]
        self.stiffnesses = [200000.0, 180000.0, 150000.0]
        self.building = ShearBuilding(
            masses=self.masses,
            stiffnesses=self.stiffnesses,
            heights=self.heights,
        )
        self.damping = RayleighDamping.from_uniform_ratio(self.building, zeta=0.03)
        self.solver = NewmarkSolver.average_acceleration(self.building, self.damping)

    def test_derived_kinematic_and_kinetic_quantities(self):
        dt = 0.01
        time = np.arange(0, 4.0, dt)
        ag = 1.2 * np.sin(4.0 * time)

        response = self.solver.solve(ground_acceleration=ag, time=time)

        # 1. Verify drift definition: Delta_1 = u_1, Delta_i = u_i - u_{i-1}
        drifts = response.interstorey_drift
        np.testing.assert_allclose(drifts[0, :], response.displacement[0, :])
        np.testing.assert_allclose(drifts[1, :], response.displacement[1, :] - response.displacement[0, :])
        np.testing.assert_allclose(drifts[2, :], response.displacement[2, :] - response.displacement[1, :])

        # 2. Verify drift ratio: theta_i = Delta_i / h_i
        drift_ratios = response.interstorey_drift_ratio
        for i in range(3):
            np.testing.assert_allclose(drift_ratios[i, :], drifts[i, :] / self.heights[i])

        # 3. Verify storey shear forces: V_i = k_i * Delta_i
        storey_shears = response.storey_shear_force
        for i in range(3):
            np.testing.assert_allclose(storey_shears[i, :], self.stiffnesses[i] * drifts[i, :])

        # 4. Verify base shear: V_b = V_1
        base_shear = response.base_shear
        np.testing.assert_allclose(base_shear, storey_shears[0, :])

        # 5. Verify total acceleration: u_total_ddot = u_ddot + a_g
        total_acc = response.total_acceleration
        for i in range(3):
            np.testing.assert_allclose(total_acc[i, :], response.acceleration[i, :] + ag)

        # 6. Verify peak demand metrics
        self.assertAlmostEqual(response.max_roof_displacement, np.max(np.abs(response.displacement[2, :])))
        self.assertAlmostEqual(response.peak_base_shear, np.max(np.abs(base_shear)))
        self.assertAlmostEqual(response.max_drift_ratio, np.max(np.abs(drift_ratios)))

    def test_dynamic_equilibrium_at_all_timesteps(self):
        """Verify M * u_ddot + C * u_dot + K * u = -M * r * a_g at all time steps."""
        dt = 0.005
        time = np.arange(0, 3.0, dt)
        ag = 2.0 * np.sin(6.0 * time) * np.exp(-0.2 * time)

        response = self.solver.solve(ground_acceleration=ag, time=time)

        m_mat = self.building.mass_matrix
        k_mat = self.building.stiffness_matrix
        c_mat = self.damping.damping_matrix
        r_vec = self.building.influence_vector

        # Compute dynamic residual at each time step
        f_inertia = m_mat @ response.acceleration               # shape (3, N_t)
        f_damping = c_mat @ response.velocity                   # shape (3, N_t)
        f_elastic = k_mat @ response.displacement               # shape (3, N_t)
        f_ground = -(m_mat @ r_vec)[:, np.newaxis] * ag[np.newaxis, :]  # shape (3, N_t)

        residual = f_inertia + f_damping + f_elastic - f_ground
        max_residual = np.max(np.abs(residual))

        # Dynamic equilibrium should hold to high precision (< 1e-4 N)
        self.assertLess(max_residual, 1e-4)

    def test_summary_dictionary(self):
        dt = 0.01
        time = np.arange(0, 2.0, dt)
        ag = np.sin(5.0 * time)
        response = self.solver.solve(ground_acceleration=ag, time=time)

        summary = response.summary()
        self.assertIn("max_roof_displacement_mm", summary)
        self.assertIn("peak_base_shear_kN", summary)
        self.assertIn("max_interstorey_drift_ratio", summary)
        self.assertEqual(len(summary["peak_floor_displacements_m"]), 3)
        self.assertEqual(len(summary["peak_floor_accelerations_m_s2"]), 3)
        self.assertEqual(len(summary["peak_interstorey_drift_ratios"]), 3)


if __name__ == "__main__":
    unittest.main()
