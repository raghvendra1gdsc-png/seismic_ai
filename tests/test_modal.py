"""Unit tests for modal analysis and eigenvalue solver."""

import unittest
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.modal import ModalAnalysis


class TestModalAnalysis(unittest.TestCase):
    """Test modal properties against closed-form structural dynamics benchmarks."""

    def test_1dof_modal_properties(self):
        m = 1000.0
        k = 40000.0  # omega = sqrt(40000/1000) = sqrt(40) ~ 6.324555 rad/s
        bldg = ShearBuilding(masses=[m], stiffnesses=[k], heights=3.0)
        modal = ModalAnalysis(bldg)

        expected_omega = np.sqrt(k / m)
        expected_f = expected_omega / (2.0 * np.pi)
        expected_t = 1.0 / expected_f

        self.assertEqual(modal.num_modes, 1)
        self.assertAlmostEqual(modal.circular_frequencies[0], expected_omega, places=7)
        self.assertAlmostEqual(modal.cyclic_frequencies[0], expected_f, places=7)
        self.assertAlmostEqual(modal.periods[0], expected_t, places=7)
        self.assertAlmostEqual(modal.fundamental_period, expected_t, places=7)

        # Mass normalized phi = [1 / sqrt(m)]
        phi_mass = modal.mode_shapes("mass")
        self.assertAlmostEqual(phi_mass[0, 0], 1.0 / np.sqrt(m), places=7)

        # Participation factor Gamma = phi^T M r = (1/sqrt(m)) * m * 1 = sqrt(m)
        self.assertAlmostEqual(modal.participation_factors[0], np.sqrt(m), places=7)

        # Effective modal mass M* = Gamma^2 = m
        self.assertAlmostEqual(modal.effective_masses[0], m, places=7)
        self.assertAlmostEqual(modal.effective_mass_ratios[0], 1.0, places=7)

    def test_chopra_2dof_benchmark(self):
        """Chopra 'Dynamics of Structures' 2-DOF uniform shear frame benchmark."""
        m_val = 100.0
        k_val = 1000.0
        bldg = ShearBuilding.from_uniform(
            num_storeys=2,
            storey_mass=m_val,
            storey_stiffness=k_val,
            storey_height=3.0,
        )
        modal = ModalAnalysis(bldg)

        # Analytical circular frequencies
        # omega_1 = sqrt((3 - sqrt(5))/2) * sqrt(k/m) = 0.6180339887 * sqrt(k/m)
        # omega_2 = sqrt((3 + sqrt(5))/2) * sqrt(k/m) = 1.6180339887 * sqrt(k/m)
        base_freq = np.sqrt(k_val / m_val)
        expected_w1 = 0.618033988749895 * base_freq
        expected_w2 = 1.618033988749895 * base_freq

        self.assertAlmostEqual(modal.circular_frequencies[0], expected_w1, places=6)
        self.assertAlmostEqual(modal.circular_frequencies[1], expected_w2, places=6)

        # Roof normalized mode shapes: phi_1 = [(sqrt(5)-1)/2, 1]^T = [0.618034, 1.0]^T
        #                              phi_2 = [-(sqrt(5)+1)/2, 1]^T = [-1.618034, 1.0]^T
        phi_roof = modal.mode_shapes("roof")
        expected_phi1 = np.array([(np.sqrt(5.0) - 1.0) / 2.0, 1.0])
        expected_phi2 = np.array([-(np.sqrt(5.0) + 1.0) / 2.0, 1.0])

        np.testing.assert_allclose(phi_roof[:, 0], expected_phi1, rtol=1e-5)
        np.testing.assert_allclose(phi_roof[:, 1], expected_phi2, rtol=1e-5)

        # Effective mass sum must equal total mass
        self.assertAlmostEqual(np.sum(modal.effective_masses), bldg.total_mass, places=6)

    def test_chopra_3dof_benchmark(self):
        """Chopra 3-DOF uniform shear frame benchmark."""
        m_val = 1000.0
        k_val = 50000.0
        bldg = ShearBuilding.from_uniform(
            num_storeys=3,
            storey_mass=m_val,
            storey_stiffness=k_val,
            storey_height=3.5,
        )
        modal = ModalAnalysis(bldg)

        # Analytical circular frequencies (multipliers on sqrt(k/m)):
        # w1 = 0.445041868 * sqrt(k/m)
        # w2 = 1.246979604 * sqrt(k/m)
        # w3 = 1.801937736 * sqrt(k/m)
        base_w = np.sqrt(k_val / m_val)
        expected_w = np.array([0.445041868, 1.246979604, 1.801937736]) * base_w

        np.testing.assert_allclose(modal.circular_frequencies, expected_w, rtol=1e-5)

        # Roof normalized mode shapes
        phi_roof = modal.mode_shapes("roof")
        expected_phi1 = np.array([0.44504, 0.80194, 1.0])
        expected_phi2 = np.array([-1.24698, -0.55496, 1.0])
        expected_phi3 = np.array([1.80194, -2.24698, 1.0])

        np.testing.assert_allclose(phi_roof[:, 0], expected_phi1, atol=1e-4)
        np.testing.assert_allclose(phi_roof[:, 1], expected_phi2, atol=1e-4)
        np.testing.assert_allclose(phi_roof[:, 2], expected_phi3, atol=1e-4)

    def test_orthogonality_and_mass_completeness(self):
        """Verify mass- and stiffness-orthonormality and 100% mass participation."""
        # Non-uniform 5-storey building
        m = [2000.0, 1800.0, 1600.0, 1400.0, 1200.0]
        k = [500000.0, 450000.0, 400000.0, 350000.0, 300000.0]
        bldg = ShearBuilding(masses=m, stiffnesses=k, heights=3.2)
        modal = ModalAnalysis(bldg)

        phi = modal.mode_shapes("mass")
        m_mat = bldg.mass_matrix
        k_mat = bldg.stiffness_matrix

        # Mass orthonormality: Phi^T M Phi = I
        phi_m_phi = phi.T @ m_mat @ phi
        np.testing.assert_allclose(phi_m_phi, np.eye(5), atol=1e-10)

        # Stiffness orthonormality: Phi^T K Phi = Omega^2 = diag(omega_n^2)
        phi_k_phi = phi.T @ k_mat @ phi
        omega_sq = np.diag(modal.circular_frequencies**2)
        np.testing.assert_allclose(phi_k_phi, omega_sq, rtol=1e-8, atol=1e-8)

        # Total modal effective mass sum equals total mass
        self.assertAlmostEqual(np.sum(modal.effective_masses), bldg.total_mass, places=7)
        self.assertAlmostEqual(modal.cumulative_mass_ratios[-1], 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
