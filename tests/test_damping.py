"""Unit tests for Rayleigh proportional damping formulation."""

import unittest
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.modal import ModalAnalysis
from src.dynamics.damping import RayleighDamping


class TestRayleighDamping(unittest.TestCase):
    """Test Rayleigh proportional damping matrix assembly and coefficient calculation."""

    def setUp(self):
        # 3-DOF uniform building
        self.building = ShearBuilding.from_uniform(
            num_storeys=3,
            storey_mass=1000.0,
            storey_stiffness=200000.0,
            storey_height=3.0,
        )
        self.modal = ModalAnalysis(self.building)

    def test_rayleigh_from_frequencies_exact_target_damping(self):
        zeta_1 = 0.05
        zeta_2 = 0.05
        w1 = self.modal.circular_frequencies[0]
        w2 = self.modal.circular_frequencies[1]

        damping = RayleighDamping.from_frequencies(
            building=self.building,
            zeta_i=zeta_1,
            zeta_j=zeta_2,
            omega_i=w1,
            omega_j=w2,
        )

        # Check damping at w1 and w2
        self.assertAlmostEqual(damping.damping_ratio_at_frequency(w1), zeta_1, places=7)
        self.assertAlmostEqual(damping.damping_ratio_at_frequency(w2), zeta_2, places=7)

        # Check damping matrix properties: C = alpha * M + beta * K
        c_mat = damping.damping_matrix
        expected_c = damping.alpha * self.building.mass_matrix + damping.beta * self.building.stiffness_matrix
        np.testing.assert_allclose(c_mat, expected_c)
        np.testing.assert_allclose(c_mat, c_mat.T)

        # Positive semi-definiteness
        eigs = np.linalg.eigvalsh(c_mat)
        self.assertTrue(np.all(eigs >= 0.0))

    def test_rayleigh_from_modes(self):
        zeta = 0.03
        damping = RayleighDamping.from_modes(
            building=self.building,
            zeta_i=zeta,
            zeta_j=zeta,
            mode_i=1,
            mode_j=3,
        )
        modal_zetas = damping.modal_damping_ratios(self.modal)
        self.assertAlmostEqual(modal_zetas[0], zeta, places=7)
        self.assertAlmostEqual(modal_zetas[2], zeta, places=7)
        # Mode 2 damping ratio should be slightly lower than 0.03 due to concave Rayleigh curve
        self.assertLess(modal_zetas[1], zeta)
        self.assertGreater(modal_zetas[1], 0.0)

    def test_rayleigh_1dof_edge_case(self):
        bldg_1dof = ShearBuilding(masses=[1000.0], stiffnesses=[50000.0], heights=3.0)
        damping = RayleighDamping.from_uniform_ratio(bldg_1dof, zeta=0.05)
        modal = ModalAnalysis(bldg_1dof)
        w = modal.circular_frequencies[0]

        self.assertAlmostEqual(damping.damping_ratio_at_frequency(w), 0.05, places=7)
        self.assertEqual(damping.beta, 0.0)
        self.assertAlmostEqual(damping.alpha, 2.0 * 0.05 * w, places=7)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            RayleighDamping(self.building, alpha=-0.1, beta=0.01)
        with self.assertRaises(ValueError):
            RayleighDamping(self.building, alpha=0.1, beta=-0.01)
        with self.assertRaises(ValueError):
            RayleighDamping.from_frequencies(self.building, 0.05, 0.05, 10.0, 10.0)
        with self.assertRaises(ValueError):
            RayleighDamping.from_modes(self.building, 0.05, 0.05, mode_i=1, mode_j=1)
        with self.assertRaises(ValueError):
            RayleighDamping.from_modes(self.building, 0.05, 0.05, mode_i=1, mode_j=5)


if __name__ == "__main__":
    unittest.main()
