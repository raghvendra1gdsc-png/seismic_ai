"""Unit tests for structural modeling and matrix assembly."""

import unittest
import numpy as np

from src.structural.building import ShearBuilding
from src.structural.properties import compute_storey_shear_stiffness


class TestStructuralProperties(unittest.TestCase):
    """Test structural property calculation helpers."""

    def test_compute_storey_shear_stiffness_single_column(self):
        # E = 200 GPa = 2e11 Pa, I = 0.001 m^4, h = 3.0 m
        # k = 12 * 2e11 * 0.001 / 27 = 2.4e9 / 27 = 88888888.888... N/m
        e = 2e11
        i = 1e-3
        h = 3.0
        k = compute_storey_shear_stiffness(e, i, h, num_columns=1)
        expected = 12.0 * e * i / (h**3)
        self.assertAlmostEqual(k, expected, places=5)

    def test_compute_storey_shear_stiffness_multi_column(self):
        e = 2.5e10
        i = 5e-4
        h = 3.5
        num_cols = 4
        k = compute_storey_shear_stiffness(e, i, h, num_columns=num_cols)
        expected = num_cols * (12.0 * e * i / (h**3))
        self.assertAlmostEqual(k, expected, places=5)

    def test_invalid_property_inputs(self):
        with self.assertRaises(ValueError):
            compute_storey_shear_stiffness(-1.0, 1e-3, 3.0)
        with self.assertRaises(ValueError):
            compute_storey_shear_stiffness(2e11, -1e-3, 3.0)
        with self.assertRaises(ValueError):
            compute_storey_shear_stiffness(2e11, 1e-3, 0.0)
        with self.assertRaises(ValueError):
            compute_storey_shear_stiffness(2e11, 1e-3, 3.0, num_columns=0)


class TestShearBuilding(unittest.TestCase):
    """Test ShearBuilding class and matrix assembly."""

    def test_1dof_building(self):
        building = ShearBuilding(masses=[1000.0], stiffnesses=[50000.0], heights=3.0)
        self.assertEqual(building.num_storeys, 1)
        self.assertEqual(building.total_mass, 1000.0)
        self.assertEqual(building.total_height, 3.0)

        # M = [1000], K = [50000]
        np.testing.assert_array_equal(building.mass_matrix, np.array([[1000.0]]))
        np.testing.assert_array_equal(building.stiffness_matrix, np.array([[50000.0]]))
        np.testing.assert_array_equal(building.influence_vector, np.array([1.0]))

    def test_3dof_building_matrices(self):
        m = [100.0, 100.0, 80.0]
        k = [3000.0, 2000.0, 1000.0]
        h = [3.5, 3.0, 3.0]
        building = ShearBuilding(masses=m, stiffnesses=k, heights=h, name="Test3Storey")

        self.assertEqual(building.num_storeys, 3)
        self.assertEqual(building.total_mass, 280.0)
        self.assertEqual(building.total_height, 9.5)
        np.testing.assert_array_equal(building.storey_elevations, np.array([3.5, 6.5, 9.5]))

        # Mass matrix: diagonal [100, 100, 80]
        expected_m = np.diag(m)
        np.testing.assert_allclose(building.mass_matrix, expected_m)

        # Stiffness matrix:
        # K = [[k1 + k2, -k2, 0],
        #      [-k2, k2 + k3, -k3],
        #      [0, -k3, k3]]
        # K = [[5000, -2000, 0],
        #      [-2000, 3000, -1000],
        #      [0, -1000, 1000]]
        expected_k = np.array([
            [5000.0, -2000.0, 0.0],
            [-2000.0, 3000.0, -1000.0],
            [0.0, -1000.0, 1000.0],
        ])
        np.testing.assert_allclose(building.stiffness_matrix, expected_k)

        # Symmetry and positive-definiteness
        k_mat = building.stiffness_matrix
        np.testing.assert_allclose(k_mat, k_mat.T)
        eigs = np.linalg.eigvalsh(k_mat)
        self.assertTrue(np.all(eigs > 0))

    def test_uniform_factory_method(self):
        building = ShearBuilding.from_uniform(
            num_storeys=5,
            storey_mass=500.0,
            storey_stiffness=10000.0,
            storey_height=3.2,
        )
        self.assertEqual(building.num_storeys, 5)
        self.assertEqual(building.total_mass, 2500.0)
        self.assertEqual(building.total_height, 16.0)
        self.assertEqual(building.mass_matrix.shape, (5, 5))
        self.assertEqual(building.stiffness_matrix.shape, (5, 5))

    def test_input_validation(self):
        # Empty arrays
        with self.assertRaises(ValueError):
            ShearBuilding(masses=[], stiffnesses=[100.0], heights=3.0)

        # Dimension mismatch
        with self.assertRaises(ValueError):
            ShearBuilding(masses=[100.0, 100.0], stiffnesses=[100.0], heights=3.0)

        # Negative / zero mass
        with self.assertRaises(ValueError):
            ShearBuilding(masses=[100.0, -50.0], stiffnesses=[100.0, 100.0], heights=3.0)
        with self.assertRaises(ValueError):
            ShearBuilding(masses=[100.0, 0.0], stiffnesses=[100.0, 100.0], heights=3.0)

        # Negative stiffness
        with self.assertRaises(ValueError):
            ShearBuilding(masses=[100.0, 100.0], stiffnesses=[-100.0, 100.0], heights=3.0)

        # Negative height
        with self.assertRaises(ValueError):
            ShearBuilding(masses=[100.0, 100.0], stiffnesses=[100.0, 100.0], heights=-3.0)


if __name__ == "__main__":
    unittest.main()
