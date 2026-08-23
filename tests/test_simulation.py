"""Unit tests for simulation dataset generator and splitting pipeline."""

import unittest
import os
import tempfile
import numpy as np

from src.structural.building import ShearBuilding
from src.earthquake.database import GroundMotionDatabase
from src.simulation.generator import BuildingGenerator
from src.simulation.runner import SimulationRunner
from src.simulation.dataset import SimulationDataset


class TestSimulationPipeline(unittest.TestCase):
    """Test simulation data generation, feature extraction, and 4-tier splitting."""

    def setUp(self):
        self.generator = BuildingGenerator(seed=123)
        self.db = GroundMotionDatabase()

    def test_building_generator(self):
        bldg_suite = self.generator.generate_suite(num_buildings=5, min_storeys=3, max_storeys=6)
        self.assertEqual(len(bldg_suite), 5)
        for bldg, zeta in bldg_suite:
            self.assertIsInstance(bldg, ShearBuilding)
            self.assertGreaterEqual(bldg.num_storeys, 3)
            self.assertLessEqual(bldg.num_storeys, 6)
            self.assertGreater(bldg.total_mass, 0.0)
            self.assertGreater(bldg.total_height, 0.0)
            self.assertTrue(0.01 <= zeta <= 0.08)

    def test_simulation_runner_single(self):
        bldg = ShearBuilding.from_uniform(num_storeys=3, storey_mass=1e5, storey_stiffness=1e8, storey_height=3.5)
        rec = self.db.get_record("El_Centro_1940_NS")

        sample = SimulationRunner.run_single(building=bldg, record=rec, damping_ratio=0.05)

        self.assertEqual(sample["building_id"], bldg.name)
        self.assertEqual(sample["earthquake_id"], rec.name)
        self.assertIn("fundamental_period_T1_s", sample)
        self.assertIn("Sa_T1_g", sample)
        self.assertIn("target_max_pidr", sample)
        self.assertIn("target_peak_base_shear_N", sample)
        self.assertGreater(sample["target_max_pidr"], 0.0)
        self.assertGreater(sample["target_peak_base_shear_N"], 0.0)

    def test_dataset_generation_and_4tier_splits(self):
        # Small test grid: 3 buildings x 2 earthquakes = 6 simulations
        buildings = self.generator.generate_suite(num_buildings=3, min_storeys=3, max_storeys=4)
        records = [self.db["El_Centro_1940_NS"], self.db["Kobe_1995_NS"]]

        dataset = SimulationDataset.generate(buildings_and_damping=buildings, records=records)
        self.assertEqual(dataset.num_samples, 6)
        self.assertGreaterEqual(len(dataset.get_feature_columns()), 15)
        self.assertGreaterEqual(len(dataset.get_target_columns()), 4)

        # 1. Random split
        train_r, val_r, test_r = dataset.split_random(test_size=0.33, val_size=0.0, seed=42)
        self.assertEqual(len(train_r) + len(test_r), 6)

        # 2. Split by earthquake
        train_eq, val_eq, test_eq = dataset.split_by_earthquake(test_earthquake_names=["Kobe_1995_NS"])
        self.assertEqual(len(test_eq), 3)  # All 3 buildings with Kobe
        self.assertEqual(len(train_eq), 3) # All 3 buildings with El Centro
        for r in test_eq:
            self.assertEqual(r["earthquake_id"], "Kobe_1995_NS")

        # 3. Split by building
        test_bldg_name = buildings[0][0].name
        train_b, val_b, test_b = dataset.split_by_building(test_building_ids=[test_bldg_name])
        self.assertEqual(len(test_b), 2)  # Building 0 with both earthquakes
        self.assertEqual(len(train_b), 4)

        # 4. Dual unseen split
        train_dual, test_dual = dataset.split_dual_unseen(
            test_earthquake_names=["Kobe_1995_NS"],
            test_building_ids=[test_bldg_name],
        )
        self.assertEqual(len(test_dual), 1)  # (Building 0, Kobe)
        self.assertEqual(len(train_dual), 2) # (Building 1 & 2, El Centro)

        # 5. CSV save and load
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            dataset.save_csv(tmp_path)
            loaded = SimulationDataset.load_csv(tmp_path)
            self.assertEqual(loaded.num_samples, dataset.num_samples)
            self.assertEqual(len(loaded.columns), len(dataset.columns))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
