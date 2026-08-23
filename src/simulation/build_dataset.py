"""CLI script and utility to build and serialize the benchmark simulation dataset."""

import os
import json
import numpy as np

from src.structural.building import ShearBuilding
from src.earthquake.database import GroundMotionDatabase, generate_stochastic_ground_motion
from src.simulation.generator import BuildingGenerator
from src.simulation.dataset import SimulationDataset


def build_full_dataset(
    output_dir: str = "data/datasets/simulation_dataset",
    num_buildings: int = 40,
    seed: int = 42,
) -> SimulationDataset:
    """Build, split, and save the complete structural dynamics simulation dataset."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Generating {num_buildings} parametric multi-storey building models...")
    generator = BuildingGenerator(seed=seed)
    buildings = generator.generate_suite(num_buildings=num_buildings, min_storeys=3, max_storeys=10)

    print("[*] Compiling ground motion suite (historical + scaled records)...")
    db = GroundMotionDatabase()
    base_records = db.get_all_records()

    # Create scaled variants for rich ground motion space
    all_records = []
    for rec in base_records:
        all_records.append(rec)
        # Moderate intensity scaling
        all_records.append(rec.scale_to_pga(target_pga_g=max(0.15, rec.pga_g * 0.6)))
        all_records.append(rec.scale_to_pga(target_pga_g=min(1.20, rec.pga_g * 1.4)))

    total_sims = len(buildings) * len(all_records)
    print(f"[*] Executing {total_sims} non-linear/linear time-history simulations ({len(buildings)} bldgs x {len(all_records)} earthquakes)...")

    def progress(done, total):
        if done % 100 == 0 or done == total:
            print(f"    Progress: {done}/{total} ({done/total*100:.1f}%)")

    dataset = SimulationDataset.generate(
        buildings_and_damping=buildings,
        records=all_records,
        progress_callback=progress,
    )

    full_path = os.path.join(output_dir, "simulation_dataset_full.csv")
    dataset.save_csv(full_path)
    print(f"[+] Saved full dataset ({dataset.num_samples} samples) to: {full_path}")

    # 4-Tier Splits:
    # Tier 1: Random 80/20 split
    train_r, val_r, test_r = dataset.split_random(test_size=0.20, val_size=0.10, seed=seed)
    SimulationDataset(train_r).save_csv(os.path.join(output_dir, "train_tier1_random.csv"))
    SimulationDataset(val_r).save_csv(os.path.join(output_dir, "val_tier1_random.csv"))
    SimulationDataset(test_r).save_csv(os.path.join(output_dir, "test_tier1_random.csv"))

    # Tier 2: Unseen Earthquakes (Hold out Kobe and Chi-Chi suite)
    test_eq_names = [r.name for r in all_records if "Kobe" in r.name or "ChiChi" in r.name]
    train_eq, val_eq, test_eq = dataset.split_by_earthquake(test_earthquake_names=test_eq_names)
    SimulationDataset(train_eq).save_csv(os.path.join(output_dir, "train_tier2_unseen_eq.csv"))
    SimulationDataset(test_eq).save_csv(os.path.join(output_dir, "test_tier2_unseen_eq.csv"))

    # Tier 3: Unseen Buildings (Hold out buildings 31 to 40)
    test_bldg_ids = [b[0].name for b in buildings[-8:]]
    train_bldg, val_bldg, test_bldg = dataset.split_by_building(test_building_ids=test_bldg_ids)
    SimulationDataset(train_bldg).save_csv(os.path.join(output_dir, "train_tier3_unseen_bldg.csv"))
    SimulationDataset(test_bldg).save_csv(os.path.join(output_dir, "test_tier3_unseen_bldg.csv"))

    # Tier 4: Dual Unseen (Blind)
    train_dual, test_dual = dataset.split_dual_unseen(
        test_earthquake_names=test_eq_names,
        test_building_ids=test_bldg_ids,
    )
    SimulationDataset(train_dual).save_csv(os.path.join(output_dir, "train_tier4_dual_unseen.csv"))
    SimulationDataset(test_dual).save_csv(os.path.join(output_dir, "test_tier4_dual_unseen.csv"))

    # Metadata
    metadata = {
        "num_samples": dataset.num_samples,
        "num_buildings": len(buildings),
        "num_earthquakes": len(all_records),
        "features": dataset.get_feature_columns(),
        "targets": dataset.get_target_columns(),
        "tier1_random_counts": {"train": len(train_r), "val": len(val_r), "test": len(test_r)},
        "tier2_unseen_eq_counts": {"train": len(train_eq), "test": len(test_eq)},
        "tier3_unseen_bldg_counts": {"train": len(train_bldg), "test": len(test_bldg)},
        "tier4_dual_unseen_counts": {"train": len(train_dual), "test": len(test_dual)},
    }
    with open(os.path.join(output_dir, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Dataset generation complete. Splits and metadata saved in: {output_dir}")
    return dataset


if __name__ == "__main__":
    build_full_dataset()
