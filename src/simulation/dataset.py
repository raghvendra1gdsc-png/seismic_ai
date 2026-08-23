"""Dataset generation, splitting, and serialization pipeline.

Automates the simulation sweep over parametric buildings and earthquake records,
extracts tabular data, and partitions datasets into 4-tier generalization splits.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import json
import numpy as np

from src.structural.building import ShearBuilding
from src.earthquake.record import GroundMotionRecord
from src.earthquake.spectra import ResponseSpectrum
from src.simulation.runner import SimulationRunner

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


class SimulationDataset:
    """Simulation dataset builder and manager for physics-to-ML surrogate workflows."""

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self._raw_data = data
        self.num_samples: int = len(data)

        if self.num_samples > 0:
            self.columns: List[str] = list(data[0].keys())
        else:
            self.columns: List[str] = []

    @classmethod
    def generate(
        cls,
        buildings_and_damping: List[Tuple[ShearBuilding, float]],
        records: List[GroundMotionRecord],
        progress_callback: Optional[callable] = None,
    ) -> "SimulationDataset":
        """Generate simulation dataset across all building-earthquake pairs.

        Parameters
        ----------
        buildings_and_damping : List[Tuple[ShearBuilding, float]]
            List of (building, damping_ratio) tuples.
        records : List[GroundMotionRecord]
            List of earthquake ground motion records.
        progress_callback : Optional[callable]
            Optional callback f(completed, total) for progress reporting.

        Returns
        -------
        SimulationDataset
            Constructed simulation dataset.
        """
        data = []
        total_runs = len(buildings_and_damping) * len(records)
        completed = 0

        # Pre-compute response spectra for each record at standard damping ratios to optimize runtime
        # Typical dampings: 0.02, 0.03, 0.05
        cached_spectra: Dict[Tuple[str, float], ResponseSpectrum] = {}

        for bldg, zeta in buildings_and_damping:
            # Pre-compute modal analysis once per building
            for rec in records:
                spec_key = (rec.name, round(zeta, 4))
                if spec_key not in cached_spectra:
                    cached_spectra[spec_key] = ResponseSpectrum(record=rec, damping_ratio=zeta)
                spectrum = cached_spectra[spec_key]

                sample = SimulationRunner.run_single(
                    building=bldg,
                    record=rec,
                    damping_ratio=zeta,
                    spectrum=spectrum,
                )
                data.append(sample)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_runs)

        return cls(data)

    def to_dataframe(self) -> Any:
        """Convert dataset to a pandas DataFrame."""
        if _HAS_PANDAS:
            return pd.DataFrame(self._raw_data)
        return self._raw_data

    def get_feature_columns(self) -> List[str]:
        """List of input feature column names (excluding IDs and targets)."""
        return [
            c for c in self.columns
            if c not in ["building_id", "earthquake_id"] and not c.startswith("target_")
        ]

    def get_target_columns(self) -> List[str]:
        """List of target column names."""
        return [c for c in self.columns if c.startswith("target_")]

    def split_random(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset randomly into train, validation, and test sets (Tier 1)."""
        rng = np.random.RandomState(seed)
        indices = np.arange(self.num_samples)
        rng.shuffle(indices)

        n_test = int(np.round(test_size * self.num_samples))
        n_val = int(np.round(val_size * self.num_samples))
        n_train = self.num_samples - n_test - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_data = [self._raw_data[i] for i in train_idx]
        val_data = [self._raw_data[i] for i in val_idx]
        test_data = [self._raw_data[i] for i in test_idx]

        return train_data, val_data, test_data

    def split_by_earthquake(
        self,
        test_earthquake_names: List[str],
        val_earthquake_names: Optional[List[str]] = None,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset by earthquake identity for Unseen Earthquake Generalization (Tier 2)."""
        test_set = set(test_earthquake_names)
        val_set = set(val_earthquake_names or [])

        train_data = []
        val_data = []
        test_data = []

        for row in self._raw_data:
            eq = row["earthquake_id"]
            if eq in test_set:
                test_data.append(row)
            elif eq in val_set:
                val_data.append(row)
            else:
                train_data.append(row)

        return train_data, val_data, test_data

    def split_by_building(
        self,
        test_building_ids: List[str],
        val_building_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset by building identity for Unseen Building Generalization (Tier 3)."""
        test_set = set(test_building_ids)
        val_set = set(val_building_ids or [])

        train_data = []
        val_data = []
        test_data = []

        for row in self._raw_data:
            bldg = row["building_id"]
            if bldg in test_set:
                test_data.append(row)
            elif bldg in val_set:
                val_data.append(row)
            else:
                train_data.append(row)

        return train_data, val_data, test_data

    def split_dual_unseen(
        self,
        test_earthquake_names: List[str],
        test_building_ids: List[str],
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset for Dual Unseen (Blind) Generalization (Tier 4).

        Train set: Seen buildings AND Seen earthquakes.
        Test set: Unseen buildings AND Unseen earthquakes simultaneously.
        """
        test_eqs = set(test_earthquake_names)
        test_bldgs = set(test_building_ids)

        train_data = []
        test_data = []

        for row in self._raw_data:
            is_test_eq = row["earthquake_id"] in test_eqs
            is_test_bldg = row["building_id"] in test_bldgs

            if not is_test_eq and not is_test_bldg:
                train_data.append(row)
            elif is_test_eq and is_test_bldg:
                test_data.append(row)

        return train_data, test_data

    def save_csv(self, output_path: str) -> None:
        """Save dataset rows to a CSV file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if _HAS_PANDAS:
            df = pd.DataFrame(self._raw_data)
            df.to_csv(output_path, index=False)
        else:
            # Fallback manual CSV export
            if not self._raw_data:
                return
            header = ",".join(self.columns)
            lines = [header]
            for row in self._raw_data:
                vals = [str(row.get(col, "")) for col in self.columns]
                lines.append(",".join(vals))
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    @classmethod
    def load_csv(cls, file_path: str) -> "SimulationDataset":
        """Load dataset from a CSV file."""
        if _HAS_PANDAS:
            df = pd.read_csv(file_path)
            data = df.to_dict(orient="records")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            header = lines[0].split(",")
            data = []
            for l in lines[1:]:
                parts = l.split(",")
                row = {}
                for col, val in zip(header, parts):
                    try:
                        row[col] = float(val)
                    except ValueError:
                        row[col] = val
                data.append(row)
        return cls(data)
