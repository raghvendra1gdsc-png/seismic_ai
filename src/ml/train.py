"""Surrogate model training and cross-validation pipeline."""

from typing import Dict, Any, List, Tuple, Optional
import os
import json
import numpy as np

from src.simulation.dataset import SimulationDataset
from src.ml.models import (
    BaseSurrogate,
    LinearRidgeSurrogate,
    RandomForestSurrogate,
    GradientBoostingSurrogate,
    NeuralSurrogate,
)
from src.ml.evaluate import compute_regression_metrics, benchmark_speedup


class SurrogateSuite:
    """Manages training, comparison, and persistence of the 4 surrogate models."""

    def __init__(self, target_name: str = "target_max_pidr") -> None:
        self.target_name = target_name
        self.models: Dict[str, BaseSurrogate] = {
            "LinearRidge": LinearRidgeSurrogate(alpha=2.0),
            "RandomForest": RandomForestSurrogate(n_estimators=45, max_depth=7, seed=42),
            "GradientBoosting": GradientBoostingSurrogate(n_estimators=50, learning_rate=0.12, max_depth=4, seed=42),
            "NeuralMLP": NeuralSurrogate(hidden_layers=(64, 32, 16), learning_rate=0.006, max_epochs=120, seed=42),
        }
        self.feature_columns: List[str] = []

    def _extract_arrays(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        if not data:
            raise ValueError("Dataset is empty.")
        if not self.feature_columns:
            sample = data[0]
            self.feature_columns = [
                k for k in sample.keys()
                if k not in ["building_id", "earthquake_id"] and not k.startswith("target_")
            ]

        x_list = []
        y_list = []
        for row in data:
            feat_vals = [float(row[col]) for col in self.feature_columns]
            x_list.append(feat_vals)
            y_list.append(float(row[self.target_name]))

        return np.array(x_list, dtype=np.float64), np.array(y_list, dtype=np.float64)

    def train_all(
        self,
        train_data: List[Dict],
        val_data: Optional[List[Dict]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Train all 4 surrogate models on training data and evaluate."""
        x_train, y_train = self._extract_arrays(train_data)
        if val_data:
            x_val, y_val = self._extract_arrays(val_data)
        else:
            x_val, y_val = x_train, y_train

        results = {}
        for name, model in self.models.items():
            print(f"[*] Training surrogate: {name} (target: {self.target_name})...")
            model.fit(x_train, y_train)

            # Evaluate on validation set
            y_pred = model.predict(x_val)
            metrics = compute_regression_metrics(y_val, y_pred)
            speed_metrics = benchmark_speedup(model, x_val)
            metrics.update({
                "speedup_factor": speed_metrics["speedup_factor"],
                "inference_time_ms": speed_metrics["surrogate_per_sample_ms"],
            })
            results[name] = metrics
            print(f"    -> {name} R^2 = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.5f}, Speedup = {metrics['speedup_factor']:.0f}x")

        return results

    def evaluate_test(self, test_data: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Evaluate all models on an unseen test set."""
        x_test, y_test = self._extract_arrays(test_data)
        test_results = {}
        for name, model in self.models.items():
            y_pred = model.predict(x_test)
            metrics = compute_regression_metrics(y_test, y_pred)
            test_results[name] = metrics
        return test_results

    def save_all(self, output_dir: str = "models/trained") -> None:
        """Save all trained models and feature metadata."""
        os.makedirs(output_dir, exist_ok=True)
        for name, model in self.models.items():
            path = os.path.join(output_dir, f"{name}_{self.target_name}.pkl")
            model.save(path)

        meta = {
            "target_name": self.target_name,
            "feature_columns": self.feature_columns,
            "models": list(self.models.keys()),
        }
        with open(os.path.join(output_dir, f"metadata_{self.target_name}.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[+] Saved trained surrogate suite to: {output_dir}")


def train_surrogate_suite(
    dataset_dir: str = "data/datasets/simulation_dataset",
    output_model_dir: str = "models/trained",
    target_name: str = "target_max_pidr",
) -> Dict[str, Any]:
    """Train and evaluate the surrogate models on the Tier 1 Random dataset."""
    train_path = os.path.join(dataset_dir, "train_tier1_random.csv")
    val_path = os.path.join(dataset_dir, "val_tier1_random.csv")
    test_path = os.path.join(dataset_dir, "test_tier1_random.csv")

    train_ds = SimulationDataset.load_csv(train_path)
    val_ds = SimulationDataset.load_csv(val_path)
    test_ds = SimulationDataset.load_csv(test_path)

    suite = SurrogateSuite(target_name=target_name)
    val_results = suite.train_all(train_ds._raw_data, val_ds._raw_data)
    test_results = suite.evaluate_test(test_ds._raw_data)

    suite.save_all(output_model_dir)

    comparison = {
        "target": target_name,
        "validation_metrics": val_results,
        "test_metrics": test_results,
    }
    return comparison
