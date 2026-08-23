"""4-Tier Scientific Generalization testing framework for surrogate models."""

from typing import Dict, Any, List, Optional
import os
import json
import numpy as np

from src.simulation.dataset import SimulationDataset
from src.ml.models import BaseSurrogate, RandomForestSurrogate, GradientBoostingSurrogate, NeuralSurrogate, LinearRidgeSurrogate
from src.ml.evaluate import compute_regression_metrics


class GeneralizationEvaluator:
    """Evaluates surrogate models across the 4-Tier Scientific Generalization Protocol."""

    def __init__(self, dataset_dir: str = "data/datasets/simulation_dataset") -> None:
        self.dataset_dir = dataset_dir

    def evaluate_4tiers(
        self,
        target_name: str = "target_max_pidr",
        output_results_dir: str = "results/generalization",
    ) -> Dict[str, Any]:
        """Run evaluation across Tier 1 (Random), Tier 2 (Unseen EQ), Tier 3 (Unseen Bldg), and Tier 4 (Dual Unseen).

        Parameters
        ----------
        target_name : str
            Target EDP column name.
        output_results_dir : str
            Directory to save evaluation summary.

        Returns
        -------
        Dict[str, Any]
            Comparative results table and metadata.
        """
        os.makedirs(output_results_dir, exist_ok=True)

        tiers = {
            "Tier 1 (Random Split)": ("train_tier1_random.csv", "test_tier1_random.csv"),
            "Tier 2 (Unseen Earthquakes)": ("train_tier2_unseen_eq.csv", "test_tier2_unseen_eq.csv"),
            "Tier 3 (Unseen Buildings)": ("train_tier3_unseen_bldg.csv", "test_tier3_unseen_bldg.csv"),
            "Tier 4 (Dual Unseen Blind)": ("train_tier4_dual_unseen.csv", "test_tier4_dual_unseen.csv"),
        }

        tier_results = {}

        for tier_name, (train_file, test_file) in tiers.items():
            print(f"[*] Evaluating {tier_name}...")
            train_path = os.path.join(self.dataset_dir, train_file)
            test_path = os.path.join(self.dataset_dir, test_file)

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                print(f"    [!] Skipping {tier_name}: files not found.")
                continue

            train_ds = SimulationDataset.load_csv(train_path)
            test_ds = SimulationDataset.load_csv(test_path)

            feat_cols = train_ds.get_feature_columns()

            def to_arrays(ds_raw):
                x = np.array([[float(r[c]) for c in feat_cols] for r in ds_raw], dtype=np.float64)
                y = np.array([float(r[target_name]) for r in ds_raw], dtype=np.float64)
                return x, y

            x_train, y_train = to_arrays(train_ds._raw_data)
            x_test, y_test = to_arrays(test_ds._raw_data)

            models = {
                "LinearRidge": LinearRidgeSurrogate(alpha=2.0),
                "RandomForest": RandomForestSurrogate(n_estimators=45, max_depth=7, seed=42),
                "GradientBoosting": GradientBoostingSurrogate(n_estimators=50, learning_rate=0.12, max_depth=4, seed=42),
                "NeuralMLP": NeuralSurrogate(hidden_layers=(64, 32, 16), learning_rate=0.006, max_epochs=100, seed=42),
            }

            model_metrics = {}
            for m_name, model in models.items():
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                metrics = compute_regression_metrics(y_test, y_pred)
                model_metrics[m_name] = metrics
                print(f"    -> {m_name}: R^2 = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.5f}")

            tier_results[tier_name] = {
                "train_samples": len(x_train),
                "test_samples": len(x_test),
                "models": model_metrics,
            }

        # Save JSON results
        output_file = os.path.join(output_results_dir, f"generalization_4tier_{target_name}.json")
        with open(output_file, "w") as f:
            json.dump(tier_results, f, indent=2)

        print(f"[+] 4-Tier Generalization evaluation saved to: {output_file}")
        return tier_results
