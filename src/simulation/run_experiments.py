"""Master experiment runner executing surrogate training, 4-tier generalization, sensitivity, and optimization."""

import os
import json
import numpy as np

from src.simulation.dataset import SimulationDataset
from src.ml.train import train_surrogate_suite
from src.uncertainty.generalization import GeneralizationEvaluator
from src.uncertainty.sensitivity import MonteCarloSensitivity
from src.ml.models import RandomForestSurrogate, GradientBoostingSurrogate, NeuralSurrogate, LinearRidgeSurrogate
from src.earthquake.database import GroundMotionDatabase
from src.optimization.problem import SeismicOptimizationProblem
from src.optimization.optimizer import DifferentialEvolutionOptimizer
from src.optimization.verifier import ClosedLoopVerifier


def run_all_experiments():
    print("==========================================================")
    print("      SEISMIC-AI MASTER EXPERIMENT PIPELINE")
    print("==========================================================")

    # 1. Train Surrogate Suite
    print("\n>>> STEP 1: Training Surrogate Suite on Tier 1 (Random Split)...")
    comp_pidr = train_surrogate_suite(target_name="target_max_pidr")
    comp_vbase = train_surrogate_suite(target_name="target_peak_base_shear_N")

    os.makedirs("results/model_comparison", exist_ok=True)
    with open("results/model_comparison/surrogate_comparison_pidr.json", "w") as f:
        json.dump(comp_pidr, f, indent=2)
    with open("results/model_comparison/surrogate_comparison_base_shear.json", "w") as f:
        json.dump(comp_vbase, f, indent=2)

    # 2. Run 4-Tier Generalization Protocol
    print("\n>>> STEP 2: Running 4-Tier Scientific Generalization Protocol...")
    evaluator = GeneralizationEvaluator(dataset_dir="data/datasets/simulation_dataset")
    gen_results = evaluator.evaluate_4tiers(target_name="target_max_pidr", output_results_dir="results/generalization")

    # 3. Monte Carlo Sensitivity Analysis
    print("\n>>> STEP 3: Running Monte Carlo Parameter Sensitivity Analysis...")
    # Load trained RandomForest surrogate
    rf_model = RandomForestSurrogate.load("models/trained/RandomForest_target_max_pidr.pkl")
    test_ds = SimulationDataset.load_csv("data/datasets/simulation_dataset/test_tier1_random.csv")
    feat_cols = test_ds.get_feature_columns()

    nom_sample = np.array([float(test_ds._raw_data[0][c]) for c in feat_cols], dtype=np.float64)
    mc = MonteCarloSensitivity(surrogate=rf_model, seed=42)
    mc_results = mc.analyze_perturbations(
        nominal_features=nom_sample,
        noise_levels=[0.02, 0.05, 0.10, 0.15, 0.20],
        num_mc_samples=500,
    )

    os.makedirs("results/uncertainty", exist_ok=True)
    with open("results/uncertainty/monte_carlo_sensitivity.json", "w") as f:
        json.dump(mc_results, f, indent=2)
    print(f"[+] Monte Carlo Sensitivity saved to: results/uncertainty/monte_carlo_sensitivity.json")

    # 4. Seismic Structural Optimization & Closed-Loop Verification
    print("\n>>> STEP 4: Running Seismic Design Optimization & Closed-Loop Verification...")
    db = GroundMotionDatabase()
    design_eq = db.get_record("Kobe_1995_NS")  # Severe design earthquake

    opt_problem = SeismicOptimizationProblem(
        num_storeys=5,
        floor_mass_kg=120000.0,
        storey_height_m=3.5,
        design_earthquake=design_eq,
        surrogate_model=rf_model,
        feature_columns=feat_cols,
        max_allowable_idr=0.010,  # 1.0% drift limit
        k_bounds=(6e7, 3.5e8),
        damping_ratio=0.05,
    )

    optimizer = DifferentialEvolutionOptimizer(
        problem=opt_problem,
        pop_size=35,
        max_generations=30,
        seed=42,
    )
    opt_output = optimizer.optimize()

    # Closed-loop verification
    verification = ClosedLoopVerifier.verify_design(
        problem=opt_problem,
        optimal_stiffnesses=opt_output["optimal_stiffnesses"],
    )

    opt_summary = {
        "optimization": opt_output,
        "verification": verification,
    }
    os.makedirs("results/optimization", exist_ok=True)
    with open("results/optimization/design_optimization_verified.json", "w") as f:
        json.dump(opt_summary, f, indent=2)
    print(f"[+] Optimization & Verification saved to: results/optimization/design_optimization_verified.json")

    print("\n==========================================================")
    print("      ALL EXPERIMENTS SUCCESSFULLY EXECUTED")
    print("==========================================================")


if __name__ == "__main__":
    run_all_experiments()
