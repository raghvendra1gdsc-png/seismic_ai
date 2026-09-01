"""Unit tests for NSGA-II Multi-Objective Resilient Structural Optimization."""

import pytest
import numpy as np
import json

from src.earthquake.database import GroundMotionDatabase
from src.ml.models import GradientBoostingSurrogate
from src.optimization.multiobjective import NSGA2Optimizer, ParetoSolution, ParetoFrontierResult


def test_nsga2_multiobjective_optimizer():
    """Test NSGA-II Pareto frontier generation with trained surrogate model."""
    db = GroundMotionDatabase()
    rec = db.get_record("Kobe_1995_NS")

    with open("models/trained/metadata_target_max_pidr.json", "r") as f:
        meta = json.load(f)
    surr = GradientBoostingSurrogate.load("models/trained/GradientBoosting_target_max_pidr.pkl")

    optimizer = NSGA2Optimizer(
        num_storeys=3,
        storey_mass=100000.0,
        storey_height=3.5,
        k_bounds=(8e7, 3e8),
        population_size=16,
        num_generations=5,
    )

    res = optimizer.optimize(
        record=rec,
        surrogate_fn=lambda x: float(surr.predict(x)[0]),
        feature_columns=meta["feature_columns"],
        drift_limit_pct=2.0,
    )

    assert isinstance(res, ParetoFrontierResult)
    assert len(res.pareto_front) > 0
    # Solutions on Pareto front should have monotonic cost progression
    costs = [s.f1_carbon_mass_score for s in res.pareto_front]
    drifts = [s.f2_seismic_drift_pct for s in res.pareto_front]
    assert np.all(np.diff(costs) >= 0.0)

    # Tradeoff: Higher stiffness cost should generally achieve lower seismic drift
    assert res.best_safety_solution.f2_seismic_drift_pct <= res.best_cost_solution.f2_seismic_drift_pct
    assert res.compromise_balanced_solution is not None
