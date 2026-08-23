"""Seismic structural design optimization problem formulation."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.modal import ModalAnalysis
from src.earthquake.record import GroundMotionRecord
from src.earthquake.spectra import ResponseSpectrum
from src.features.extractor import extract_features
from src.ml.models import BaseSurrogate


class SeismicOptimizationProblem:
    """Formulates multi-storey shear building stiffness/mass optimization.

    Objective:
        Minimize total stiffness/cost: f(k) = sum(k_i)
    Subject to:
        PIDR_max(k) <= max_allowable_idr (e.g. 1.0% = 0.010)
        k_1 >= k_2 >= ... >= k_N (monotonic stiffness profile)
        k_min <= k_i <= k_max
    """

    def __init__(
        self,
        num_storeys: int,
        floor_mass_kg: float,
        storey_height_m: float,
        design_earthquake: GroundMotionRecord,
        surrogate_model: BaseSurrogate,
        feature_columns: List[str],
        max_allowable_idr: float = 0.010,
        k_bounds: Tuple[float, float] = (5e7, 3.5e8),
        damping_ratio: float = 0.05,
    ) -> None:
        self.num_storeys = num_storeys
        self.floor_mass_kg = floor_mass_kg
        self.storey_height_m = storey_height_m
        self.design_earthquake = design_earthquake
        self.surrogate = surrogate_model
        self.feature_columns = feature_columns
        self.max_allowable_idr = float(max_allowable_idr)
        self.k_bounds = k_bounds
        self.damping_ratio = damping_ratio

        # Pre-compute design earthquake spectrum
        self.spectrum = ResponseSpectrum(record=design_earthquake, damping_ratio=damping_ratio)

    def evaluate_surrogate(self, stiffness_vector: np.ndarray) -> Dict[str, float]:
        """Evaluate surrogate predicted drift and objective value for a candidate stiffness vector."""
        k_arr = np.asarray(stiffness_vector, dtype=np.float64).flatten()
        masses = np.full(self.num_storeys, self.floor_mass_kg)
        masses[-1] = masses[-1] * 0.8  # lighter roof

        bldg = ShearBuilding(
            masses=masses,
            stiffnesses=k_arr,
            heights=self.storey_height_m,
            name="CandidateBuilding",
        )

        modal = ModalAnalysis(bldg)
        feat_dict = extract_features(
            building=bldg,
            record=self.design_earthquake,
            spectrum=self.spectrum,
            damping_ratio=self.damping_ratio,
            modal=modal,
        )

        # Form feature vector in identical order
        x_vec = np.array([float(feat_dict[c]) for c in self.feature_columns], dtype=np.float64).reshape(1, -1)
        pred_pidr = float(self.surrogate.predict(x_vec)[0])

        # Objective: Total lateral stiffness (proxy for member structural material cost)
        total_stiffness = float(np.sum(k_arr))

        # Penalty for constraint violations
        drift_violation = max(0.0, pred_pidr - self.max_allowable_idr)
        
        # Monotonicity violation: penalize if upper storey is stiffer than lower storey
        mono_violations = np.maximum(0.0, k_arr[1:] - k_arr[:-1])
        total_mono_violation = float(np.sum(mono_violations))

        penalty = 1e11 * (drift_violation / self.max_allowable_idr) + 1e5 * total_mono_violation
        loss = total_stiffness + penalty

        return {
            "loss": loss,
            "total_stiffness": total_stiffness,
            "predicted_pidr": pred_pidr,
            "drift_violation": drift_violation,
            "is_feasible": bool(drift_violation == 0.0 and total_mono_violation == 0.0),
        }
