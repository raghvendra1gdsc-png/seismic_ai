"""Simulation runner executing time-history analysis and extracting dataset samples."""

from typing import Dict, Any, Optional
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.modal import ModalAnalysis
from src.dynamics.damping import RayleighDamping
from src.dynamics.solver import NewmarkSolver
from src.dynamics.response import DynamicResponse
from src.earthquake.record import GroundMotionRecord
from src.earthquake.spectra import ResponseSpectrum
from src.features.extractor import extract_features, extract_targets


class SimulationRunner:
    """Orchestrates physics simulation for a building and earthquake pair."""

    @staticmethod
    def run_single(
        building: ShearBuilding,
        record: GroundMotionRecord,
        damping_ratio: float = 0.05,
        spectrum: Optional[ResponseSpectrum] = None,
        modal: Optional[ModalAnalysis] = None,
    ) -> Dict[str, Any]:
        """Execute a single structural dynamic simulation and extract feature-target sample.

        Parameters
        ----------
        building : ShearBuilding
            Structural building model.
        record : GroundMotionRecord
            Earthquake ground motion record.
        damping_ratio : float, default=0.05
            Modal damping ratio.
        spectrum : Optional[ResponseSpectrum], optional
            Pre-computed response spectrum.
        modal : Optional[ModalAnalysis], optional
            Pre-computed modal analysis.

        Returns
        -------
        Dict[str, Any]
            Complete data record with identifier columns, features, and target EDPs.
        """
        if modal is None:
            modal = ModalAnalysis(building)

        if spectrum is None:
            spectrum = ResponseSpectrum(record=record, damping_ratio=damping_ratio)

        # Assemble damping and run Newmark integration
        damping = RayleighDamping.from_uniform_ratio(building, zeta=damping_ratio)
        solver = NewmarkSolver.average_acceleration(building, damping)

        response = solver.solve(
            ground_acceleration=record.acceleration,
            dt=record.dt,
        )

        # Extract features and targets
        features = extract_features(
            building=building,
            record=record,
            spectrum=spectrum,
            damping_ratio=damping_ratio,
            modal=modal,
        )
        targets = extract_targets(response)

        # Combine
        sample = {
            "building_id": building.name,
            "earthquake_id": record.name,
        }
        sample.update(features)
        sample.update(targets)

        return sample
