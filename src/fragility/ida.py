"""Incremental Dynamic Analysis (IDA) Engine.

Traces multi-record structural capacity curves from linear elasticity through
yielding and softening to global dynamic collapse (Vamvatsikos & Cornell, 2002).
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np

from src.structural.building import ShearBuilding
from src.earthquake.record import GroundMotionRecord
from src.dynamics.damping import RayleighDamping
from src.dynamics.solver import NewmarkSolver
from src.features.extractor import extract_features


@dataclass
class IDACurve:
    """Individual record Incremental Dynamic Analysis curve."""
    record_name: str
    im_levels_pga_g: np.ndarray      # Monotonically increasing intensity levels (PGA in g)
    edp_levels_pidr_pct: np.ndarray  # Resulting engineering demand parameter (Max PIDR in %)
    collapse_capacity_g: float       # Intensity at which slope falls below 20% of elastic slope
    yield_capacity_g: float          # Intensity at which yielding onset occurs


@dataclass
class IDAResult:
    """Aggregated multi-record IDA summary."""
    building_name: str
    record_count: int
    curves: List[IDACurve]
    im_grid_g: np.ndarray
    percentile_16_pidr_pct: np.ndarray
    median_50_pidr_pct: np.ndarray
    percentile_84_pidr_pct: np.ndarray
    median_collapse_capacity_g: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "building_name": self.building_name,
            "record_count": self.record_count,
            "im_grid_g": self.im_grid_g.tolist(),
            "percentile_16": self.percentile_16_pidr_pct.tolist(),
            "median_50": self.median_50_pidr_pct.tolist(),
            "percentile_84": self.percentile_84_pidr_pct.tolist(),
            "median_collapse_capacity_g": self.median_collapse_capacity_g,
        }


class IncrementalDynamicAnalysis:
    """Performs Incremental Dynamic Analysis using direct physics solver or ML surrogate acceleration.

    Parameters
    ----------
    building : ShearBuilding
        Target structural building model.
    damping_ratio : float, default=0.05
        Inherent viscous damping ratio.
    im_min_g : float, default=0.05
        Minimum PGA intensity level in g.
    im_max_g : float, default=2.50
        Maximum PGA intensity level in g.
    num_scale_points : int, default=20
        Number of intensity scaling steps per record.
    """

    def __init__(
        self,
        building: ShearBuilding,
        damping_ratio: float = 0.05,
        im_min_g: float = 0.05,
        im_max_g: float = 2.50,
        num_scale_points: int = 20,
    ) -> None:
        self.building = building
        self.damping_ratio = damping_ratio
        self.im_grid = np.linspace(im_min_g, im_max_g, num_scale_points)
        self.damp = RayleighDamping.from_uniform_ratio(building, damping_ratio)
        self.solver = NewmarkSolver.average_acceleration(building, self.damp)

    def run_record_physics(self, record: GroundMotionRecord) -> IDACurve:
        """Run step-by-step physical time-history integration over all scale factors."""
        base_pga = record.pga_g
        pidr_vals = []

        for target_im in self.im_grid:
            scale_fac = target_im / max(base_pga, 1e-4)
            scaled_ag = record.acceleration * scale_fac
            resp = self.solver.solve(ground_acceleration=scaled_ag, dt=record.dt)
            pidr_vals.append(resp.max_drift_ratio * 100.0)

        pidr_arr = np.array(pidr_vals)
        collapse_im = self._detect_collapse_capacity(self.im_grid, pidr_arr)
        yield_im = self._detect_yield_capacity(self.im_grid, pidr_arr)

        return IDACurve(
            record_name=record.name,
            im_levels_pga_g=self.im_grid.copy(),
            edp_levels_pidr_pct=pidr_arr,
            collapse_capacity_g=collapse_im,
            yield_capacity_g=yield_im,
        )

    def run_record_surrogate(
        self,
        record: GroundMotionRecord,
        surrogate_predict_fn: Callable[[np.ndarray], float],
        feature_columns: List[str],
    ) -> IDACurve:
        """Evaluate IDA curve in sub-milliseconds using trained ML surrogate."""
        base_pga = record.pga_g
        pidr_vals = []

        for target_im in self.im_grid:
            scale_fac = target_im / max(base_pga, 1e-4)
            scaled_rec = record.scale(scale_fac)
            feats = extract_features(self.building, scaled_rec, damping_ratio=self.damping_ratio)
            x_vec = np.array([float(feats[c]) for c in feature_columns]).reshape(1, -1)
            pred_pidr = float(surrogate_predict_fn(x_vec))
            pidr_vals.append(pred_pidr * 100.0)

        pidr_arr = np.array(pidr_vals)
        collapse_im = self._detect_collapse_capacity(self.im_grid, pidr_arr)
        yield_im = self._detect_yield_capacity(self.im_grid, pidr_arr)

        return IDACurve(
            record_name=record.name,
            im_levels_pga_g=self.im_grid.copy(),
            edp_levels_pidr_pct=pidr_arr,
            collapse_capacity_g=collapse_im,
            yield_capacity_g=yield_im,
        )

    def run_suite(
        self,
        records: List[GroundMotionRecord],
        use_surrogate: bool = False,
        surrogate_fn: Optional[Callable[[np.ndarray], float]] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> IDAResult:
        """Run IDA across a suite of ground-motion records and compute statistical percentiles."""
        curves = []
        for rec in records:
            if use_surrogate and surrogate_fn is not None and feature_columns is not None:
                c = self.run_record_surrogate(rec, surrogate_fn, feature_columns)
            else:
                c = self.run_record_physics(rec)
            curves.append(c)

        # Collect matrix of shape (num_records, num_im_points)
        edp_matrix = np.array([c.edp_levels_pidr_pct for c in curves])

        p16 = np.percentile(edp_matrix, 16, axis=0)
        p50 = np.percentile(edp_matrix, 50, axis=0)
        p84 = np.percentile(edp_matrix, 84, axis=0)

        collapse_capacities = [c.collapse_capacity_g for c in curves]
        median_collapse = float(np.median(collapse_capacities))

        return IDAResult(
            building_name=self.building.name or "Building",
            record_count=len(records),
            curves=curves,
            im_grid_g=self.im_grid.copy(),
            percentile_16_pidr_pct=p16,
            median_50_pidr_pct=p50,
            percentile_84_pidr_pct=p84,
            median_collapse_capacity_g=median_collapse,
        )

    def _detect_collapse_capacity(self, ims: np.ndarray, edps: np.ndarray) -> float:
        """Detect collapse capacity when tangent slope d(IM)/d(EDP) <= 0.20 * K_elastic or drift >= 4.0%."""
        # Elastic initial slope
        if len(edps) < 3 or edps[1] <= 0.0:
            return float(ims[-1])
        k_elastic = (ims[1] - ims[0]) / max(edps[1] - edps[0], 1e-4)

        for i in range(2, len(ims)):
            if edps[i] >= 4.0:  # 4% drift collapse limit
                return float(ims[i])
            slope = (ims[i] - ims[i-1]) / max(edps[i] - edps[i-1], 1e-4)
            if slope <= 0.20 * k_elastic:
                return float(ims[i])

        return float(ims[-1])

    def _detect_yield_capacity(self, ims: np.ndarray, edps: np.ndarray) -> float:
        """Detect yielding capacity when drift reaches 0.5%."""
        idx = np.where(edps >= 0.50)[0]
        if len(idx) > 0:
            return float(ims[idx[0]])
        return float(ims[-1])
