"""Pydantic data schemas for FastAPI backend validation."""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class BuildingInput(BaseModel):
    """Building definition input."""

    num_storeys: int = Field(5, ge=1, le=50, description="Number of storeys")
    storey_mass_kg: Optional[float] = Field(None, description="Uniform storey mass in kg")
    storey_stiffness_n_m: Optional[float] = Field(None, description="Uniform lateral stiffness in N/m")
    storey_height_m: float = Field(3.5, gt=0, description="Storey height in meters")
    masses: Optional[List[float]] = Field(None, description="Custom mass per floor [m1, ..., mN] in kg")
    stiffnesses: Optional[List[float]] = Field(None, description="Custom stiffness per storey [k1, ..., kN] in N/m")
    heights: Optional[List[float]] = Field(None, description="Custom heights per storey [h1, ..., hN] in m")
    damping_ratio: float = Field(0.05, ge=0.001, le=0.30, description="Viscous damping ratio zeta")
    name: str = Field("Building_Model", description="Building identifier label")


class PredictRequest(BaseModel):
    """Request schema for /predict."""

    building: BuildingInput
    earthquake_name: str = Field("El_Centro_1940_NS", description="Earthquake record name from database")
    model_name: str = Field("GradientBoosting", description="ML Surrogate model (GradientBoosting, NeuralMLP, RandomForest, LinearRidge)")
    scale_factor: float = Field(1.0, gt=0.0, le=10.0, description="Ground motion amplitude scaling factor")


class PredictResponse(BaseModel):
    """Response schema for /predict."""

    status: str = "success"
    target: str = "target_max_pidr"
    predicted_max_pidr: float
    predicted_max_pidr_pct: float
    predicted_peak_base_shear_kn: float
    predicted_peak_base_shear_n: float
    inference_time_ms: float
    model_traceability: Dict[str, Any]
    features_used: Dict[str, float]


class SimulateRequest(BaseModel):
    """Request schema for /simulate."""

    building: BuildingInput
    earthquake_name: str = Field("El_Centro_1940_NS", description="Earthquake record name from database")
    scale_factor: float = Field(1.0, gt=0.0, le=10.0, description="Ground motion amplitude scaling factor")


class SimulateResponse(BaseModel):
    """Response schema for /simulate."""

    status: str = "success"
    true_max_pidr: float
    true_max_pidr_pct: float
    true_peak_base_shear_kn: float
    true_peak_base_shear_n: float
    true_max_roof_disp_m: float
    fundamental_period_t1_s: float
    simulation_time_ms: float
    time_history_preview: Dict[str, List[float]]
    storey_drift_profile: List[float]
    modal_properties: Dict[str, Any]


class IS1893CompareRequest(BaseModel):
    """Request schema for /is1893/compare."""

    building: BuildingInput
    earthquake_name: str = Field("El_Centro_1940_NS", description="Earthquake record to test against")
    zone: str = Field("Zone IV", description="IS 1893 Seismic Zone (Zone II, Zone III, Zone IV, Zone V)")
    soil_type: str = Field("Type II (Medium)", description="IS 1893 Soil Type (Type I, Type II, Type III)")
    model_name: str = Field("GradientBoosting", description="ML surrogate model to use")


class IS1893CompareResponse(BaseModel):
    """Response schema for /is1893/compare."""

    status: str = "success"
    building_name: str
    num_storeys: int
    comparison_table: List[Dict[str, Any]]
    is1893_details: Dict[str, Any]
    physics_solver_details: Dict[str, Any]
    surrogate_details: Dict[str, Any]
    speedup_factor: float


class HealthResponse(BaseModel):
    """Response schema for /health."""

    status: str = "healthy"
    app_name: str = "Seismic-AI Backend API"
    version: str = "2.0.0"
    registered_models: Dict[str, List[str]]
    available_earthquakes: int
    available_indian_earthquakes: int
    benchmark_buildings_count: int
    traceability_enforced: bool = True
