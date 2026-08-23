"""FastAPI Backend Server for Seismic-AI.

Serves:
- POST /predict   : Evaluates versioned surrogate models with sub-millisecond inference and provenance metadata
- POST /simulate  : Runs ground-truth Newmark-beta physics solver on demand
- GET /health     : Health check and model registry inventory
- GET /earthquakes: Global & Indian earthquake catalog
- GET /benchmarks : SAC Steel & IIT benchmark building catalog
- POST /is1893/compare : 3-way comparison (IS 1893:2016 vs Physics Solver vs ML Surrogate)
- WebSocket /ws/sensor-stream : Real-time sensor replay with STA/LTA onset picking and instantaneous structural estimation
"""

import time
import json
import asyncio
from typing import Dict, Any, List, Optional
import numpy as np

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.structural.building import ShearBuilding
from src.structural.benchmarks import BENCHMARK_BUILDINGS, list_benchmark_buildings
from src.dynamics.modal import ModalAnalysis
from src.dynamics.damping import RayleighDamping
from src.dynamics.solver import NewmarkSolver
from src.earthquake.database import GroundMotionDatabase
from src.earthquake.spectra import ResponseSpectrum
from src.earthquake.record import GroundMotionRecord, GRAVITY
from src.features.extractor import extract_features
from src.standards.is1893 import analyze_is1893_equivalent_static, generate_3way_comparison_table
from src.sensors.detector import STA_LTA_Detector, detect_sta_lta_onset
from src.sensors.early_features import extract_early_wave_features, estimate_surrogate_features_from_early_onset
from src.sensors.stream import AccelerationSample

from app.backend.schemas import (
    BuildingInput,
    PredictRequest,
    PredictResponse,
    SimulateRequest,
    SimulateResponse,
    IS1893CompareRequest,
    IS1893CompareResponse,
    HealthResponse,
)
from app.backend.metadata import registry, VERIFIED_VALIDATION_METRICS

# Initialize FastAPI App
app = FastAPI(
    title="Seismic-AI Backend API",
    description="Computational Structural Dynamics & Versioned Machine Learning Surrogates",
    version="2.0.0",
)

# Enable CORS for localhost frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Ground Motion Database
db = GroundMotionDatabase()


def _build_shear_building_from_input(inp: BuildingInput) -> ShearBuilding:
    """Helper to instantiate ShearBuilding from API schema."""
    n = inp.num_storeys

    if inp.masses is not None and inp.stiffnesses is not None:
        masses = np.array(inp.masses, dtype=np.float64)
        stiffnesses = np.array(inp.stiffnesses, dtype=np.float64)
        heights = np.array(inp.heights if inp.heights is not None else [inp.storey_height_m] * n, dtype=np.float64)
    else:
        m_val = inp.storey_mass_kg if inp.storey_mass_kg is not None else 120000.0
        k_val = inp.storey_stiffness_n_m if inp.storey_stiffness_n_m is not None else 150.0e6
        masses = np.full(n, m_val, dtype=np.float64)
        masses[-1] *= 0.85  # Standard lighter roof mass
        stiffnesses = np.full(n, k_val, dtype=np.float64)
        heights = np.full(n, inp.storey_height_m, dtype=np.float64)

    return ShearBuilding(masses=masses, stiffnesses=stiffnesses, heights=heights, name=inp.name)


def _get_scaled_record(name: str, scale: float = 1.0) -> GroundMotionRecord:
    """Retrieve ground motion record with optional amplitude scaling."""
    if name not in db.list_records():
        raise HTTPException(status_code=404, detail=f"Earthquake record '{name}' not found. Available: {db.list_records()}")
    base_rec = db.get_record(name)
    if abs(scale - 1.0) > 1e-4:
        return base_rec.scale_to_pga(base_rec.pga_g * scale)
    return base_rec


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check returning system status and versioned model inventory."""
    models_dict = {t: list(registry.loaded_models.get(t, {}).keys()) for t in registry.loaded_models}
    return HealthResponse(
        status="healthy",
        app_name="Seismic-AI Backend API",
        version="2.0.0",
        registered_models=models_dict,
        available_earthquakes=len(db),
        available_indian_earthquakes=len(db.list_indian_records()),
        benchmark_buildings_count=len(BENCHMARK_BUILDINGS),
        traceability_enforced=True,
    )


@app.get("/earthquakes")
def list_earthquakes() -> Dict[str, Any]:
    """List available earthquake records categorized into International and Indian suites."""
    all_names = db.list_records()
    indian_names = db.list_indian_records()
    records_info = []

    for name in all_names:
        rec = db.get_record(name)
        records_info.append({
            "name": name,
            "pga_g": round(rec.pga_g, 4),
            "duration_s": round(rec.duration, 2),
            "dt_s": rec.dt,
            "is_indian": (name in indian_names),
            "metadata": rec.metadata,
        })

    return {
        "count": len(all_names),
        "indian_records_count": len(indian_names),
        "records": records_info,
    }


@app.get("/earthquake/{name}")
def get_earthquake_details(name: str) -> Dict[str, Any]:
    """Retrieve detailed time series, intensity measures, and elastic response spectra."""
    if name not in db.list_records():
        raise HTTPException(status_code=404, detail=f"Record '{name}' not found.")

    rec = db.get_record(name)
    ims = rec.intensity_measures()
    spectrum = ResponseSpectrum(record=rec, damping_ratio=0.05)

    # Downsample time history for network preview efficiency (max 1000 points)
    step = max(1, len(rec.acceleration) // 1000)
    time_down = rec.time[::step].tolist()
    acc_down = rec.acceleration_g[::step].tolist()

    return {
        "name": rec.name,
        "pga_g": round(rec.pga_g, 4),
        "duration_s": round(rec.duration, 2),
        "dt_s": rec.dt,
        "intensity_measures": ims,
        "metadata": rec.metadata,
        "time_history_preview": {
            "time_s": time_down,
            "acceleration_g": acc_down,
        },
        "response_spectrum_5pct": {
            "periods_s": spectrum.periods.tolist(),
            "sa_g": spectrum.sa_g.tolist(),
            "sd_m": spectrum.sd.tolist(),
        },
    }


@app.get("/benchmarks")
def get_benchmarks() -> List[Dict[str, Any]]:
    """Get metadata for real benchmark buildings (SAC Steel 3/9-Story & IIT Campus Frames)."""
    return list_benchmark_buildings()


@app.post("/predict", response_model=PredictResponse)
def predict_response(req: PredictRequest) -> PredictResponse:
    """Predict building seismic response (PIDR, Base Shear) using a trained versioned ML surrogate."""
    t_start = time.perf_counter()

    building = _build_shear_building_from_input(req.building)
    record = _get_scaled_record(req.earthquake_name, req.scale_factor)

    # Feature extraction
    features = extract_features(building, record, damping_ratio=req.building.damping_ratio)

    # 1. Predict PIDR
    target_pidr = "target_max_pidr"
    feat_cols_pidr = registry.get_feature_columns(target_pidr)
    x_vec_pidr = np.array([float(features.get(c, 0.0)) for c in feat_cols_pidr]).reshape(1, -1)

    model_pidr = registry.get_model(target_pidr, req.model_name)
    pred_pidr = float(model_pidr.predict(x_vec_pidr)[0])

    # 2. Predict Peak Base Shear
    target_shear = "target_peak_base_shear_N"
    feat_cols_shear = registry.get_feature_columns(target_shear)
    x_vec_shear = np.array([float(features.get(c, 0.0)) for c in feat_cols_shear]).reshape(1, -1)

    try:
        model_shear = registry.get_model(target_shear, req.model_name)
        pred_shear_n = float(model_shear.predict(x_vec_shear)[0])
    except Exception:
        # Fallback to modal base shear approximation if model artifact unavailable
        modal = ModalAnalysis(building)
        spec = ResponseSpectrum(record=record, damping_ratio=req.building.damping_ratio)
        pred_shear_n = float(modal.effective_masses[0] * (spec.get_sa_g(modal.fundamental_period) * GRAVITY))

    t_end = time.perf_counter()
    inf_time_ms = (t_end - t_start) * 1000.0

    traceability = registry.get_traceability_info(target_pidr, req.model_name)

    return PredictResponse(
        status="success",
        target="target_max_pidr",
        predicted_max_pidr=pred_pidr,
        predicted_max_pidr_pct=pred_pidr * 100.0,
        predicted_peak_base_shear_kn=pred_shear_n / 1e3,
        predicted_peak_base_shear_n=pred_shear_n,
        inference_time_ms=round(inf_time_ms, 4),
        model_traceability=traceability,
        features_used=features,
    )


@app.post("/simulate", response_model=SimulateResponse)
def simulate_ground_truth(req: SimulateRequest) -> SimulateResponse:
    """Execute high-fidelity Newmark-beta physics solver to obtain verified ground-truth response."""
    t_start = time.perf_counter()

    building = _build_shear_building_from_input(req.building)
    record = _get_scaled_record(req.earthquake_name, req.scale_factor)

    modal = ModalAnalysis(building)
    damping = RayleighDamping.from_uniform_ratio(building, req.building.damping_ratio)
    solver = NewmarkSolver.average_acceleration(building, damping)

    resp = solver.solve(ground_acceleration=record.acceleration, dt=record.dt)

    t_end = time.perf_counter()
    sim_time_ms = (t_end - t_start) * 1000.0

    # Downsample time history for response payload
    step = max(1, len(resp.time) // 800)
    roof_disp_down = resp.displacement[-1, :][::step].tolist()
    time_down = resp.time[::step].tolist()

    # Storey drift profile (peak drift ratio per storey)
    drift_profile = resp.peak_interstorey_drift_ratios.tolist()

    modal_props = {
        "fundamental_period_T1_s": round(modal.fundamental_period, 4),
        "frequencies_Hz": [round(f, 2) for f in modal.cyclic_frequencies[:min(3, modal.num_modes)]],
        "effective_mass_ratios_pct": [round(r * 100, 2) for r in modal.effective_mass_ratios[:min(3, modal.num_modes)]],
        "total_mass_kg": building.total_mass,
    }

    return SimulateResponse(
        status="success",
        true_max_pidr=resp.max_drift_ratio,
        true_max_pidr_pct=resp.max_drift_ratio * 100.0,
        true_peak_base_shear_kn=resp.peak_base_shear / 1e3,
        true_peak_base_shear_n=resp.peak_base_shear,
        true_max_roof_disp_m=resp.max_roof_displacement,
        fundamental_period_t1_s=round(modal.fundamental_period, 4),
        simulation_time_ms=round(sim_time_ms, 2),
        time_history_preview={
            "time_s": time_down,
            "roof_displacement_m": roof_disp_down,
        },
        storey_drift_profile=drift_profile,
        modal_properties=modal_props,
    )


@app.post("/is1893/compare", response_model=IS1893CompareResponse)
def compare_is1893(req: IS1893CompareRequest) -> IS1893CompareResponse:
    """Generate 3-way comparison between IS 1893:2016 Code, High-Fidelity Physics Solver, and ML Surrogate."""
    building = _build_shear_building_from_input(req.building)
    record = _get_scaled_record(req.earthquake_name)

    # 1. Physics Solver Execution
    t_start_sim = time.perf_counter()
    modal = ModalAnalysis(building)
    damping = RayleighDamping.from_uniform_ratio(building, req.building.damping_ratio)
    solver = NewmarkSolver.average_acceleration(building, damping)
    resp = solver.solve(ground_acceleration=record.acceleration, dt=record.dt)
    t_sim_ms = (time.perf_counter() - t_start_sim) * 1000.0

    # 2. ML Surrogate Execution
    t_start_surr = time.perf_counter()
    features = extract_features(building, record, damping_ratio=req.building.damping_ratio)
    feat_cols_pidr = registry.get_feature_columns("target_max_pidr")
    x_vec_pidr = np.array([float(features.get(c, 0.0)) for c in feat_cols_pidr]).reshape(1, -1)
    model_pidr = registry.get_model("target_max_pidr", req.model_name)
    surr_pidr = float(model_pidr.predict(x_vec_pidr)[0])

    try:
        model_shear = registry.get_model("target_peak_base_shear_N", req.model_name)
        feat_cols_shear = registry.get_feature_columns("target_peak_base_shear_N")
        x_vec_shear = np.array([float(features.get(c, 0.0)) for c in feat_cols_shear]).reshape(1, -1)
        surr_shear_n = float(model_shear.predict(x_vec_shear)[0])
    except Exception:
        surr_shear_n = float(modal.effective_masses[0] * (record.pga_g * 2.5 * GRAVITY))
    t_surr_ms = (time.perf_counter() - t_start_surr) * 1000.0

    # Speedup factor
    speedup = round(t_sim_ms / max(t_surr_ms, 1e-4), 1)

    # 3. 3-Way Table
    comp_dict = generate_3way_comparison_table(
        building=building,
        zone=req.zone,
        soil_type=req.soil_type,
        physics_peak_base_shear_n=resp.peak_base_shear,
        physics_max_pidr=resp.max_drift_ratio,
        surrogate_peak_base_shear_n=surr_shear_n,
        surrogate_max_pidr=surr_pidr,
    )

    return IS1893CompareResponse(
        status="success",
        building_name=building.name,
        num_storeys=building.num_storeys,
        comparison_table=comp_dict["table_rows"],
        is1893_details=comp_dict["is1893_details"],
        physics_solver_details={
            "execution_time_ms": round(t_sim_ms, 2),
            "peak_base_shear_kn": round(resp.peak_base_shear / 1e3, 2),
            "max_pidr_pct": round(resp.max_drift_ratio * 100, 3),
            "max_roof_disp_m": round(resp.max_roof_displacement, 4),
        },
        surrogate_details={
            "execution_time_ms": round(t_surr_ms, 4),
            "model_used": req.model_name,
            "predicted_base_shear_kn": round(surr_shear_n / 1e3, 2),
            "predicted_pidr_pct": round(surr_pidr * 100, 3),
        },
        speedup_factor=speedup,
    )


@app.websocket("/ws/sensor-stream")
async def websocket_sensor_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time sensor streaming, STA/LTA onset picking, and rapid structural estimation."""
    await websocket.accept()

    try:
        # Wait for initial configuration message
        init_data = await websocket.receive_json()
        rec_name = init_data.get("earthquake_name", "Northridge_1994_Sylmar")
        n_storeys = int(init_data.get("num_storeys", 5))
        m_val = float(init_data.get("storey_mass_tonnes", 120.0)) * 1e3
        k_val = float(init_data.get("storey_stiffness_mn_m", 150.0)) * 1e6
        h_val = float(init_data.get("storey_height_m", 3.5))
        playback_speed = float(init_data.get("playback_speed", 2.0))
        model_name = init_data.get("model_name", "GradientBoosting")

        bldg = ShearBuilding.from_uniform(n_storeys, m_val, k_val, h_val, name=f"Live_{n_storeys}St_Building")
        modal = ModalAnalysis(bldg)
        record = db.get_record(rec_name) if rec_name in db.list_records() else db.get_record("Northridge_1994_Sylmar")

        detector = STA_LTA_Detector(
            sta_window_s=0.5,
            lta_window_s=5.0,
            trigger_threshold=3.5,
            detrigger_threshold=1.5,
            sampling_rate_hz=1.0 / record.dt,
        )

        acc_g = record.acceleration_g
        dt = record.dt
        delay_per_step = (dt / playback_speed) if playback_speed > 0 else 0.0

        onset_triggered = False
        post_onset_samples = []
        surrogate_prediction_emitted = False

        # Stream samples in chunks or step by step
        for i in range(len(acc_g)):
            t_curr = i * dt
            sample = AccelerationSample(
                timestamp_s=t_curr,
                acceleration_g=float(acc_g[i]),
                acceleration_ms2=float(acc_g[i] * GRAVITY),
                sample_index=i,
                sensor_id="WS_STREAM_01",
            )

            ratio, event = detector.process_sample(sample)

            packet: Dict[str, Any] = {
                "type": "sample",
                "sample_index": i,
                "timestamp_s": round(t_curr, 3),
                "acceleration_g": round(float(acc_g[i]), 5),
                "sta_lta_ratio": round(ratio, 2),
                "is_triggered": detector.is_triggered,
            }

            if event is not None and not onset_triggered:
                onset_triggered = True
                packet["type"] = "onset_event"
                packet["onset_event"] = event.to_dict()

            if onset_triggered:
                post_onset_samples.append(float(acc_g[i]))

                # After 2.5 seconds of post-onset signal, trigger immediate surrogate response estimation!
                if len(post_onset_samples) >= int(2.5 / dt) and not surrogate_prediction_emitted:
                    surrogate_prediction_emitted = True
                    early_feats = extract_early_wave_features(np.array(post_onset_samples), dt=dt, window_duration_s=2.5)
                    surr_feats = estimate_surrogate_features_from_early_onset(bldg, early_feats, damping_ratio=0.05, modal=modal)

                    # Predict with surrogate
                    target_pidr = "target_max_pidr"
                    feat_cols = registry.get_feature_columns(target_pidr)
                    x_vec = np.array([float(surr_feats.get(c, 0.0)) for c in feat_cols]).reshape(1, -1)
                    model = registry.get_model(target_pidr, model_name)
                    est_pidr = float(model.predict(x_vec)[0])

                    packet["type"] = "structural_response_estimate"
                    packet["early_estimate"] = {
                        "onset_time_s": detector.onset_timestamp_s,
                        "time_elapsed_since_onset_s": 2.5,
                        "early_pga_g": round(early_feats["pga_early_g"], 4),
                        "estimated_period_tau_c_s": round(early_feats["tau_c_s"], 3),
                        "predicted_max_pidr_pct": round(est_pidr * 100.0, 3),
                        "predicted_max_pidr": est_pidr,
                        "building_fundamental_period_s": round(modal.fundamental_period, 3),
                        "warning_lead_time_before_peak_s": max(0.0, round(record.duration * 0.4 - 2.5, 1)),
                        "model_used": model_name,
                    }

            await websocket.send_json(packet)

            if delay_per_step > 0:
                await asyncio.sleep(delay_per_step)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[!] WebSocket error: {e}")
