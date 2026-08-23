"""Tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns status and model registry inventory."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "registered_models" in data
    assert data["available_earthquakes"] > 0
    assert data["available_indian_earthquakes"] >= 5


def test_earthquakes_endpoints():
    """Verify /earthquakes and /earthquake/{name} endpoints."""
    resp = client.get("/earthquakes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 10
    assert data["indian_records_count"] >= 5

    # Details for Indian record (Chamoli)
    resp_chamoli = client.get("/earthquake/Chamoli_1999_Gopeshwar")
    assert resp_chamoli.status_code == 200
    det = resp_chamoli.json()
    assert det["name"] == "Chamoli_1999_Gopeshwar"
    assert "response_spectrum_5pct" in det
    assert "intensity_measures" in det


def test_benchmarks_endpoint():
    """Verify /benchmarks returns SAC Steel and IIT frames."""
    resp = client.get("/benchmarks")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 4
    labels = [d["label"] for d in data]
    assert any("SAC 3-Story" in l for l in labels)
    assert any("SAC 9-Story" in l for l in labels)


def test_predict_endpoint():
    """Verify POST /predict returns traceable surrogate prediction."""
    payload = {
        "building": {
            "num_storeys": 5,
            "storey_mass_kg": 120000.0,
            "storey_stiffness_n_m": 150000000.0,
            "storey_height_m": 3.5,
            "damping_ratio": 0.05,
            "name": "Test_Building_5St",
        },
        "earthquake_name": "El_Centro_1940_NS",
        "model_name": "GradientBoosting",
        "scale_factor": 1.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["predicted_max_pidr"] > 0.0
    assert data["predicted_peak_base_shear_kn"] > 0.0
    assert data["inference_time_ms"] >= 0.0
    assert "model_traceability" in data
    assert data["model_traceability"]["traceability_status"] == "VERIFIED_PHASE6_RUN"


def test_simulate_endpoint():
    """Verify POST /simulate runs full Newmark physics solver on demand."""
    payload = {
        "building": {
            "num_storeys": 3,
            "storey_mass_kg": 100000.0,
            "storey_stiffness_n_m": 120000000.0,
            "storey_height_m": 3.5,
            "damping_ratio": 0.05,
            "name": "Test_Building_3St",
        },
        "earthquake_name": "El_Centro_1940_NS",
        "scale_factor": 1.0,
    }
    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["true_max_pidr"] > 0.0
    assert data["true_peak_base_shear_kn"] > 0.0
    assert "time_history_preview" in data
    assert len(data["storey_drift_profile"]) == 3


def test_is1893_compare_endpoint():
    """Verify POST /is1893/compare generates 3-way engineering comparison."""
    payload = {
        "building": {
            "num_storeys": 4,
            "storey_mass_kg": 110000.0,
            "storey_stiffness_n_m": 130000000.0,
            "storey_height_m": 3.5,
            "damping_ratio": 0.05,
            "name": "Test_Building_4St",
        },
        "earthquake_name": "Chamoli_1999_Gopeshwar",
        "zone": "Zone IV",
        "soil_type": "Type II (Medium)",
        "model_name": "GradientBoosting",
    }
    resp = client.post("/is1893/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert len(data["comparison_table"]) >= 3
    assert data["speedup_factor"] > 0.0
