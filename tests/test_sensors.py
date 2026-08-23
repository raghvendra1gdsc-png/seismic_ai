"""Tests for real-time sensor streaming, STA/LTA onset picking, and early feature extraction."""

import pytest
import numpy as np

from src.earthquake.database import GroundMotionDatabase
from src.structural.building import ShearBuilding
from src.sensors.stream import SensorStream, AccelerationSample
from src.sensors.replay import ReplaySensor
from src.sensors.live import LiveSensor
from src.sensors.detector import STA_LTA_Detector, detect_sta_lta_onset
from src.sensors.early_features import (
    extract_early_wave_features,
    estimate_surrogate_features_from_early_onset,
)


def test_replay_sensor_streaming():
    """Verify ReplaySensor correctly yields acceleration samples sequentially."""
    db = GroundMotionDatabase()
    rec = db.get_record("El_Centro_1940_NS")
    sensor = ReplaySensor(record=rec, realtime_delay=False)

    samples = list(sensor.stream_samples())
    assert len(samples) == rec.num_points
    assert samples[0].timestamp_s == 0.0
    assert np.isclose(samples[0].acceleration_g, rec.acceleration_g[0])

    meta = sensor.get_metadata()
    assert meta["sensor_type"] == "ReplaySensor"
    assert meta["record_name"] == "El_Centro_1940_NS"


def test_live_sensor_stub():
    """Verify LiveSensor clean stub interface behavior."""
    live = LiveSensor(sensor_id="LIVE_TEST_01")
    assert not live.is_connected
    assert not live.connect()
    meta = live.get_metadata()
    assert meta["hardware_ready"] is False

    with pytest.raises(RuntimeError):
        list(live.stream_samples())


def test_sta_lta_detector_onset():
    """Verify STA/LTA detector picks P-wave onset from synthetic step signal."""
    # Synthetic signal: 2s quiet noise, followed by 0.5g shock
    dt = 0.01
    t = np.arange(0, 5.0, dt)
    acc = np.zeros_like(t)
    acc[200:] = 0.5 * np.sin(2.0 * np.pi * 5.0 * t[200:])  # Step onset at t=2.0s

    res = detect_sta_lta_onset(acc, dt=dt, sta_s=0.2, lta_s=2.0, trigger_ratio=3.0)
    assert res["onset_detected"] is True
    assert res["onset_time_s"] is not None
    # Onset time should be around 2.0s (+/- 0.2s)
    assert 1.9 <= res["onset_time_s"] <= 2.3


def test_early_wave_feature_extraction():
    """Verify early feature extraction from a post-onset window."""
    dt = 0.01
    t = np.arange(0, 3.0, dt)
    acc_window = 0.25 * np.sin(2.0 * np.pi * 3.0 * t)  # 0.25g, 3Hz harmonic wave

    feats = extract_early_wave_features(acc_window, dt=dt, window_duration_s=3.0)
    assert np.isclose(feats["pga_early_g"], 0.25, atol=1e-2)
    assert feats["pgv_early_m_s"] > 0.0
    assert 0.1 <= feats["tau_c_s"] <= 4.0


def test_estimate_surrogate_features_from_early_onset():
    """Verify mapping from early features to surrogate feature vector format."""
    bldg = ShearBuilding.from_uniform(5, 120000.0, 150.0e6, 3.5)
    early_feats = {
        "pga_early_g": 0.20,
        "pgv_early_m_s": 0.15,
        "pgd_early_m": 0.03,
        "tau_c_s": 0.8,
        "early_arias_m_s": 0.4,
    }

    surr_feats = estimate_surrogate_features_from_early_onset(bldg, early_feats)

    # Check that required keys are present
    required_keys = [
        "num_storeys",
        "total_mass_kg",
        "fundamental_period_T1_s",
        "pga_g",
        "Sa_T1_g",
        "static_drift_proxy",
    ]
    for k in required_keys:
        assert k in surr_feats
        assert surr_feats[k] > 0.0
