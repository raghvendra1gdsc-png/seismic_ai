"""Unit tests for Sensor Hardware Abstraction Layer (HAL) and Park-Ang damage index."""

import pytest
import numpy as np

from src.sensors.hal import (
    SerialAccelerometerDriver,
    MQTTNetworkSensorDriver,
    DigitalSignalConditioner,
    RawSensorReading,
)
from src.sensors.damage_index import ParkAngDamageEvaluator, DamageEvaluationResult


def test_serial_accelerometer_driver():
    """Test serial accelerometer driver mock injection and sampling."""
    driver = SerialAccelerometerDriver(sensor_id="TEST_ADXL355", sampling_rate_hz=100.0)
    assert not driver.is_connected

    assert driver.connect()
    assert driver.is_connected

    # Push 5 mock readings
    for i in range(5):
        driver.push_mock_reading(ax=0.05 * (i + 1), ay=0.01, az=1.0)

    samples = driver.read_samples(max_samples=5)
    assert len(samples) == 5
    assert samples[0].ax == 0.05
    assert samples[-1].ax == 0.25

    meta = driver.get_metadata()
    assert meta["sensor_id"] == "TEST_ADXL355"
    assert meta["sampling_rate_hz"] == 100.0

    driver.disconnect()
    assert not driver.is_connected


def test_mqtt_network_sensor_driver():
    """Test MQTT IoT sensor packet ingestion."""
    driver = MQTTNetworkSensorDriver(sensor_id="TEST_RASP_SHAKE")
    driver.connect()

    packet = {"timestamp": 12345.67, "ax_g": 0.12, "ay_g": 0.04, "az_g": 0.98}
    driver.ingest_mqtt_packet(packet)

    samples = driver.read_samples(max_samples=1)
    assert len(samples) == 1
    assert samples[0].ax == 0.12
    assert samples[0].sensor_id == "TEST_RASP_SHAKE"


def test_digital_signal_conditioner():
    """Test baseline detrending and digital filtering."""
    conditioner = DigitalSignalConditioner(sampling_rate_hz=100.0, low_cut_hz=0.1)
    # Signal with constant DC offset + sinusoid
    t = np.linspace(0, 2.0, 200)
    raw = 0.5 + 0.1 * np.sin(2.0 * np.pi * 2.0 * t)  # 0.5 DC offset
    filtered = conditioner.filter_stream(raw)

    assert len(filtered) == len(raw)
    # Mean of filtered signal should be close to 0 (DC removed)
    assert abs(np.mean(filtered)) < 0.05


def test_park_ang_damage_index():
    """Test Park-Ang damage index evaluation across limit states."""
    evaluator = ParkAngDamageEvaluator(beta_pa=0.08, ultimate_ductility_capacity=6.0)

    # 1. Elastic/Minor case (u_m = 0.5 * u_y, small energy)
    res_minor = evaluator.evaluate_storey(
        max_drift_m=0.005,
        yield_disp_m=0.010,
        yield_force_n=1e6,
        hysteretic_energy_j=100.0,
    )
    assert res_minor.park_ang_damage_index < 0.20
    assert res_minor.is_safe_for_occupancy
    assert res_minor.is_repairable

    # 2. Severe/Yielded case (u_m = 4.0 * u_y, high energy)
    res_severe = evaluator.evaluate_storey(
        max_drift_m=0.040,
        yield_disp_m=0.010,
        yield_force_n=1e6,
        hysteretic_energy_j=2e5,
    )
    assert res_severe.park_ang_damage_index > 0.50
    assert not res_severe.is_safe_for_occupancy

    # 3. Multi-storey building envelope test
    envelope = evaluator.evaluate_building_envelope(
        storey_drifts_m=[0.005, 0.008, 0.006],
        storey_yield_disps_m=[0.010, 0.010, 0.010],
        storey_yield_forces_n=[1e6, 1e6, 1e6],
        storey_hysteretic_energies_j=[500.0, 800.0, 400.0],
    )
    assert "max_storey_damage_index" in envelope
    assert envelope["global_safety_status"] == "SAFE"
