"""Real-time sensor streaming, Hardware Abstraction Layer (HAL), STA/LTA onset picking, and rapid damage estimation."""

from src.sensors.stream import SensorStream, AccelerationSample, OnsetTriggerEvent
from src.sensors.replay import ReplaySensor
from src.sensors.live import LiveSensor
from src.sensors.detector import STA_LTA_Detector, detect_sta_lta_onset
from src.sensors.early_features import (
    extract_early_wave_features,
    estimate_surrogate_features_from_early_onset,
)
from src.sensors.hal import (
    BaseSensorDriver,
    RawSensorReading,
    SerialAccelerometerDriver,
    MQTTNetworkSensorDriver,
    DigitalSignalConditioner,
)
from src.sensors.damage_index import (
    ParkAngDamageEvaluator,
    DamageEvaluationResult,
)

__all__ = [
    "SensorStream",
    "AccelerationSample",
    "OnsetTriggerEvent",
    "ReplaySensor",
    "LiveSensor",
    "STA_LTA_Detector",
    "detect_sta_lta_onset",
    "extract_early_wave_features",
    "estimate_surrogate_features_from_early_onset",
    "BaseSensorDriver",
    "RawSensorReading",
    "SerialAccelerometerDriver",
    "MQTTNetworkSensorDriver",
    "DigitalSignalConditioner",
    "ParkAngDamageEvaluator",
    "DamageEvaluationResult",
]
