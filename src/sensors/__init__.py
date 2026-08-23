"""Real-time sensor streaming, STA/LTA onset picking, and rapid structural response estimation."""

from src.sensors.stream import SensorStream, AccelerationSample, OnsetTriggerEvent
from src.sensors.replay import ReplaySensor
from src.sensors.live import LiveSensor
from src.sensors.detector import STA_LTA_Detector, detect_sta_lta_onset
from src.sensors.early_features import (
    extract_early_wave_features,
    estimate_surrogate_features_from_early_onset,
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
]
