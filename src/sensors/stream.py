"""Sensor streaming interfaces and data structures.

Defines the abstract SensorStream base class and data packet representations
for real-time acceleration streaming, onset picking, and structural response estimation.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np


@dataclass
class AccelerationSample:
    """Single sample packet from an accelerometer sensor stream."""

    timestamp_s: float
    acceleration_g: float
    acceleration_ms2: float
    sample_index: int
    sensor_id: str = "ACCEL_STREAM_01"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_s": round(self.timestamp_s, 4),
            "acceleration_g": round(self.acceleration_g, 6),
            "acceleration_ms2": round(self.acceleration_ms2, 6),
            "sample_index": self.sample_index,
            "sensor_id": self.sensor_id,
        }


@dataclass
class OnsetTriggerEvent:
    """Event emitted when STA/LTA algorithm triggers on P-wave onset."""

    onset_timestamp_s: float
    onset_sample_index: int
    trigger_ratio: float
    trigger_threshold: float
    estimated_pga_early_g: float
    early_signal_duration_s: float
    sensor_id: str
    message: str = "P-wave onset detected. Initiating immediate structural response estimation."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "onset_timestamp_s": round(self.onset_timestamp_s, 4),
            "onset_sample_index": self.onset_sample_index,
            "trigger_ratio": round(self.trigger_ratio, 2),
            "trigger_threshold": self.trigger_threshold,
            "estimated_pga_early_g": round(self.estimated_pga_early_g, 4),
            "early_signal_duration_s": round(self.early_signal_duration_s, 2),
            "sensor_id": self.sensor_id,
            "message": self.message,
        }


class SensorStream(ABC):
    """Abstract base class for all accelerometer stream sources (replay and live)."""

    def __init__(self, sensor_id: str = "ACCEL_STREAM_01", sampling_rate_hz: float = 100.0) -> None:
        self.sensor_id = str(sensor_id)
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.dt = 1.0 / self.sampling_rate_hz

    @abstractmethod
    def stream_samples(self) -> Iterator[AccelerationSample]:
        """Yield acceleration samples sequentially."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the sensor stream state back to start."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return sensor stream metadata dictionary."""
        raise NotImplementedError
