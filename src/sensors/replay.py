"""Historical ground motion replay sensor.

Replays actual recorded historical ground motions (PEER / Indian PESMOS)
sample-by-sample at real or accelerated playback speeds for testing and demonstration.
"""

from typing import Iterator, Dict, Any, Optional
import time
import numpy as np

from src.earthquake.record import GroundMotionRecord, GRAVITY
from src.sensors.stream import SensorStream, AccelerationSample


class ReplaySensor(SensorStream):
    """Streams a real historical earthquake record sample-by-sample.

    Parameters
    ----------
    record : GroundMotionRecord
        Historical ground motion record to replay.
    sensor_id : str, default='REPLAY_SENSOR_01'
        Sensor identifier.
    realtime_delay : bool, default=False
        If True, sleeps between samples to simulate physical real-time streaming.
        If False, yields instantaneously as a generator.
    playback_speed : float, default=1.0
        Speed multiplier if realtime_delay is True (e.g. 1.0 = real time, 5.0 = 5x speed).
    """

    def __init__(
        self,
        record: GroundMotionRecord,
        sensor_id: str = "REPLAY_SENSOR_01",
        realtime_delay: bool = False,
        playback_speed: float = 1.0,
    ) -> None:
        if not isinstance(record, GroundMotionRecord):
            raise TypeError(f"Expected GroundMotionRecord, got {type(record).__name__}")
        sampling_rate = 1.0 / record.dt
        super().__init__(sensor_id=sensor_id, sampling_rate_hz=sampling_rate)
        self.record = record
        self.realtime_delay = bool(realtime_delay)
        self.playback_speed = max(0.1, float(playback_speed))
        self._current_index: int = 0

    def stream_samples(self) -> Iterator[AccelerationSample]:
        """Yield acceleration samples from the record."""
        acc_ms2 = self.record.acceleration
        acc_g = self.record.acceleration_g
        dt = self.record.dt
        delay = (dt / self.playback_speed) if self.realtime_delay else 0.0

        for i in range(len(acc_ms2)):
            self._current_index = i
            t_curr = i * dt
            sample = AccelerationSample(
                timestamp_s=t_curr,
                acceleration_g=float(acc_g[i]),
                acceleration_ms2=float(acc_ms2[i]),
                sample_index=i,
                sensor_id=self.sensor_id,
            )

            if delay > 0:
                time.sleep(delay)

            yield sample

    def reset(self) -> None:
        """Reset replay position to index 0."""
        self._current_index = 0

    def get_metadata(self) -> Dict[str, Any]:
        """Metadata for replay stream."""
        return {
            "sensor_type": "ReplaySensor",
            "sensor_id": self.sensor_id,
            "record_name": self.record.name,
            "sampling_rate_hz": self.sampling_rate_hz,
            "dt_s": self.dt,
            "duration_s": self.record.duration,
            "num_samples": self.record.num_points,
            "pga_g": self.record.pga_g,
            "realtime_delay": self.realtime_delay,
            "playback_speed": self.playback_speed,
            "earthquake_metadata": self.record.metadata,
        }
