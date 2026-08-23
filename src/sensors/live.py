"""Hardware live sensor interface stub.

Provides a clean, typed stub for physical hardware accelerometer integration
(e.g., Raspberry Shake, ADXL355 MEMS accelerometer, Serial/I2C streaming).

NOTE: Hardware integration is a future extension when physical instrumentation
is physically connected. This class establishes the contract without false claims.
"""

from typing import Iterator, Dict, Any, Optional
import numpy as np

from src.sensors.stream import SensorStream, AccelerationSample


class LiveSensor(SensorStream):
    """Stub interface for physical hardware accelerometer streaming.

    Parameters
    ----------
    sensor_id : str, default='LIVE_HARDWARE_STUB'
        Hardware sensor identifier.
    port : Optional[str]
        Serial or network port identifier (e.g. '/dev/ttyUSB0' or 'tcp://192.168.1.100:5000').
    sampling_rate_hz : float, default=100.0
        Hardware acquisition rate in Hz.
    """

    def __init__(
        self,
        sensor_id: str = "LIVE_HARDWARE_STUB",
        port: Optional[str] = None,
        sampling_rate_hz: float = 100.0,
    ) -> None:
        super().__init__(sensor_id=sensor_id, sampling_rate_hz=sampling_rate_hz)
        self.port = port
        self.is_connected: bool = False

    def connect(self) -> bool:
        """Connect to hardware device.

        Returns False in stub mode with informative message.
        """
        # Hardware stub - returns False until physical hardware driver is mounted
        self.is_connected = False
        return False

    def stream_samples(self) -> Iterator[AccelerationSample]:
        """Yield live samples from hardware. Raises RuntimeError if hardware is not connected."""
        if not self.is_connected:
            raise RuntimeError(
                "LiveSensor hardware is currently in stub mode. Physical accelerometer hardware "
                "is not connected. Use ReplaySensor to stream historical earthquake records."
            )
        # Yield stub loop if connected
        yield from ()

    def reset(self) -> None:
        """Reset hardware buffers."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return hardware status metadata."""
        return {
            "sensor_type": "LiveSensor (Hardware Stub)",
            "sensor_id": self.sensor_id,
            "port": self.port,
            "sampling_rate_hz": self.sampling_rate_hz,
            "is_connected": self.is_connected,
            "hardware_ready": False,
            "notice": "Hardware integration stub ready for physical instrumentation.",
        }
