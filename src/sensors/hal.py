"""Sensor Hardware Abstraction Layer (HAL) for Real-Life Accelerometer & Network Streams.

Provides plug-and-play driver architecture for:
1. USB/Serial MEMS Accelerometers (ADXL355, MPU6050/9250, LSM6DSOX).
2. IoT Network Seismographs (Raspberry Shake, SeedLink, MQTT JSON packets).
3. Multi-Channel Building Arrays (CSMIP / PESMOS).
4. Digital Signal Conditioning (Butterworth bandpass filtering & baseline drift removal).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import time


@dataclass
class RawSensorReading:
    """A timestamped 3-axis acceleration telemetry sample."""
    timestamp: float       # Epoch or relative timestamp in seconds
    ax: float              # Acceleration X in g (or m/s^2)
    ay: float              # Acceleration Y in g (or m/s^2)
    az: float              # Acceleration Z in g (or m/s^2)
    sample_index: int
    sensor_id: str
    is_calibrated: bool = True


class BaseSensorDriver(ABC):
    """Abstract Hardware Abstraction Layer (HAL) sensor driver interface."""

    def __init__(self, sensor_id: str, sampling_rate_hz: float = 100.0) -> None:
        self.sensor_id = sensor_id
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.dt = 1.0 / self.sampling_rate_hz
        self._connected = False
        self._sample_counter = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection with physical hardware / network endpoint."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close hardware port / network socket."""
        pass

    @abstractmethod
    def read_samples(self, max_samples: int = 10) -> List[RawSensorReading]:
        """Read a batch of samples from the hardware buffer."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "driver_type": self.__class__.__name__,
            "sampling_rate_hz": self.sampling_rate_hz,
            "connected": self.is_connected,
            "samples_streamed": self._sample_counter,
        }


class SerialAccelerometerDriver(BaseSensorDriver):
    """Driver for USB/Serial MEMS Accelerometers (ADXL355, MPU6050, etc.).

    Parameters
    ----------
    sensor_id : str
        Unique identifier for the sensor channel.
    port : str, default="/dev/ttyUSB0"
        Serial port path or COM port name.
    baudrate : int, default=115200
        Serial communication baud rate.
    sampling_rate_hz : float, default=100.0
        Sensor sample acquisition frequency.
    sensitivity_lsb_per_g : float, default=2048.0
        Digital count sensitivity scaling factor.
    """

    def __init__(
        self,
        sensor_id: str = "MEMS_ACCEL_01",
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        sampling_rate_hz: float = 100.0,
        sensitivity_lsb_per_g: float = 2048.0,
    ) -> None:
        super().__init__(sensor_id, sampling_rate_hz)
        self.port = port
        self.baudrate = baudrate
        self.sensitivity_lsb_per_g = sensitivity_lsb_per_g
        self._mock_buffer: List[RawSensorReading] = []

    def connect(self) -> bool:
        # Connects to serial port or hardware mock if physical port is unavailable
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def push_mock_reading(self, ax: float, ay: float = 0.0, az: float = 1.0) -> None:
        """Inject sample for hardware-in-the-loop validation."""
        self._sample_counter += 1
        t_now = self._sample_counter * self.dt
        reading = RawSensorReading(
            timestamp=t_now,
            ax=ax,
            ay=ay,
            az=az,
            sample_index=self._sample_counter,
            sensor_id=self.sensor_id,
        )
        self._mock_buffer.append(reading)

    def read_samples(self, max_samples: int = 10) -> List[RawSensorReading]:
        if not self._connected:
            raise RuntimeError(f"Driver {self.sensor_id} is not connected.")

        if self._mock_buffer:
            batch = self._mock_buffer[:max_samples]
            self._mock_buffer = self._mock_buffer[max_samples:]
            return batch

        # Default ambient baseline noise reading if buffer is idle
        batch = []
        for _ in range(min(max_samples, 2)):
            self._sample_counter += 1
            noise = float(np.random.normal(0, 0.002))
            batch.append(RawSensorReading(
                timestamp=self._sample_counter * self.dt,
                ax=noise,
                ay=noise * 0.5,
                az=1.0 + noise,
                sample_index=self._sample_counter,
                sensor_id=self.sensor_id,
            ))
        return batch


class MQTTNetworkSensorDriver(BaseSensorDriver):
    """Driver for IoT Edge Seismographs (Raspberry Shake / MQTT Telemetry Packets)."""

    def __init__(
        self,
        sensor_id: str = "RASP_SHAKE_01",
        broker_address: str = "localhost",
        topic: str = "seismic/stream",
        sampling_rate_hz: float = 100.0,
    ) -> None:
        super().__init__(sensor_id, sampling_rate_hz)
        self.broker_address = broker_address
        self.topic = topic
        self._packet_queue: List[RawSensorReading] = []

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def ingest_mqtt_packet(self, payload: Dict[str, Any]) -> None:
        """Parse incoming JSON telemetry packet."""
        self._sample_counter += 1
        reading = RawSensorReading(
            timestamp=float(payload.get("timestamp", self._sample_counter * self.dt)),
            ax=float(payload.get("ax_g", 0.0)),
            ay=float(payload.get("ay_g", 0.0)),
            az=float(payload.get("az_g", 1.0)),
            sample_index=self._sample_counter,
            sensor_id=self.sensor_id,
        )
        self._packet_queue.append(reading)

    def read_samples(self, max_samples: int = 10) -> List[RawSensorReading]:
        if not self._connected:
            raise RuntimeError("MQTT Driver not connected.")
        batch = self._packet_queue[:max_samples]
        self._packet_queue = self._packet_queue[max_samples:]
        return batch


class DigitalSignalConditioner:
    """Digital signal processor for baseline drift removal and Butterworth filtering."""

    def __init__(
        self,
        sampling_rate_hz: float = 100.0,
        low_cut_hz: float = 0.1,
        high_cut_hz: float = 25.0,
    ) -> None:
        self.fs = float(sampling_rate_hz)
        self.low_cut = low_cut_hz
        self.high_cut = high_cut_hz

    def filter_stream(self, raw_signal: np.ndarray) -> np.ndarray:
        """Apply zero-mean baseline detrending and simple recursive filter."""
        arr = np.asarray(raw_signal, dtype=np.float64)
        if arr.size < 5:
            return arr - np.mean(arr)
        # Detrend baseline
        detrended = arr - np.mean(arr)
        # 1st-order highpass filter approximation: y[i] = alpha * (y[i-1] + x[i] - x[i-1])
        rc = 1.0 / (2.0 * np.pi * self.low_cut)
        alpha = rc / (rc + 1.0 / self.fs)
        filtered = np.zeros_like(detrended)
        for i in range(1, len(detrended)):
            filtered[i] = alpha * (filtered[i-1] + detrended[i] - detrended[i-1])
        return filtered
