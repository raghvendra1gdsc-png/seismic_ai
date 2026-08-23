"""STA/LTA (Short-Term Average / Long-Term Average) earthquake onset detector.

Implements the classical recursive Allen / Earle & Shearer STA/LTA energy ratio algorithm
for robust P-wave arrival picking from continuous or streaming accelerogram records.
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np

from src.sensors.stream import AccelerationSample, OnsetTriggerEvent


class STA_LTA_Detector:
    """Recursive Short-Term-Average / Long-Term-Average (STA/LTA) Onset Detector.

    The Characteristic Function CF(t) = a(t)^2 is filtered through short and long
    exponential smoothing windows:
        STA(t) = c_sta * STA(t - dt) + (1 - c_sta) * CF(t)
        LTA(t) = c_lta * LTA(t - dt) + (1 - c_lta) * CF(t)
        Ratio r(t) = STA(t) / (LTA(t) + eps)

    Onset is declared when r(t) crosses trigger_threshold from below.

    Parameters
    ----------
    sta_window_s : float, default=0.5
        Duration of short-term averaging window in seconds (captures onset energy).
    lta_window_s : float, default=5.0
        Duration of long-term averaging window in seconds (captures background noise).
    trigger_threshold : float, default=3.5
        STA/LTA ratio required to trigger onset.
    detrigger_threshold : float, default=1.5
        STA/LTA ratio at which trigger deactivates.
    sampling_rate_hz : float, default=100.0
        Sampling frequency in Hz (1/dt).
    """

    def __init__(
        self,
        sta_window_s: float = 0.5,
        lta_window_s: float = 5.0,
        trigger_threshold: float = 3.5,
        detrigger_threshold: float = 1.5,
        sampling_rate_hz: float = 100.0,
    ) -> None:
        self.sta_window_s = float(sta_window_s)
        self.lta_window_s = float(lta_window_s)
        self.trigger_threshold = float(trigger_threshold)
        self.detrigger_threshold = float(detrigger_threshold)
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.dt = 1.0 / self.sampling_rate_hz

        # Smoothing coefficients: c = exp(-dt / T)
        self.c_sta = np.exp(-self.dt / self.sta_window_s)
        self.c_lta = np.exp(-self.dt / self.lta_window_s)

        # State variables
        self.sta_val: float = 0.0
        self.lta_val: float = 1e-6
        self.is_triggered: bool = False
        self.onset_sample_index: Optional[int] = None
        self.onset_timestamp_s: Optional[float] = None
        self.total_samples_processed: int = 0
        self.history_ratios: List[float] = []

    def process_sample(self, sample: AccelerationSample) -> Tuple[float, Optional[OnsetTriggerEvent]]:
        """Process a single incoming sample and return (current_sta_lta_ratio, trigger_event).

        Parameters
        ----------
        sample : AccelerationSample
            Incoming acceleration sample.

        Returns
        -------
        Tuple[float, Optional[OnsetTriggerEvent]]
            Current STA/LTA ratio, and OnsetTriggerEvent if onset was just triggered (None otherwise).
        """
        self.total_samples_processed += 1
        acc = sample.acceleration_g
        cf = acc * acc  # Energy characteristic function

        # Recursive update
        self.sta_val = self.c_sta * self.sta_val + (1.0 - self.c_sta) * cf
        self.lta_val = self.c_lta * self.lta_val + (1.0 - self.c_lta) * cf

        ratio = float(self.sta_val / (self.lta_val + 1e-9))
        self.history_ratios.append(ratio)

        event = None
        # Check trigger condition
        if not self.is_triggered and ratio >= self.trigger_threshold:
            self.is_triggered = True
            self.onset_sample_index = sample.sample_index
            self.onset_timestamp_s = sample.timestamp_s
            event = OnsetTriggerEvent(
                onset_timestamp_s=sample.timestamp_s,
                onset_sample_index=sample.sample_index,
                trigger_ratio=ratio,
                trigger_threshold=self.trigger_threshold,
                estimated_pga_early_g=abs(sample.acceleration_g),
                early_signal_duration_s=0.0,
                sensor_id=sample.sensor_id,
            )

        elif self.is_triggered and ratio < self.detrigger_threshold:
            self.is_triggered = False

        return ratio, event

    def reset(self) -> None:
        """Reset internal detector state."""
        self.sta_val = 0.0
        self.lta_val = 1e-6
        self.is_triggered = False
        self.onset_sample_index = None
        self.onset_timestamp_s = None
        self.total_samples_processed = 0
        self.history_ratios.clear()


def detect_sta_lta_onset(
    acceleration_g: np.ndarray,
    dt: float = 0.01,
    sta_s: float = 0.5,
    lta_s: float = 5.0,
    trigger_ratio: float = 3.5,
) -> Dict[str, Any]:
    """Run STA/LTA onset picking across an entire accelerogram array (vectorized/batch mode).

    Parameters
    ----------
    acceleration_g : np.ndarray
        Array of ground acceleration in units of g.
    dt : float, default=0.01
        Sampling interval in seconds.
    sta_s : float, default=0.5
        Short-term averaging window in seconds.
    lta_s : float, default=5.0
        Long-term averaging window in seconds.
    trigger_ratio : float, default=3.5
        Threshold ratio to declare onset.

    Returns
    -------
    Dict[str, Any]
        Dictionary with onset index, onset time in seconds, ratio series, and trigger status.
    """
    acc = np.asarray(acceleration_g, dtype=np.float64).flatten()
    n_pts = acc.size
    sr = 1.0 / dt

    detector = STA_LTA_Detector(
        sta_window_s=sta_s,
        lta_window_s=lta_s,
        trigger_threshold=trigger_ratio,
        sampling_rate_hz=sr,
    )

    ratios = np.zeros(n_pts, dtype=np.float64)
    onset_idx = None
    onset_time_s = None

    for i in range(n_pts):
        sample = AccelerationSample(
            timestamp_s=i * dt,
            acceleration_g=float(acc[i]),
            acceleration_ms2=float(acc[i] * 9.80665),
            sample_index=i,
        )
        r, ev = detector.process_sample(sample)
        ratios[i] = r
        if ev is not None and onset_idx is None:
            onset_idx = ev.onset_sample_index
            onset_time_s = ev.onset_timestamp_s

    return {
        "onset_detected": (onset_idx is not None),
        "onset_index": onset_idx,
        "onset_time_s": onset_time_s,
        "trigger_threshold": trigger_ratio,
        "sta_lta_ratios": ratios.tolist(),
        "max_ratio": float(np.max(ratios)) if n_pts > 0 else 0.0,
    }
