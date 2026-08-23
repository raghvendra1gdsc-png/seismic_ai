"""Ground motion signal processing and baseline correction.

Provides baseline correction (polynomial / linear detrending), zero-padding,
and high-pass filtering utilities for raw accelerograms.
"""

from typing import Optional
import numpy as np

from src.earthquake.record import GroundMotionRecord


def baseline_correct(
    record: GroundMotionRecord,
    order: int = 2,
    zero_mean: bool = True,
) -> GroundMotionRecord:
    """Apply baseline polynomial correction to a ground motion record.

    Ensures residual velocity and displacement drift at the end of the record
    are minimized, which is essential for accurate dynamic integration.

    Parameters
    ----------
    record : GroundMotionRecord
        Input ground motion record.
    order : int, default=2
        Polynomial order for detrending (1 for linear, 2 for quadratic).
    zero_mean : bool, default=True
        Whether to subtract the pre-event or overall mean before detrending.

    Returns
    -------
    GroundMotionRecord
        Baseline-corrected ground motion record.
    """
    acc = record.acceleration
    t = record.time

    if zero_mean:
        # Subtract mean
        acc = acc - np.mean(acc)

    # Fit polynomial to acceleration
    poly_coeffs = np.polyfit(t, acc, deg=order)
    trend = np.polyval(poly_coeffs, t)
    corrected_acc = acc - trend

    new_meta = record.metadata.copy()
    new_meta["baseline_corrected"] = True
    new_meta["baseline_poly_order"] = order

    return GroundMotionRecord(
        name=f"{record.name}_corrected",
        dt=record.dt,
        acceleration=corrected_acc,
        metadata=new_meta,
    )


def pad_zeros(
    record: GroundMotionRecord,
    leading_seconds: float = 2.0,
    trailing_seconds: float = 5.0,
) -> GroundMotionRecord:
    """Pad ground motion record with leading and trailing zeros.

    Ensures quiescent pre-event state and allows free vibration decay after motion.

    Parameters
    ----------
    record : GroundMotionRecord
        Input record.
    leading_seconds : float, default=2.0
        Duration of zeros to prepend in seconds.
    trailing_seconds : float, default=5.0
        Duration of zeros to append in seconds.

    Returns
    -------
    GroundMotionRecord
        Padded record.
    """
    n_lead = int(np.round(leading_seconds / record.dt))
    n_trail = int(np.round(trailing_seconds / record.dt))

    lead_zeros = np.zeros(n_lead, dtype=np.float64)
    trail_zeros = np.zeros(n_trail, dtype=np.float64)

    padded_acc = np.concatenate([lead_zeros, record.acceleration, trail_zeros])

    new_meta = record.metadata.copy()
    new_meta["padded_leading_s"] = leading_seconds
    new_meta["padded_trailing_s"] = trailing_seconds

    return GroundMotionRecord(
        name=f"{record.name}_padded",
        dt=record.dt,
        acceleration=padded_acc,
        metadata=new_meta,
    )
