"""Early-signal feature extraction and rapid structural response estimation.

Extracts seismological early-warning parameters (early PGA, early PGV, tau_c, P_d)
from the first 2-3 seconds following P-wave onset, and approximates surrogate input features
for immediate structural response estimation before peak S-wave arrival.
"""

from typing import Dict, Any, Optional
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.modal import ModalAnalysis
from src.earthquake.record import GroundMotionRecord, GRAVITY


def extract_early_wave_features(
    acceleration_g_window: np.ndarray,
    dt: float = 0.01,
    window_duration_s: float = 3.0,
) -> Dict[str, float]:
    """Extract early-signal seismological intensity parameters from a post-onset time window.

    Parameters
    ----------
    acceleration_g_window : np.ndarray
        Ground acceleration samples (in g) in the post-onset window (e.g. 2-3 s).
    dt : float, default=0.01
        Sampling interval in seconds.
    window_duration_s : float, default=3.0
        Duration of window in seconds.

    Returns
    -------
    Dict[str, float]
        Dictionary containing early PGA, early PGV, early PGD, tau_c (estimated period), and Arias intensity.
    """
    acc_g = np.asarray(acceleration_g_window, dtype=np.float64).flatten()
    if acc_g.size == 0:
        return {
            "pga_early_g": 0.0,
            "pgv_early_m_s": 0.0,
            "pgd_early_m": 0.0,
            "tau_c_s": 0.5,
            "early_arias_m_s": 0.0,
        }

    acc_ms2 = acc_g * GRAVITY
    n_pts = acc_ms2.size

    # Numerical integration for velocity and displacement
    vel_ms = np.cumsum(acc_ms2) * dt
    vel_ms = vel_ms - np.mean(vel_ms)  # Remove linear drift
    disp_m = np.cumsum(vel_ms) * dt

    pga_early = float(np.max(np.abs(acc_g)))
    pgv_early = float(np.max(np.abs(vel_ms)))
    pgd_early = float(np.max(np.abs(disp_m)))

    # Early Arias Intensity: Ia = (pi / 2g) * integral(a^2 dt)
    arias_early = float((np.pi / (2.0 * GRAVITY)) * np.sum(acc_ms2 ** 2) * dt)

    # Predominant Period tau_c = 2*pi / sqrt( r / d ) where r = int(v^2 dt), d = int(u^2 dt)
    # Reference: Kanamori (2005) Real-Time Seismology and Earthquake Early Warning
    r_val = float(np.sum(vel_ms ** 2) * dt)
    d_val = float(np.sum(disp_m ** 2) * dt)
    if d_val > 1e-12 and r_val > 1e-12:
        tau_c = float(2.0 * np.pi / np.sqrt(r_val / d_val))
        tau_c = np.clip(tau_c, 0.1, 4.0)
    else:
        tau_c = 0.5

    return {
        "pga_early_g": pga_early,
        "pgv_early_m_s": pgv_early,
        "pgd_early_m": pgd_early,
        "tau_c_s": tau_c,
        "early_arias_m_s": arias_early,
    }


def estimate_surrogate_features_from_early_onset(
    building: ShearBuilding,
    early_features: Dict[str, float],
    damping_ratio: float = 0.05,
    modal: Optional[ModalAnalysis] = None,
) -> Dict[str, float]:
    """Map early onset signal features and building parameters to the surrogate feature format.

    Uses physics-based scaling: for early response estimation, the full-motion PGA is projected
    from early $PGA_p$ and $\tau_c$ scaling ratios, allowing the pre-trained surrogate model to
    predict peak building demand (PIDR, Base Shear) instantaneously.

    Parameters
    ----------
    building : ShearBuilding
        Target structural building model.
    early_features : Dict[str, float]
        Features extracted from post-onset early window.
    damping_ratio : float, default=0.05
        Damping ratio.
    modal : Optional[ModalAnalysis]
        Modal properties (computed if None).

    Returns
    -------
    Dict[str, float]
        Dictionary matching the standard feature schema for surrogate model inference.
    """
    if modal is None:
        modal = ModalAnalysis(building)

    t1 = float(modal.fundamental_period)
    t2 = float(modal.periods[1]) if modal.num_modes > 1 else t1
    t3 = float(modal.periods[2]) if modal.num_modes > 2 else t2

    pga_p = max(early_features.get("pga_early_g", 0.05), 1e-4)
    pgv_p = max(early_features.get("pgv_early_m_s", 0.02), 1e-4)
    pgd_p = max(early_features.get("pgd_early_m", 0.005), 1e-5)
    tau_c = early_features.get("tau_c_s", 0.5)
    ia_early = early_features.get("early_arias_m_s", 0.1)

    # Standard empirical scaling: S-wave peak is typically ~2.0x - 3.5x early P-wave amplitude
    estimated_pga = pga_p * 2.2
    estimated_pgv = pgv_p * 2.5
    estimated_pgd = pgd_p * 2.5
    estimated_ia = ia_early * 4.0
    estimated_duration = max(15.0, tau_c * 20.0)

    # Spectral amplification factor beta(T1) ~ 2.0 to 2.5 near resonance (T1 ~ tau_c)
    tuning_ratio = t1 / max(tau_c, 0.1)
    amp_factor = 1.0 + 1.5 * np.exp(-0.5 * ((tuning_ratio - 1.0) / 0.6) ** 2)

    sa_t1_g = estimated_pga * amp_factor
    sa_t2_g = estimated_pga * (1.0 + 1.2 * np.exp(-0.5 * ((t2 / max(tau_c, 0.1) - 1.0) / 0.6) ** 2))
    sa_t3_g = estimated_pga * 1.2
    sd_t1_m = (sa_t1_g * GRAVITY) / ((2.0 * np.pi / max(t1, 0.05)) ** 2)

    k_arr = building.stiffnesses
    m_arr = building.masses
    h_arr = building.heights
    n_storeys = building.num_storeys

    avg_k = float(np.mean(k_arr))
    avg_m = float(np.mean(m_arr))
    base_k = float(k_arr[0])
    top_k = float(k_arr[-1])
    taper_ratio = top_k / base_k

    v_st_est = modal.effective_masses[0] * (sa_t1_g * GRAVITY)
    drift_proxy = v_st_est / (base_k * h_arr[0])

    return {
        "num_storeys": float(n_storeys),
        "total_mass_kg": building.total_mass,
        "total_height_m": building.total_height,
        "fundamental_period_T1_s": t1,
        "period_T2_s": t2,
        "period_T3_s": t3,
        "period_ratio_T2_T1": t2 / t1,
        "first_mode_mass_ratio": float(modal.effective_mass_ratios[0]),
        "first_mode_participation": float(modal.participation_factors[0]),
        "avg_storey_mass_kg": avg_m,
        "avg_storey_stiffness_N_m": avg_k,
        "base_stiffness_N_m": base_k,
        "stiffness_taper_ratio": taper_ratio,
        "damping_ratio": damping_ratio,
        "pga_g": estimated_pga,
        "pgv_m_s": estimated_pgv,
        "pgd_m": estimated_pgd,
        "arias_intensity_m_s": estimated_ia,
        "significant_duration_s": estimated_duration,
        "mean_period_Tm_s": tau_c,
        "predominant_period_Tp_s": tau_c,
        "Sa_T1_g": sa_t1_g,
        "Sa_T2_g": sa_t2_g,
        "Sa_T3_g": sa_t3_g,
        "Sd_T1_m": sd_t1_m,
        "Sa_over_PGA": sa_t1_g / max(estimated_pga, 1e-4),
        "T1_over_Tp": t1 / max(tau_c, 1e-3),
        "T1_over_Tm": t1 / max(tau_c, 1e-3),
        "static_drift_proxy": drift_proxy,
    }
