"""Physics-informed feature engineering and EDP extraction.

Combines structural dynamics modal properties, ground-motion intensity measures,
and spectral response quantities into structured feature vectors for machine learning.
"""

from typing import Dict, Any, Optional
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.modal import ModalAnalysis
from src.dynamics.damping import RayleighDamping
from src.dynamics.response import DynamicResponse
from src.earthquake.record import GroundMotionRecord, GRAVITY
from src.earthquake.spectra import ResponseSpectrum


def extract_features(
    building: ShearBuilding,
    record: GroundMotionRecord,
    spectrum: Optional[ResponseSpectrum] = None,
    damping_ratio: float = 0.05,
    modal: Optional[ModalAnalysis] = None,
) -> Dict[str, float]:
    """Extract physics-informed features from building and ground motion record.

    Parameters
    ----------
    building : ShearBuilding
        Structural building model.
    record : GroundMotionRecord
        Earthquake ground motion record.
    spectrum : Optional[ResponseSpectrum], optional
        Pre-computed elastic response spectrum (computed if None).
    damping_ratio : float, default=0.05
        Damping ratio used for modal properties and spectrum.
    modal : Optional[ModalAnalysis], optional
        Pre-computed modal analysis (computed if None).

    Returns
    -------
    Dict[str, float]
        Dictionary of numerical features.
    """
    if modal is None:
        modal = ModalAnalysis(building)

    if spectrum is None:
        spectrum = ResponseSpectrum(record=record, damping_ratio=damping_ratio)

    t1 = float(modal.fundamental_period)
    t2 = float(modal.periods[1]) if modal.num_modes > 1 else t1
    t3 = float(modal.periods[2]) if modal.num_modes > 2 else t2

    # Spectral quantities
    sa_t1_g = spectrum.get_sa_g(t1)
    sa_t2_g = spectrum.get_sa_g(t2)
    sa_t3_g = spectrum.get_sa_g(t3)
    sd_t1_m = spectrum.get_sd(t1)

    # Building stiffness and mass summary
    k_arr = building.stiffnesses
    m_arr = building.masses
    h_arr = building.heights
    n_storeys = building.num_storeys

    avg_k = float(np.mean(k_arr))
    avg_m = float(np.mean(m_arr))
    base_k = float(k_arr[0])
    top_k = float(k_arr[-1])
    taper_ratio = top_k / base_k

    # Intensity measures
    ims = record.intensity_measures()
    pga_g = ims["pga_g"]
    pgv_m_s = ims["pgv_m_s"]
    pgd_m = ims["pgd_m"]
    ia = ims["arias_intensity_m_s"]
    d5_95 = ims["significant_duration_5_95_s"]
    tm = ims["mean_period_s"]
    tp = ims["predominant_period_s"]

    # Physics-based dimensionless ratios
    sa_pga_ratio = (sa_t1_g / pga_g) if pga_g > 1e-6 else 1.0
    period_ratio_t1_tp = (t1 / tp) if tp > 1e-4 else 1.0
    period_ratio_t1_tm = (t1 / tm) if tm > 1e-4 else 1.0

    # Static equivalent base shear estimate: V_st = m_eff_1 * Sa(T1)
    v_st_est = modal.effective_masses[0] * (sa_t1_g * GRAVITY)
    drift_proxy = v_st_est / (base_k * h_arr[0])

    features = {
        # Building Geometric & Dynamic Features
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
        # Earthquake Intensity Measures (IMs)
        "pga_g": pga_g,
        "pgv_m_s": pgv_m_s,
        "pgd_m": pgd_m,
        "arias_intensity_m_s": ia,
        "significant_duration_s": d5_95,
        "mean_period_Tm_s": tm,
        "predominant_period_Tp_s": tp,
        # Spectral Quantities
        "Sa_T1_g": sa_t1_g,
        "Sa_T2_g": sa_t2_g,
        "Sa_T3_g": sa_t3_g,
        "Sd_T1_m": sd_t1_m,
        # Physics Interaction Features
        "Sa_over_PGA": sa_pga_ratio,
        "T1_over_Tp": period_ratio_t1_tp,
        "T1_over_Tm": period_ratio_t1_tm,
        "static_drift_proxy": drift_proxy,
    }

    return features


def extract_targets(response: DynamicResponse) -> Dict[str, float]:
    """Extract engineering demand parameter (EDP) targets from a simulation response."""
    return {
        "target_max_pidr": response.max_drift_ratio,
        "target_max_roof_disp_m": response.max_roof_displacement,
        "target_peak_base_shear_N": response.peak_base_shear,
        "target_peak_pfa_m_s2": float(np.max(response.peak_floor_accelerations)),
    }
