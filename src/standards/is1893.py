"""Bureau of Indian Standards (BIS) IS 1893 (Part 1): 2016 Seismic Code Provisions.

Implements:
1. Seismic Zone Factors (Z) for Zones II, III, IV, and V.
2. Soil Categories: Type I (Rock / Hard), Type II (Medium), Type III (Soft).
3. Design Acceleration Spectrum Sa/g(T) for 5% damping (Clause 6.4.2).
4. Approximate Fundamental Natural Period Ta (Clause 7.6.2).
5. Design Horizontal Seismic Coefficient Ah = (Z/2) * (I/R) * (Sa/g) (Clause 6.4.2).
6. Equivalent Static Design Base Shear VB = Ah * W (Clause 7.6.1).
7. Vertical Distribution of Lateral Seismic Forces Qi (Clause 7.6.3).
8. Storey Drift Limitation (Clause 7.11.1) - max allowable drift <= 0.004 * hi.
9. 3-Way Engineering Audit: IS 1893 Code vs Physics Solver vs ML Surrogate.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.modal import ModalAnalysis
from src.earthquake.record import GRAVITY


# IS 1893:2016 Table 3 - Seismic Zone Factors
ZONE_FACTORS: Dict[str, float] = {
    "Zone II": 0.10,   # Low seismic intensity
    "Zone III": 0.16,  # Moderate seismic intensity
    "Zone IV": 0.24,   # Severe seismic intensity (e.g. Delhi, NCR, parts of Bihar/HP)
    "Zone V": 0.36,    # Very severe seismic intensity (e.g. Northeast India, Kutch, Himalayas)
}

# IS 1893:2016 Clause 6.4.2 - Soil Types
SOIL_TYPES = ["Type I (Rock/Hard)", "Type II (Medium)", "Type III (Soft)"]

# IS 1893:2016 Table 8 - Response Reduction Factors (R)
RESPONSE_REDUCTION_FACTORS: Dict[str, float] = {
    "SMRF (Special Moment Resisting Frame)": 5.0,
    "OMRF (Ordinary Moment Resisting Frame)": 3.0,
    "Steel Frame with Concentric Braces": 4.0,
    "Steel Frame with Eccentric Braces": 5.0,
    "RC Dual System with Shear Walls": 5.0,
}

# IS 1893:2016 Table 7 - Importance Factors (I)
IMPORTANCE_FACTORS: Dict[str, float] = {
    "Normal Residential / Commercial (I=1.0)": 1.0,
    "Residential / Commercial > 200 occupancy (I=1.2)": 1.2,
    "Critical / Hospital / Emergency / School (I=1.5)": 1.5,
}


def compute_is1893_spectral_shape(period: float, soil_type: str = "Type II (Medium)") -> float:
    """Compute normalized design acceleration coefficient (Sa/g) for 5% damping per IS 1893:2016 Fig 2.

    Parameters
    ----------
    period : float
        Fundamental natural period T in seconds (T > 0).
    soil_type : str
        Soil classification: 'Type I (Rock/Hard)', 'Type II (Medium)', or 'Type III (Soft)'.

    Returns
    -------
    float
        Design spectral acceleration coefficient Sa/g.
    """
    if period <= 0.0:
        raise ValueError(f"Period must be positive, got {period}")

    # Simplify soil type matching
    soil_clean = soil_type.upper()

    if "TYPE III" in soil_clean or "SOFT" in soil_clean:
        # Type III (Soft Soil):
        # 1 + 15*T for 0.00 <= T < 0.10
        # 2.50     for 0.10 <= T <= 0.67
        # 1.67 / T for 0.67 < T <= 4.00
        if period < 0.10:
            return float(1.0 + 15.0 * period)
        elif period <= 0.67:
            return 2.50
        else:
            return float(1.67 / period)

    elif "TYPE II" in soil_clean or "MEDIUM" in soil_clean:
        # Type II (Medium Soil) per IS 1893:2016
        # 1 + 15*T for 0.00 <= T < 0.10
        # 2.50     for 0.10 <= T <= 0.55
        # 1.36 / T for 0.55 < T <= 4.00
        if period < 0.10:
            return float(1.0 + 15.0 * period)
        elif period <= 0.55:
            return 2.50
        else:
            return float(1.36 / period)

    else:
        # Default: Type I (Rock / Hard Soil):
        # 1 + 15*T for 0.00 <= T < 0.10
        # 2.50     for 0.10 <= T <= 0.40
        # 1.00 / T for 0.40 < T <= 4.00
        if period < 0.10:
            return float(1.0 + 15.0 * period)
        elif period <= 0.40:
            return 2.50
        else:
            return float(1.00 / period)


def compute_is1893_approx_period(
    total_height_m: float,
    frame_type: str = "RC Moment Frame without Infill",
    base_dimension_d_m: Optional[float] = None,
) -> float:
    """Compute approximate fundamental natural period Ta per IS 1893:2016 Clause 7.6.2.

    Parameters
    ----------
    total_height_m : float
        Total height of building in meters (h).
    frame_type : str
        Structural system description.
    base_dimension_d_m : Optional[float]
        Base dimension along the direction of shaking in meters (d).

    Returns
    -------
    float
        Approximate period Ta in seconds.
    """
    h = float(total_height_m)
    if h <= 0.0:
        raise ValueError(f"Height must be positive, got {h}")

    if "STEEL" in frame_type.upper():
        # Ta = 0.085 * h^0.75 for steel moment-resisting frames
        return float(0.085 * (h ** 0.75))
    elif "INFILL" in frame_type.upper() or "MASONRY" in frame_type.upper():
        # Ta = 0.09 * h / sqrt(d) for all other buildings with masonry infill
        d = base_dimension_d_m if (base_dimension_d_m and base_dimension_d_m > 0) else max(10.0, h * 0.5)
        return float(0.09 * h / np.sqrt(d))
    else:
        # Default: Ta = 0.075 * h^0.75 for bare RC moment-resisting frames (Clause 7.6.2a)
        return float(0.075 * (h ** 0.75))


@dataclass
class IS1893DesignResult:
    """Structured results container for IS 1893:2016 Equivalent Static Analysis."""

    zone: str
    zone_factor_z: float
    soil_type: str
    importance_factor_i: float
    response_reduction_r: float
    period_used_s: float
    period_source: str
    spectral_acceleration_sa_g: float
    seismic_coefficient_ah: float
    total_seismic_weight_kn: float
    design_base_shear_kn: float
    design_base_shear_n: float
    storey_lateral_forces_n: List[float]
    storey_shear_forces_n: List[float]
    storey_displacements_m: List[float]
    storey_drifts_m: List[float]
    interstorey_drift_ratios: List[float]
    max_interstorey_drift_ratio: float
    allowable_drift_limit: float
    drift_compliance_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone": self.zone,
            "zone_factor_z": self.zone_factor_z,
            "soil_type": self.soil_type,
            "importance_factor_i": self.importance_factor_i,
            "response_reduction_r": self.response_reduction_r,
            "period_used_s": self.period_used_s,
            "period_source": self.period_source,
            "spectral_acceleration_sa_g": self.spectral_acceleration_sa_g,
            "seismic_coefficient_ah": self.seismic_coefficient_ah,
            "total_seismic_weight_kn": self.total_seismic_weight_kn,
            "design_base_shear_kn": self.design_base_shear_kn,
            "design_base_shear_n": self.design_base_shear_n,
            "storey_lateral_forces_n": self.storey_lateral_forces_n,
            "storey_shear_forces_n": self.storey_shear_forces_n,
            "storey_displacements_m": self.storey_displacements_m,
            "storey_drifts_m": self.storey_drifts_m,
            "interstorey_drift_ratios": self.interstorey_drift_ratios,
            "max_interstorey_drift_ratio": self.max_interstorey_drift_ratio,
            "allowable_drift_limit": self.allowable_drift_limit,
            "drift_compliance_status": self.drift_compliance_status,
        }


def analyze_is1893_equivalent_static(
    building: ShearBuilding,
    zone: str = "Zone IV",
    soil_type: str = "Type II (Medium)",
    importance_factor: float = 1.0,
    response_reduction: float = 5.0,
    use_exact_period: bool = True,
) -> IS1893DesignResult:
    """Perform complete IS 1893 (Part 1): 2016 Equivalent Static Lateral Force Analysis.

    Calculates:
    - Ah = (Z / 2) * (I / R) * (Sa / g)
    - VB = Ah * W
    - Qi = VB * (Wi * hi^2) / sum(Wj * hj^2)
    - Elastic Storey Displacements and Drifts under design static loads
    - Drift check against Clause 7.11.1 limit of 0.004 * hi

    Parameters
    ----------
    building : ShearBuilding
        Building model.
    zone : str
        Seismic zone: 'Zone II', 'Zone III', 'Zone IV', or 'Zone V'.
    soil_type : str
        Soil type: 'Type I (Rock/Hard)', 'Type II (Medium)', or 'Type III (Soft)'.
    importance_factor : float
        Importance factor I (default=1.0).
    response_reduction : float
        Response reduction factor R (default=5.0 for SMRF).
    use_exact_period : bool
        If True, uses eigenvalue modal analysis fundamental period T1;
        if False, uses empirical period Ta.

    Returns
    -------
    IS1893DesignResult
        Full design analysis result.
    """
    z_val = ZONE_FACTORS.get(zone, 0.24)
    i_val = float(importance_factor)
    r_val = float(response_reduction)

    modal = ModalAnalysis(building)
    t1_exact = float(modal.fundamental_period)
    t_approx = compute_is1893_approx_period(building.total_height)

    if use_exact_period:
        # Per IS 1893:2016 Clause 7.6.2, dynamic period can be used
        t_design = t1_exact
        source = f"Exact Modal Eigenvalue T1 ({t1_exact:.3f} s)"
    else:
        t_design = t_approx
        source = f"IS 1893 Empirical Ta ({t_approx:.3f} s)"

    # 1. Design Sa/g
    sa_over_g = compute_is1893_spectral_shape(t_design, soil_type)

    # 2. Design Horizontal Seismic Coefficient Ah = (Z/2) * (I/R) * (Sa/g)
    # Clause 6.4.2: Ah shall not be taken less than (Z/2) * (I/R) * (0.24 / 2.5) or minimum limit
    ah = (z_val / 2.0) * (i_val / r_val) * sa_over_g

    # 3. Total Seismic Weight W = sum(m_i * g)
    floor_weights_n = building.masses * GRAVITY
    total_w_n = float(np.sum(floor_weights_n))
    total_w_kn = total_w_n / 1e3

    # 4. Design Base Shear VB = Ah * W
    vb_n = float(ah * total_w_n)
    vb_kn = vb_n / 1e3

    # 5. Vertical Distribution of Lateral Forces:
    # Qi = VB * (Wi * hi^2) / sum(Wj * hj^2) per Clause 7.6.3
    elevations_h = building.storey_elevations
    wh2 = floor_weights_n * (elevations_h ** 2)
    sum_wh2 = float(np.sum(wh2))

    if sum_wh2 > 0:
        q_forces_n = vb_n * (wh2 / sum_wh2)
    else:
        q_forces_n = np.full(building.num_storeys, vb_n / building.num_storeys)

    # 6. Storey Shear Forces (from roof down to base)
    # V_i = sum_{j=i}^N Q_j
    storey_shears_n = np.zeros(building.num_storeys, dtype=np.float64)
    running_shear = 0.0
    for idx in range(building.num_storeys - 1, -1, -1):
        running_shear += q_forces_n[idx]
        storey_shears_n[idx] = running_shear

    # 7. Elastic Displacements and Drifts under Static Lateral Forces:
    # Solve K * u = Q
    k_matrix = building.stiffness_matrix
    displacements_m = np.linalg.solve(k_matrix, q_forces_n)

    # Storey drifts (delta_i = u_i - u_{i-1})
    u_prev = 0.0
    drifts_m = np.zeros(building.num_storeys, dtype=np.float64)
    pidr_list = np.zeros(building.num_storeys, dtype=np.float64)
    for idx in range(building.num_storeys):
        u_curr = displacements_m[idx]
        d_val = u_curr - u_prev
        drifts_m[idx] = d_val
        pidr_list[idx] = d_val / building.heights[idx]
        u_prev = u_curr

    max_pidr = float(np.max(pidr_list))
    allowable_limit = 0.004  # Clause 7.11.1 (0.4% under design lateral force)
    compliance = "PASSED (Compliant with Clause 7.11.1)" if max_pidr <= allowable_limit else "EXCEEDED (Non-Compliant)"

    return IS1893DesignResult(
        zone=zone,
        zone_factor_z=z_val,
        soil_type=soil_type,
        importance_factor_i=i_val,
        response_reduction_r=r_val,
        period_used_s=round(t_design, 4),
        period_source=source,
        spectral_acceleration_sa_g=round(sa_over_g, 4),
        seismic_coefficient_ah=round(ah, 6),
        total_seismic_weight_kn=round(total_w_kn, 2),
        design_base_shear_kn=round(vb_kn, 2),
        design_base_shear_n=round(vb_n, 2),
        storey_lateral_forces_n=q_forces_n.tolist(),
        storey_shear_forces_n=storey_shears_n.tolist(),
        storey_displacements_m=displacements_m.tolist(),
        storey_drifts_m=drifts_m.tolist(),
        interstorey_drift_ratios=pidr_list.tolist(),
        max_interstorey_drift_ratio=round(max_pidr, 6),
        allowable_drift_limit=allowable_limit,
        drift_compliance_status=compliance,
    )


def generate_3way_comparison_table(
    building: ShearBuilding,
    zone: str = "Zone IV",
    soil_type: str = "Type II (Medium)",
    physics_peak_base_shear_n: Optional[float] = None,
    physics_max_pidr: Optional[float] = None,
    surrogate_peak_base_shear_n: Optional[float] = None,
    surrogate_max_pidr: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate the first-class 3-way engineering comparison table.

    Compares:
    1. IS 1893:2016 Code Design (Equivalent Static / Response Spectrum)
    2. High-Fidelity Physics Solver (Newmark-beta Nonlinear/Time-History)
    3. AI Surrogate Model Prediction
    """
    is_res = analyze_is1893_equivalent_static(building, zone=zone, soil_type=soil_type)

    table_rows = [
        {
            "Engineering Parameter": "Design Base Shear (kN)",
            "IS 1893:2016 Code": f"{is_res.design_base_shear_kn:.2f} kN",
            "High-Fidelity Physics Solver": (
                f"{physics_peak_base_shear_n / 1e3:.2f} kN" if physics_peak_base_shear_n is not None else "N/A"
            ),
            "AI ML Surrogate": (
                f"{surrogate_peak_base_shear_n / 1e3:.2f} kN" if surrogate_peak_base_shear_n is not None else "N/A"
            ),
            "Engineering Context": f"IS 1893 Ah = {is_res.seismic_coefficient_ah:.4f} (R={is_res.response_reduction_r}, I={is_res.importance_factor_i})",
        },
        {
            "Engineering Parameter": "Maximum Interstorey Drift Ratio (PIDR)",
            "IS 1893:2016 Code": f"{is_res.max_interstorey_drift_ratio * 100:.3f} %",
            "High-Fidelity Physics Solver": (
                f"{physics_max_pidr * 100:.3f} %" if physics_max_pidr is not None else "N/A"
            ),
            "AI ML Surrogate": (
                f"{surrogate_max_pidr * 100:.3f} %" if surrogate_max_pidr is not None else "N/A"
            ),
            "Engineering Context": "IS 1893 Cl. 7.11.1 Limit = 0.400% (Elastic Design Force)",
        },
        {
            "Engineering Parameter": "Fundamental Natural Period (T1)",
            "IS 1893:2016 Code": f"{is_res.period_used_s:.3f} s ({is_res.period_source})",
            "High-Fidelity Physics Solver": f"{is_res.period_used_s:.3f} s (Eigenvalue modal analysis)",
            "AI ML Surrogate": f"{is_res.period_used_s:.3f} s (Physics feature input)",
            "Engineering Context": "Rayleigh / Matrix eigenvalue vs empirical Ta",
        },
        {
            "Engineering Parameter": "Code Drift Compliance",
            "IS 1893:2016 Code": is_res.drift_compliance_status,
            "High-Fidelity Physics Solver": (
                "Compliant (Transient Safety)" if (physics_max_pidr is not None and physics_max_pidr <= 0.015) else "Exceeds 1.5% Limit"
            ),
            "AI ML Surrogate": (
                "Compliant (Surrogate Verified)" if (surrogate_max_pidr is not None and surrogate_max_pidr <= 0.015) else "Exceeds Limit"
            ),
            "Engineering Context": "IS 1893 0.4% (Design) vs ASCE 1.0%-1.5% (MCE/Transient)",
        },
    ]

    return {
        "building_name": building.name,
        "num_storeys": building.num_storeys,
        "total_height_m": building.total_height,
        "total_mass_tonnes": building.total_mass / 1e3,
        "zone": zone,
        "soil_type": soil_type,
        "table_rows": table_rows,
        "is1893_details": is_res.to_dict(),
    }
