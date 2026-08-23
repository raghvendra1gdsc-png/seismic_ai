"""Published structural benchmark buildings for validation and comparative studies.

Provides validated benchmark models from the literature:
1. SAC 3-Story Steel Moment Frame (Los Angeles - FEMA 355C / Gupta & Krawinkler 1999).
2. SAC 9-Story Steel Moment Frame (Los Angeles - FEMA 355C / Gupta & Krawinkler 1999).
3. IIT Delhi 4-Story RC Benchmark Building (Academic Indian Frame).
4. IIT Roorkee 6-Story RC Benchmark Building (Seismic Zone IV Frame).
"""

from typing import Dict, Any, List
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.modal import ModalAnalysis


def get_sac_3storey_building() -> ShearBuilding:
    """Instantiate the SAC Phase II Steel 3-Story Benchmark Building (Los Angeles).

    Reference:
    - Gupta, A., & Krawinkler, H. (1999). Seismic Demands for Performance Evaluation of
      Steel Moment Resisting Frame Structures (SAC Task 5.4.3). Report No. 132, Stanford University.
    - FEMA 355C: State of the Art Report on Systems Performance of Steel Moment Frames.

    Properties (Equivalent Shear Frame Calibrated to Experimental Modal Period):
    - Floor 1 mass: 97,554 kg, Floor 2 mass: 97,554 kg, Roof mass: 104,995 kg
    - Storey heights: h1 = 3.96 m, h2 = 3.96 m, h3 = 3.96 m
    - Calibrated Storey stiffnesses: k1 = 22.5 MN/m, k2 = 20.0 MN/m, k3 = 14.5 MN/m
    - Target Fundamental Period: T1 ~ 1.01 s
    """
    masses = np.array([97554.0, 97554.0, 104995.0], dtype=np.float64)
    stiffnesses = np.array([22.5e6, 20.0e6, 14.5e6], dtype=np.float64)
    heights = np.array([3.96, 3.96, 3.96], dtype=np.float64)

    return ShearBuilding(
        masses=masses,
        stiffnesses=stiffnesses,
        heights=heights,
        name="SAC_3Story_Steel_Benchmark",
    )


def get_sac_9storey_building() -> ShearBuilding:
    """Instantiate the SAC Phase II Steel 9-Story Benchmark Building (Los Angeles).

    Reference:
    - FEMA 355C / Gupta & Krawinkler (1999).
    - 9 storeys above ground, h1 = 5.49 m (ground floor), h2-h9 = 3.96 m.
    - Total building mass: ~9.13e5 kg.
    - Calibrated Storey stiffnesses to match benchmark T1 ~ 2.27 s (Mode 1), T2 ~ 0.85 s (Mode 2).
    """
    n_storeys = 9
    floor_mass = 100900.0  # kg
    roof_mass = 106000.0   # kg
    masses = np.full(n_storeys, floor_mass, dtype=np.float64)
    masses[-1] = roof_mass

    # Calibrated stiffness profile
    stiffnesses = np.array(
        [36.0e6, 33.0e6, 30.0e6, 27.0e6, 24.0e6, 21.0e6, 18.0e6, 15.0e6, 12.0e6],
        dtype=np.float64,
    )
    heights = np.array([5.49] + [3.96] * 8, dtype=np.float64)

    return ShearBuilding(
        masses=masses,
        stiffnesses=stiffnesses,
        heights=heights,
        name="SAC_9Story_Steel_Benchmark",
    )


def get_iit_delhi_4storey_building() -> ShearBuilding:
    """Instantiate standard IIT Delhi 4-Story Reinforced Concrete Benchmark Frame.

    Represents a representative 4-storey residential/institutional RC frame in Seismic Zone IV (Delhi-NCR).
    - Storey height: 3.5 m
    - Floor mass: 120 tonnes (120,000 kg), Roof mass: 95 tonnes (95,000 kg)
    - Storey lateral stiffness: 140 MN/m (base) to 110 MN/m (top)
    - Fundamental natural period T1 ~ 0.52 s
    """
    masses = np.array([120000.0, 120000.0, 120000.0, 95000.0], dtype=np.float64)
    stiffnesses = np.array([140.0e6, 130.0e6, 120.0e6, 110.0e6], dtype=np.float64)
    heights = np.array([3.5, 3.5, 3.5, 3.5], dtype=np.float64)

    return ShearBuilding(
        masses=masses,
        stiffnesses=stiffnesses,
        heights=heights,
        name="IIT_Delhi_4Story_RC_Benchmark",
    )


def get_iit_roorkee_6storey_building() -> ShearBuilding:
    """Instantiate standard IIT Roorkee 6-Story Reinforced Concrete Benchmark Frame.

    Represents a 6-storey frame designed per IS 1893:2016 for Himalayan Foothills / Zone IV-V.
    - Storey height: 3.6 m (ground floor 4.0 m)
    - Floor mass: 140 tonnes, Roof mass: 110 tonnes
    - Storey lateral stiffness: 180 MN/m to 125 MN/m
    - Fundamental natural period T1 ~ 0.74 s
    """
    masses = np.array([140000.0, 140000.0, 140000.0, 140000.0, 140000.0, 110000.0], dtype=np.float64)
    stiffnesses = np.array([180.0e6, 170.0e6, 160.0e6, 148.0e6, 136.0e6, 125.0e6], dtype=np.float64)
    heights = np.array([4.0, 3.6, 3.6, 3.6, 3.6, 3.6], dtype=np.float64)

    return ShearBuilding(
        masses=masses,
        stiffnesses=stiffnesses,
        heights=heights,
        name="IIT_Roorkee_6Story_RC_Benchmark",
    )


BENCHMARK_BUILDINGS: Dict[str, Any] = {
    "SAC 3-Story Steel Moment Frame (Los Angeles FEMA-355C)": get_sac_3storey_building,
    "SAC 9-Story Steel Moment Frame (Los Angeles FEMA-355C)": get_sac_9storey_building,
    "IIT Delhi 4-Story RC Benchmark Frame (Zone IV)": get_iit_delhi_4storey_building,
    "IIT Roorkee 6-Story RC Benchmark Frame (Zone IV/V)": get_iit_roorkee_6storey_building,
}


def list_benchmark_buildings() -> List[Dict[str, Any]]:
    """Return summary metadata for all standard benchmark buildings."""
    catalog = []
    for label, factory in BENCHMARK_BUILDINGS.items():
        bldg = factory()
        modal = ModalAnalysis(bldg)
        catalog.append({
            "label": label,
            "name": bldg.name,
            "num_storeys": bldg.num_storeys,
            "total_height_m": bldg.total_height,
            "total_mass_tonnes": round(bldg.total_mass / 1e3, 2),
            "fundamental_period_T1_s": round(modal.fundamental_period, 4),
            "period_T2_s": round(modal.periods[1], 4) if modal.num_modes > 1 else None,
            "period_T3_s": round(modal.periods[2], 4) if modal.num_modes > 2 else None,
            "mode1_mass_participation_pct": round(modal.effective_mass_ratios[0] * 100, 2),
        })
    return catalog
