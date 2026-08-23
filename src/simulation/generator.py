"""Parametric multi-storey building generator with Latin Hypercube Sampling.

Generates realistic structural engineering models with varying heights, floor masses,
stiffness taper distributions, and modal damping ratios for simulation dataset synthesis.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.structural.building import ShearBuilding


class BuildingGenerator:
    """Parametric building generator for structural dataset generation.

    Parameters
    ----------
    seed : Optional[int], default=42
        Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = 42) -> None:
        self.rng = np.random.RandomState(seed)

    def generate_single_building(
        self,
        name: str,
        num_storeys: int,
        base_mass_kg: float,
        base_stiffness_N_m: float,
        storey_height_m: float,
        taper_type: str = "linear",
        taper_factor: float = 0.25,
        soft_first_storey: bool = False,
    ) -> ShearBuilding:
        """Generate a single building instance with specified structural attributes.

        Parameters
        ----------
        name : str
            Building identifier.
        num_storeys : int
            Number of storeys N.
        base_mass_kg : float
            Typical floor mass in kg (e.g. 150,000 kg).
        base_stiffness_N_m : float
            Base storey lateral stiffness in N/m (e.g. 1.2e8 N/m).
        storey_height_m : float
            Typical storey height in meters (e.g. 3.5 m).
        taper_type : {"uniform", "linear", "stepped"}, default="linear"
            Type of vertical stiffness reduction.
        taper_factor : float, default=0.25
            Fractional reduction in stiffness from base to roof (e.g., 0.25 -> top is 75% of base).
        soft_first_storey : bool, default=False
            Whether the first storey has lower stiffness (e.g., open ground storey).
        """
        n = num_storeys

        # Heights (first storey sometimes 10-20% taller)
        heights = np.full(n, storey_height_m, dtype=np.float64)
        if soft_first_storey or self.rng.rand() < 0.3:
            heights[0] = storey_height_m * self.rng.uniform(1.1, 1.25)

        # Floor masses (roof is typically 15-30% lighter)
        masses = np.full(n, base_mass_kg, dtype=np.float64)
        # Small floor-to-floor random variation (+/- 5%)
        mass_variations = self.rng.uniform(0.95, 1.05, size=n)
        masses = masses * mass_variations
        masses[-1] = masses[-1] * self.rng.uniform(0.70, 0.85)  # Roof mass reduction

        # Stiffness distribution
        if taper_type == "uniform" or n <= 2:
            stiffnesses = np.full(n, base_stiffness_N_m, dtype=np.float64)
        elif taper_type == "linear":
            # Linear reduction from base (1.0) to roof (1.0 - taper_factor)
            factors = np.linspace(1.0, max(0.4, 1.0 - taper_factor), n)
            stiffnesses = base_stiffness_N_m * factors
        elif taper_type == "stepped":
            # Step reduction every 2-3 storeys
            step_size = max(2, n // 3)
            factors = np.ones(n)
            for i in range(n):
                step_idx = i // step_size
                factors[i] = max(0.45, 1.0 - step_idx * (taper_factor / max(1, (n // step_size))))
            stiffnesses = base_stiffness_N_m * factors
        else:
            stiffnesses = np.full(n, base_stiffness_N_m, dtype=np.float64)

        if soft_first_storey:
            stiffnesses[0] = stiffnesses[0] * 0.75  # 25% lower stiffness on first storey

        return ShearBuilding(
            masses=masses,
            stiffnesses=stiffnesses,
            heights=heights,
            name=name,
        )

    def generate_suite(
        self,
        num_buildings: int,
        min_storeys: int = 3,
        max_storeys: int = 10,
    ) -> List[Tuple[ShearBuilding, float]]:
        """Generate a diverse suite of buildings using Latin Hypercube / Stratified parameter sampling.

        Returns
        -------
        List[Tuple[ShearBuilding, float]]
            List of (ShearBuilding, damping_ratio) tuples.
        """
        suite = []

        # Stratified storey counts
        storey_counts = self.rng.randint(min_storeys, max_storeys + 1, size=num_buildings)

        # Parameter ranges
        # Mass: 80 tonnes (80,000 kg) to 350 tonnes (350,000 kg)
        # Stiffness: 50 MN/m (5e7 N/m) to 350 MN/m (3.5e8 N/m)
        # Height: 3.0 m to 4.2 m
        # Damping: 2% to 6% (0.02 to 0.06)
        # Taper factor: 0.0 to 0.45

        # Latin Hypercube intervals
        u_samples = (np.arange(num_buildings) + self.rng.uniform(0.1, 0.9, size=num_buildings)) / num_buildings
        self.rng.shuffle(u_samples)

        for i in range(num_buildings):
            n_st = int(storey_counts[i])
            u = u_samples[i]

            base_mass = 80000.0 + u * (350000.0 - 80000.0) + self.rng.uniform(-10000, 10000)
            base_mass = max(60000.0, base_mass)

            # Correlate stiffness somewhat with mass to maintain realistic periods (T ~ 0.05 - 0.1 N)
            target_period_estimate = n_st * self.rng.uniform(0.06, 0.12)
            # wn ~ 2*pi/T -> k_eff / m_eff ~ wn^2 -> k ~ m * wn^2 * (pi^2 / 4)
            wn_est = 2.0 * np.pi / target_period_estimate
            base_k = base_mass * (wn_est ** 2) * (n_st / 2.0)
            base_k = np.clip(base_k, 4e7, 4e8)

            h_st = self.rng.uniform(3.0, 4.2)
            damping_val = self.rng.uniform(0.02, 0.06)

            taper_choice = self.rng.choice(["uniform", "linear", "stepped"], p=[0.25, 0.50, 0.25])
            taper_val = self.rng.uniform(0.10, 0.45)
            soft_storey = self.rng.rand() < 0.15

            bldg = self.generate_single_building(
                name=f"Bldg_{i+1:03d}_{n_st}St",
                num_storeys=n_st,
                base_mass_kg=base_mass,
                base_stiffness_N_m=base_k,
                storey_height_m=h_st,
                taper_type=taper_choice,
                taper_factor=taper_val,
                soft_first_storey=soft_storey,
            )

            suite.append((bldg, float(damping_val)))

        return suite
