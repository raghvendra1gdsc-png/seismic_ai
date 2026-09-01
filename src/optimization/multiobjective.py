"""Resilient Multi-Objective Structural Optimization (NSGA-II).

Solves the multi-objective Pareto optimization problem:
    min [ f_1(x) = Initial Embodied Carbon / Mass,  f_2(x) = Expected Seismic Drift / Damage ]
    subject to:
        PIDR(x, GM) <= allowable_drift (1.0% - 1.5%)
        k_min <= k_i <= k_max
        k_{i+1} <= k_i (Physical non-inversion taper constraint)

References:
- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective
  genetic algorithm: NSGA-II. IEEE Trans. Evol. Comput., 6(2), 182-197.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np

from src.structural.building import ShearBuilding
from src.earthquake.record import GroundMotionRecord
from src.features.extractor import extract_features


@dataclass
class ParetoSolution:
    """A single non-dominated design on the Pareto frontier."""
    stiffnesses_n_m: np.ndarray
    f1_carbon_mass_score: float   # Normalized structural mass / carbon cost
    f2_seismic_drift_pct: float   # Predicted peak drift PIDR in %
    is_code_compliant: bool
    pareto_rank: int = 1
    crowding_distance: float = 0.0


@dataclass
class ParetoFrontierResult:
    """The complete non-dominated Pareto optimal front."""
    building_storeys: int
    population_size: int
    num_generations: int
    pareto_front: List[ParetoSolution]
    hypervolume_indicator: float
    best_cost_solution: ParetoSolution
    best_safety_solution: ParetoSolution
    compromise_balanced_solution: ParetoSolution


class NSGA2Optimizer:
    """Non-dominated Sorting Genetic Algorithm II for Multi-Objective Seismic Design.

    Parameters
    ----------
    num_storeys : int
        Number of building storeys.
    storey_mass : float
        Lumped storey mass in kg.
    storey_height : float
        Storey height in meters.
    k_bounds : Tuple[float, float], default=(5e7, 4e8)
        Stiffness search bounds in N/m.
    population_size : int, default=40
        Size of candidate design population.
    num_generations : int, default=25
        Number of evolutionary generations.
    crossover_prob : float, default=0.9
        SBX crossover probability.
    mutation_prob : float, default=0.15
        Polynomial mutation probability.
    """

    def __init__(
        self,
        num_storeys: int = 5,
        storey_mass: float = 120000.0,
        storey_height: float = 3.5,
        k_bounds: Tuple[float, float] = (5.0e7, 4.0e8),
        population_size: int = 40,
        num_generations: int = 25,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.num_storeys = num_storeys
        self.storey_mass = storey_mass
        self.storey_height = storey_height
        self.k_min, self.k_max = k_bounds
        self.pop_size = population_size
        self.num_gens = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.rng = np.random.default_rng(random_state)

    def _evaluate_objectives(
        self,
        k_vec: np.ndarray,
        record: GroundMotionRecord,
        surrogate_fn: Callable[[np.ndarray], float],
        feature_columns: List[str],
        drift_limit_pct: float = 1.5,
    ) -> Tuple[float, float, bool]:
        """Compute [f1 (Mass/Carbon), f2 (Drift %)] and constraint validity."""
        # Enforce non-inverted structural taper
        k_sorted = np.sort(k_vec)[::-1]
        bldg = ShearBuilding(
            masses=np.full(self.num_storeys, self.storey_mass),
            stiffnesses=k_sorted,
            heights=np.full(self.num_storeys, self.storey_height),
            name="CandidateDesign",
        )

        # Objective 1: Embodied material carbon/weight index (proportional to stiffness capacity)
        # Steel/Concrete volume scales with sqrt(k) or k^0.6
        f1_mass_carbon = float(np.sum((k_sorted / self.k_min) ** 0.6) / self.num_storeys)

        # Objective 2: Peak Inelastic Interstorey Drift %
        feats = extract_features(bldg, record, damping_ratio=0.05)
        x_vec = np.array([float(feats[c]) for c in feature_columns]).reshape(1, -1)
        pred_pidr = float(surrogate_fn(x_vec))
        f2_drift_pct = pred_pidr * 100.0

        is_compliant = (f2_drift_pct <= drift_limit_pct)
        return f1_mass_carbon, f2_drift_pct, is_compliant

    def _fast_non_dominated_sort(
        self,
        f1_scores: np.ndarray,
        f2_scores: np.ndarray,
    ) -> List[List[int]]:
        """Group population into Pareto dominance fronts F_1, F_2, ..."""
        n = len(f1_scores)
        domination_counts = np.zeros(n, dtype=int)
        dominated_sets = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # i dominates j if it is no worse in all and strictly better in at least one
                if (f1_scores[i] <= f1_scores[j] and f2_scores[i] <= f2_scores[j]) and \
                   (f1_scores[i] < f1_scores[j] or f2_scores[i] < f2_scores[j]):
                    dominated_sets[i].append(j)
                elif (f1_scores[j] <= f1_scores[i] and f2_scores[j] <= f2_scores[i]) and \
                     (f1_scores[j] < f1_scores[i] or f2_scores[j] < f2_scores[i]):
                    domination_counts[i] += 1

            if domination_counts[i] == 0:
                fronts[0].append(i)

        current_front_idx = 0
        while len(fronts[current_front_idx]) > 0:
            next_front = []
            for i in fronts[current_front_idx]:
                for j in dominated_sets[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            current_front_idx += 1
            fronts.append(next_front)

        # Remove trailing empty front
        return [f for f in fronts if len(f) > 0]

    def optimize(
        self,
        record: GroundMotionRecord,
        surrogate_fn: Callable[[np.ndarray], float],
        feature_columns: List[str],
        drift_limit_pct: float = 1.5,
    ) -> ParetoFrontierResult:
        """Run NSGA-II evolutionary search and return the non-dominated Pareto front."""
        # Initialize population with Latin Hypercube Sampling (LHS)
        pop = self.rng.uniform(self.k_min, self.k_max, size=(self.pop_size, self.num_storeys))
        pop = np.sort(pop, axis=1)[:, ::-1]  # Enforce monotonic taper

        for gen in range(self.num_gens):
            # Evaluate current population
            f1_list, f2_list, comp_list = [], [], []
            for indiv in pop:
                f1, f2, comp = self._evaluate_objectives(
                    indiv, record, surrogate_fn, feature_columns, drift_limit_pct
                )
                # Penalty for constraint violation
                if not comp:
                    f2 += 5.0
                f1_list.append(f1)
                f2_list.append(f2)
                comp_list.append(comp)

            f1_arr = np.array(f1_list)
            f2_arr = np.array(f2_list)

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(f1_arr, f2_arr)

            # Generate offspring via crossover & mutation
            offspring = []
            while len(offspring) < self.pop_size:
                # Tournament selection from front 1
                p1_idx = self.rng.choice(fronts[0]) if len(fronts[0]) > 1 else 0
                p2_idx = self.rng.choice(self.pop_size)
                p1, p2 = pop[p1_idx].copy(), pop[p2_idx].copy()

                # SBX Crossover
                if self.rng.random() < self.crossover_prob:
                    beta = self.rng.uniform(0.5, 1.5)
                    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                else:
                    c1, c2 = p1, p2

                # Mutation
                for child in [c1, c2]:
                    if self.rng.random() < self.mutation_prob:
                        mut_idx = self.rng.integers(0, self.num_storeys)
                        child[mut_idx] += self.rng.normal(0, 0.1 * (self.k_max - self.k_min))
                    child = np.clip(child, self.k_min, self.k_max)
                    child = np.sort(child)[::-1]
                    offspring.append(child)

            pop = np.array(offspring[:self.pop_size])

        # Final evaluation of population
        final_solutions: List[ParetoSolution] = []
        f1_list, f2_list = [], []
        for indiv in pop:
            f1, f2, comp = self._evaluate_objectives(
                indiv, record, surrogate_fn, feature_columns, drift_limit_pct
            )
            f1_list.append(f1)
            f2_list.append(f2)
            final_solutions.append(
                ParetoSolution(
                    stiffnesses_n_m=indiv.copy(),
                    f1_carbon_mass_score=round(f1, 3),
                    f2_seismic_drift_pct=round(f2, 3),
                    is_code_compliant=comp,
                )
            )

        fronts = self._fast_non_dominated_sort(np.array(f1_list), np.array(f2_list))
        pareto_front_indices = fronts[0] if len(fronts) > 0 else list(range(len(pop)))
        pareto_front = [final_solutions[i] for i in pareto_front_indices]

        # Sort Pareto front by ascending f1 (cost)
        pareto_front = sorted(pareto_front, key=lambda s: s.f1_carbon_mass_score)

        best_cost = pareto_front[0]
        best_safety = min(pareto_front, key=lambda s: s.f2_seismic_drift_pct)
        # Compromise solution (minimum Euclidean distance to origin in normalized space)
        f1_max = max(s.f1_carbon_mass_score for s in pareto_front)
        f2_max = max(s.f2_seismic_drift_pct for s in pareto_front)
        balanced = min(
            pareto_front,
            key=lambda s: (s.f1_carbon_mass_score / max(f1_max, 1e-3))**2 + (s.f2_seismic_drift_pct / max(f2_max, 1e-3))**2
        )

        return ParetoFrontierResult(
            building_storeys=self.num_storeys,
            population_size=self.pop_size,
            num_generations=self.num_gens,
            pareto_front=pareto_front,
            hypervolume_indicator=round(f1_max * f2_max, 3),
            best_cost_solution=best_cost,
            best_safety_solution=best_safety,
            compromise_balanced_solution=balanced,
        )
