"""Differential Evolution and Genetic Algorithm optimizer for structural design."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.optimization.problem import SeismicOptimizationProblem


class DifferentialEvolutionOptimizer:
    """Differential Evolution optimizer for constrained structural seismic design."""

    def __init__(
        self,
        problem: SeismicOptimizationProblem,
        pop_size: int = 35,
        max_generations: int = 30,
        mutation_factor: float = 0.7,
        crossover_rate: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.problem = problem
        self.pop_size = int(pop_size)
        self.max_generations = int(max_generations)
        self.f_mut = float(mutation_factor)
        self.cr = float(crossover_rate)
        self.rng = np.random.RandomState(seed)

    def optimize(self) -> Dict[str, Any]:
        """Execute evolutionary search for optimal storey stiffness distribution."""
        n_dim = self.problem.num_storeys
        k_min, k_max = self.problem.k_bounds

        # Initialize population with decreasing stiffness profiles
        pop = np.zeros((self.pop_size, n_dim), dtype=np.float64)
        for i in range(self.pop_size):
            base_val = self.rng.uniform(k_min * 1.5, k_max)
            taper = self.rng.uniform(0.05, 0.45)
            factors = np.linspace(1.0, 1.0 - taper, n_dim)
            pop[i] = np.clip(base_val * factors, k_min, k_max)

        # Evaluate initial population
        fitness = np.array([self.problem.evaluate_surrogate(ind)["loss"] for ind in pop])

        best_idx = np.argmin(fitness)
        best_vec = pop[best_idx].copy()
        best_loss = fitness[best_idx]
        history = [best_loss]

        # Generations loop
        for gen in range(self.max_generations):
            for i in range(self.pop_size):
                # Select 3 distinct random candidates != i
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = self.rng.choice(candidates, size=3, replace=False)

                # Mutation: mutant = r1 + F * (r2 - r3)
                mutant = pop[r1] + self.f_mut * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, k_min, k_max)

                # Binomial crossover
                cross_mask = self.rng.rand(n_dim) < self.cr
                cross_mask[self.rng.randint(0, n_dim)] = True  # Ensure at least one mutated dim
                trial = np.where(cross_mask, mutant, pop[i])

                # Selection
                trial_eval = self.problem.evaluate_surrogate(trial)
                trial_loss = trial_eval["loss"]

                if trial_loss < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_loss

                    if trial_loss < best_loss:
                        best_loss = trial_loss
                        best_vec = trial.copy()

            history.append(best_loss)

        best_eval = self.problem.evaluate_surrogate(best_vec)

        return {
            "optimal_stiffnesses": best_vec.tolist(),
            "best_loss": best_loss,
            "total_stiffness": best_eval["total_stiffness"],
            "predicted_pidr": best_eval["predicted_pidr"],
            "is_feasible": best_eval["is_feasible"],
            "history": history,
            "generations": self.max_generations,
        }
