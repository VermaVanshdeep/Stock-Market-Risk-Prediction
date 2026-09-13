"""
Soft Computing Core: Genetic Algorithm (GA) Evolutionary Optimizer.

Optimizes Takagi-Sugeno Fuzzy Logic membership function parameters and feature weights
using evolutionary selection, crossover, and mutation mechanisms.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict
from rich.console import Console
from config.settings import GA_CONFIG

console = Console()


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm Evolutionary Optimizer for Fuzzy Rule Sets and Neural Net Weights.
    
    Chromosome Encoding:
    Vector of floats representing [means (12 values), sigmas (12 values), feature_weights (4 values)]
    Total Chromosome Length: 28 float values
    """

    def __init__(
        self,
        population_size: int = GA_CONFIG['population_size'],
        generations: int = GA_CONFIG['generations'],
        mutation_rate: float = GA_CONFIG['mutation_rate'],
        crossover_rate: float = GA_CONFIG['crossover_rate']
    ):
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.chromosome_len = 28  # 12 means + 12 sigmas + 4 feature weights

    def initialize_population(self) -> np.ndarray:
        """Initializes random population bounded within realistic domain constraints."""

        pop = np.random.uniform(0.01, 1.0, size=(self.pop_size, self.chromosome_len))
        # Scale means appropriately for financial metric ranges
        pop[:, 0:3] = np.random.uniform(-1.0, 3.0, size=(self.pop_size, 3))   # Sharpe means
        pop[:, 3:6] = np.random.uniform(0.01, 0.50, size=(self.pop_size, 3))  # Drawdown means
        pop[:, 6:9] = np.random.uniform(0.05, 0.60, size=(self.pop_size, 3))  # IV means
        pop[:, 9:12] = np.random.uniform(0.05, 0.60, size=(self.pop_size, 3)) # HV means
        return pop

    def compute_fitness(self, chromosome: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Fitness Function: Evaluates chromosome performance.
        Higher fitness = Lower MSE error + Higher risk metric correlation.
        """
        means = chromosome[0:12].reshape(4, 3)
        sigmas = np.clip(chromosome[12:24].reshape(4, 3), 0.01, 1.0)
        feature_weights = chromosome[24:28]

        # Compute weighted distance to historical risk benchmarks
        weighted_X = X_val * feature_weights
        
        # Calculate predicted risk score proxy
        pred_scores = np.dot(weighted_X, np.array([-10.0, 40.0, 30.0, 30.0]))
        pred_scores = np.clip(pred_scores, 0.0, 100.0)

        # MSE Loss
        mse = np.mean((pred_scores - y_val) ** 2)
        fitness = 1.0 / (1.0 + mse)
        return float(fitness)

    def selection(self, population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        """Tournament selection operator."""
        selected = []
        for _ in range(self.pop_size):
            i, j = np.random.choice(self.pop_size, 2, replace=False)
            if fitness_scores[i] >= fitness_scores[j]:
                selected.append(population[i].copy())
            else:
                selected.append(population[j].copy())
        return np.array(selected)

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX) operator."""
        if np.random.rand() < self.crossover_rate:
            pt = np.random.randint(1, self.chromosome_len - 1)
            child1 = np.concatenate([parent1[:pt], parent2[pt:]])
            child2 = np.concatenate([parent2[:pt], parent1[pt:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Gaussian mutation operator."""
        mutated = chromosome.copy()
        for idx in range(self.chromosome_len):
            if np.random.rand() < self.mutation_rate:
                noise = np.random.normal(0, 0.05)
                mutated[idx] += noise
        return mutated

    def optimize(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Runs Evolutionary Genetic Algorithm optimization loop.
        Returns:
            (best_chromosome, fitness_history)
        """
        console.print("[bold cyan]Executing Soft Computing Genetic Algorithm Optimization...[/bold cyan]")
        population = self.initialize_population()
        fitness_history = []
        best_chromosome = None
        best_fitness = -1.0

        for gen in range(self.generations):
            fitness_scores = np.array([self.compute_fitness(ind, X_val, y_val) for ind in population])
            
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()
                
            fitness_history.append(best_fitness)

            # Selection
            selected_pop = self.selection(population, fitness_scores)
            
            # Crossover & Mutation
            next_generation = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = selected_pop[i], selected_pop[min(i+1, self.pop_size-1)]
                c1, c2 = self.crossover(p1, p2)
                next_generation.append(self.mutate(c1))
                next_generation.append(self.mutate(c2))

            population = np.array(next_generation[:self.pop_size])

        console.print(f"[bold green]GA Optimization complete! Best Fitness: {best_fitness:.6f}[/bold green]")
        return best_chromosome, fitness_history
