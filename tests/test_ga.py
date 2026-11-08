"""
Unit tests for the Genetic Algorithm Evolutionary Optimizer.

Updated for the corrected 24-gene chromosome implementation (audit fixes D1-D5):
- initialize_population(rng)  — requires a seeded Generator (D4 fix)
- compute_fitness(chromosome, X_vec, X_seq, y_val, model)  — full model (D1 fix)
- optimize(X_vec, X_seq, y_val, model, seed=...)  — requires sequence + model
"""

import unittest
import numpy as np

from models.deep_neuro_fuzzy_net import DeepNeuroFuzzyRiskModel
from models.fuzzy_layer import TORCH_AVAILABLE
from models.genetic_optimizer import (
    CHROMOSOME_LEN,
    GeneticAlgorithmOptimizer,
    chromosome_to_params,
)

if TORCH_AVAILABLE:
    import torch


def _make_model():
    np.random.seed(0)
    if TORCH_AVAILABLE:
        torch.manual_seed(0)
    return DeepNeuroFuzzyRiskModel()


def _small_data(n=40):
    rng = np.random.default_rng(42)
    X_vec = rng.uniform(0, 1, (n, 4)).astype(np.float32)
    X_seq = rng.uniform(0, 1, (n, 20, 4)).astype(np.float32)
    y_val = rng.uniform(10, 80, n).astype(np.float32)
    return X_vec, X_seq, y_val


class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        self.ga = GeneticAlgorithmOptimizer(population_size=10, generations=3)
        self.rng = np.random.default_rng(42)
        self.model = _make_model()
        self.X_vec, self.X_seq, self.y_val = _small_data()

    def test_chromosome_length(self):
        self.assertEqual(CHROMOSOME_LEN, 24,
            "Chromosome must have exactly 24 genes (12 means + 12 sigmas).")

    def test_population_initialization(self):
        """initialize_population requires a seeded rng (D4 fix)."""
        pop = self.ga.initialize_population(self.rng)
        self.assertEqual(pop.shape, (10, CHROMOSOME_LEN))

    def test_sigmas_positive_in_population(self):
        pop = self.ga.initialize_population(self.rng)
        sigmas = pop[:, 12:24]
        self.assertTrue((sigmas > 0).all(), "All sigma genes must be positive.")

    def test_means_ordered_in_population(self):
        """After repair, means must be Low<=Medium<=High for each feature."""
        pop = self.ga.initialize_population(self.rng)
        for ind in pop:
            means = ind[:12].reshape(4, 3)
            for row in means:
                self.assertTrue(
                    np.all(np.diff(row) >= 0),
                    f"Means not ordered: {row}"
                )

    def test_fitness_computation(self):
        """compute_fitness now takes X_vec, X_seq, y_val, model (D1 fix)."""
        pop = self.ga.initialize_population(self.rng)
        fitness = self.ga.compute_fitness(
            pop[0], self.X_vec, self.X_seq, self.y_val, self.model
        )
        self.assertIsInstance(fitness, float)
        self.assertGreater(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)

    def test_ga_optimization_loop(self):
        """optimize() now requires X_seq and model (D1 + D4 fix)."""
        best_chromo, history = self.ga.optimize(
            self.X_vec, self.X_seq, self.y_val, self.model, seed=42
        )
        self.assertEqual(len(best_chromo), CHROMOSOME_LEN)
        self.assertEqual(len(history), 3)
        self.assertTrue(all(f > 0 for f in history))

    def test_chromosome_to_params_shapes(self):
        chromo = np.ones(CHROMOSOME_LEN, dtype=np.float32) * 0.3
        means, sigmas = chromosome_to_params(chromo)
        self.assertEqual(means.shape, (4, 3))
        self.assertEqual(sigmas.shape, (4, 3))

    def test_chromosome_to_params_sigmas_clipped(self):
        bad = np.full(CHROMOSOME_LEN, -1.0, dtype=np.float32)
        _, sigmas = chromosome_to_params(bad)
        self.assertTrue((sigmas >= 0.01).all())

    def test_chromosome_to_params_means_ordered(self):
        """After decode+repair, every feature row must be sorted Low->High."""
        rng = np.random.default_rng(7)
        # Deliberately reversed means
        chromo = np.zeros(CHROMOSOME_LEN, dtype=np.float32)
        # Drawdown: reversed [0.5, 0.3, 0.1]
        chromo[3:6] = [0.5, 0.3, 0.1]
        chromo[12:24] = rng.uniform(0.05, 0.5, 12)
        means, _ = chromosome_to_params(chromo)
        for row in means:
            self.assertTrue(np.all(np.diff(row) >= 0), f"Not ordered: {row}")


if __name__ == "__main__":
    unittest.main()
