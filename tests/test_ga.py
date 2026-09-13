"""
Unit tests for Genetic Algorithm Evolutionary Optimizer.
"""

import unittest
import numpy as np
from models.genetic_optimizer import GeneticAlgorithmOptimizer


class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        self.ga = GeneticAlgorithmOptimizer(population_size=10, generations=3)
        np.random.seed(42)
        self.X_val = np.random.uniform(0, 1, size=(20, 4))
        self.y_val = np.random.uniform(10, 80, size=20)

    def test_population_initialization(self):
        pop = self.ga.initialize_population()
        self.assertEqual(pop.shape, (10, 28))

    def test_fitness_computation(self):
        pop = self.ga.initialize_population()
        fitness = self.ga.compute_fitness(pop[0], self.X_val, self.y_val)
        self.assertIsInstance(fitness, float)
        self.assertGreater(fitness, 0.0)

    def test_ga_optimization_loop(self):
        best_chromo, history = self.ga.optimize(self.X_val, self.y_val)
        self.assertEqual(len(best_chromo), 28)
        self.assertEqual(len(history), 3)
        # Fitness should be non-decreasing or positive
        self.assertTrue(all(f > 0 for f in history))


if __name__ == '__main__':
    unittest.main()
