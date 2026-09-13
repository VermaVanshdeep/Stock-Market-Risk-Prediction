"""
Unit tests for Soft Computing Takagi-Sugeno Fuzzy Logic Layer.
"""

import unittest
import numpy as np
from models.fuzzy_layer import PyTorchTakagiSugenoFuzzyLayer, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


class TestFuzzyLayer(unittest.TestCase):

    def setUp(self):
        self.fuzzy_layer = PyTorchTakagiSugenoFuzzyLayer()

    def test_membership_computation(self):
        x = np.array([
            [1.0, 0.15, 0.20, 0.20],
            [-0.5, 0.40, 0.50, 0.50]
        ], dtype=np.float32)

        if TORCH_AVAILABLE:
            x_input = torch.tensor(x)
        else:
            x_input = x

        mu = self.fuzzy_layer.compute_membership(x_input)
        
        # Shape should be (batch_size=2, num_inputs=4, num_mfs=3)
        self.assertEqual(mu.shape, (2, 4, 3))
        
        if TORCH_AVAILABLE:
            self.assertTrue((mu >= 0.0).all().item() and (mu <= 1.0).all().item())
        else:
            self.assertTrue((mu >= 0.0).all() and (mu <= 1.0).all())

    def test_fuzzy_layer_forward_pass(self):
        x = np.array([
            [1.5, 0.10, 0.15, 0.15]
        ], dtype=np.float32)

        if TORCH_AVAILABLE:
            x_input = torch.tensor(x)
        else:
            x_input = x

        fuzzy_score, firing_strengths = self.fuzzy_layer(x_input)
        self.assertEqual(fuzzy_score.shape, (1, 1))
        self.assertEqual(firing_strengths.shape, (1, 16))
        
        if TORCH_AVAILABLE:
            sum_val = float(torch.sum(firing_strengths).item())
        else:
            sum_val = float(np.sum(firing_strengths))
            
        self.assertAlmostEqual(sum_val, 1.0, places=4)

    def test_evaluate_fuzzification_labels(self):
        vec = np.array([2.0, 0.05, 0.12, 0.12], dtype=np.float32)
        labels = self.fuzzy_layer.evaluate_fuzzification_labels(vec)
        self.assertIn('sharpe', labels)
        self.assertEqual(labels['sharpe'], 'High')
        self.assertEqual(labels['drawdown'], 'Low')


if __name__ == '__main__':
    unittest.main()
