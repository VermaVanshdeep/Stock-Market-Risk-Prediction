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

    def test_phase7_parameter_constraints(self):
        """Phase 7: Enforce sigma in [0.01, 2.0] and Low < Medium < High ordering."""
        # Inject intentionally broken parameters: negative sigma, inverted means
        if TORCH_AVAILABLE:
            with torch.no_grad():
                self.fuzzy_layer.sigmas.copy_(torch.tensor([
                    [-0.5, 0.0, 5.0],
                    [0.001, -0.0152, 2.5],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1]
                ]))
                self.fuzzy_layer.means.copy_(torch.tensor([
                    [2.0, 0.5, -1.0],  # reversed
                    [0.5, 0.1, 0.2],   # scrambled
                    [0.1, 0.2, 0.3],
                    [0.3975, 0.5389, 0.4500]  # checkpoint defect
                ]))
        else:
            self.fuzzy_layer.sigmas = np.array([
                [-0.5, 0.0, 5.0],
                [0.001, -0.0152, 2.5],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1]
            ], dtype=np.float32)
            self.fuzzy_layer.means = np.array([
                [2.0, 0.5, -1.0],
                [0.5, 0.1, 0.2],
                [0.1, 0.2, 0.3],
                [0.3975, 0.5389, 0.4500]
            ], dtype=np.float32)

        self.fuzzy_layer.clamp_and_sort_parameters()

        if TORCH_AVAILABLE:
            sigmas = self.fuzzy_layer.sigmas.detach().cpu().numpy()
            means = self.fuzzy_layer.means.detach().cpu().numpy()
        else:
            sigmas = self.fuzzy_layer.sigmas
            means = self.fuzzy_layer.means

        # 1. Sigmas must be in [0.01, 2.0]
        self.assertTrue((sigmas >= 0.01).all(), f"Found sigma < 0.01: {sigmas}")
        self.assertTrue((sigmas <= 2.0).all(), f"Found sigma > 2.0: {sigmas}")

        # 2. Means must be strictly ordered Low <= Medium <= High
        for i, row in enumerate(means):
            self.assertTrue(np.all(np.diff(row) >= 0), f"Row {i} not sorted: {row}")

    def test_phase7_membership_finiteness_and_partition_of_unity(self):
        """Phase 7: Verify memberships remain finite and rule firings sum to 1.0."""
        # Test boundary and extreme values
        x_extreme = np.array([
            [-10.0, 0.0, 0.01, 0.01],
            [10.0, 1.0, 2.0, 2.0],
            [0.0, 0.2, 0.25, 0.25]
        ], dtype=np.float32)

        if TORCH_AVAILABLE:
            x_tensor = torch.tensor(x_extreme)
            mu = self.fuzzy_layer.compute_membership(x_tensor)
            self.assertTrue(torch.isfinite(mu).all().item())
            _, w_norm = self.fuzzy_layer(x_tensor)
            w_sums = torch.sum(w_norm, dim=1).detach().cpu().numpy()
        else:
            mu = self.fuzzy_layer.compute_membership(x_extreme)
            self.assertTrue(np.isfinite(mu).all())
            _, w_norm = self.fuzzy_layer.forward(x_extreme)
            w_sums = np.sum(w_norm, axis=1)

        for s in w_sums:
            self.assertAlmostEqual(float(s), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
