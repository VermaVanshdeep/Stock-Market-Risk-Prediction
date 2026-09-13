"""
Unit tests for Deep Neuro-Fuzzy Neural Network Architecture.
"""

import unittest
import numpy as np
from models.deep_neuro_fuzzy_net import DeepNeuroFuzzyRiskModel
from models.fuzzy_layer import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


class TestDeepNeuroFuzzyModel(unittest.TestCase):

    def setUp(self):
        self.model = DeepNeuroFuzzyRiskModel()

    def test_forward_pass_vector_and_sequence(self):
        batch_size = 4
        vec_inputs = np.random.randn(batch_size, 4).astype(np.float32)
        seq_inputs = np.random.randn(batch_size, 30, 4).astype(np.float32)

        if TORCH_AVAILABLE:
            vec_t = torch.tensor(vec_inputs)
            seq_t = torch.tensor(seq_inputs)
            risk_score, logits, pred_mdd, fuzzy_info = self.model(vec_t, seq_t)
        else:
            risk_score, logits, pred_mdd, fuzzy_info = self.model.forward(vec_inputs, seq_inputs)

        self.assertEqual(risk_score.shape, (batch_size, 1))
        self.assertEqual(logits.shape, (batch_size, 4))
        self.assertEqual(pred_mdd.shape, (batch_size, 1))

    def test_single_prediction_inference(self):
        input_vec = np.array([0.5, 0.25, 0.35, 0.30], dtype=np.float32)
        score, category, mdd, labels = self.model.predict_risk(input_vec)

        self.assertIsInstance(score, float)
        self.assertIn(category, ["Low", "Moderate", "High", "Extreme"])
        self.assertIsInstance(mdd, float)
        self.assertIn('sharpe', labels)


if __name__ == '__main__':
    unittest.main()
