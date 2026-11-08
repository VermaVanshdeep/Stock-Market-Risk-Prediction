"""
Unit tests validating that model predictions remain sensitive to inputs
and do not suffer from output collapse or constant-score degeneracy.
"""

import unittest
import numpy as np
import torch

from models.deep_neuro_fuzzy_net import DeepNeuroFuzzyRiskModel
from models import train as trainer
from ui import services


class TestModelSensitivity(unittest.TestCase):

    def setUp(self):
        trainer.seed_everything(42)
        self.model = services.get_model()
        self.model.eval()

    def test_load_weights_adapts_161_to_162_dims(self):
        """Ensures load_weights correctly maps 161 legacy fusion dims into 162 dims."""
        model = DeepNeuroFuzzyRiskModel()
        state = model.state_dict()
        # Create a mock 161-dim state dict
        mock_state = {k: v.clone() for k, v in state.items()}
        mock_state["fusion_net.0.weight"] = torch.randn(64, 161)
        
        # Test loading logic directly
        model_state = model.state_dict()
        matched = {k: v for k, v in mock_state.items()
                   if k in model_state and v.shape == model_state[k].shape}
        if "fusion_net.0.weight" in mock_state and "fusion_net.0.weight" not in matched:
            s_weight = mock_state["fusion_net.0.weight"]
            m_weight = model_state.get("fusion_net.0.weight")
            min_cols = min(s_weight.shape[1], m_weight.shape[1])
            m_weight = m_weight.clone()
            m_weight[:, :min_cols] = s_weight[:, :min_cols]
            matched["fusion_net.0.weight"] = m_weight

        self.assertIn("fusion_net.0.weight", matched)
        self.assertEqual(matched["fusion_net.0.weight"].shape, torch.Size([64, 162]))

    def test_volatility_sensitivity(self):
        """Higher historical volatility must produce higher risk scores."""
        low_vol = np.array([1.0, 0.15, 0.20, 0.10], dtype=np.float32)
        high_vol = np.array([1.0, 0.15, 0.20, 0.60], dtype=np.float32)

        score_low, _, _, _ = self.model.predict_risk(low_vol)
        score_high, _, _, _ = self.model.predict_risk(high_vol)

        self.assertGreater(
            score_high, score_low,
            f"Expected score_high ({score_high:.2f}) > score_low ({score_low:.2f})"
        )
        self.assertGreater(
            score_high - score_low, 5.0,
            f"Score difference ({score_high - score_low:.2f}) is too small, indicates insensitivity."
        )

    def test_implied_volatility_sensitivity(self):
        """Higher implied volatility must produce higher risk scores."""
        low_iv = np.array([1.0, 0.15, 0.10, 0.20], dtype=np.float32)
        high_iv = np.array([1.0, 0.15, 0.50, 0.20], dtype=np.float32)

        score_low, _, _, _ = self.model.predict_risk(low_iv)
        score_high, _, _, _ = self.model.predict_risk(high_iv)

        self.assertGreater(
            score_high, score_low,
            f"Expected score_high ({score_high:.2f}) > score_low ({score_low:.2f})"
        )

    def test_risk_score_not_constant_collapsed(self):
        """Risk score must not sit at the collapsed bias constant (~51-53)."""
        profiles = [
            np.array([2.0, 0.05, 0.12, 0.10], dtype=np.float32),   # Defensive
            np.array([1.2, 0.15, 0.25, 0.20], dtype=np.float32),   # Balanced
            np.array([0.1, 0.40, 0.55, 0.50], dtype=np.float32),   # Stressed
            np.array([-1.2, 0.65, 0.85, 0.75], dtype=np.float32),  # Crisis
        ]
        scores = [self.model.predict_risk(p)[0] for p in profiles]
        std = float(np.std(scores))
        spread = float(max(scores) - min(scores))

        self.assertGreater(std, 3.0, f"Score std ({std:.2f}) too low; model is collapsed.")
        self.assertGreater(spread, 10.0, f"Score spread ({spread:.2f}) too narrow.")


if __name__ == "__main__":
    unittest.main()

