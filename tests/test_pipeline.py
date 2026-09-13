"""
End-to-End Integration Tests for Stock Risk Prediction System.
Verifies seamless pipeline execution from data downloading -> metric extraction ->
fuzzy inference -> deep net prediction -> reporting.
"""

import unittest
from data.data_loader import SyntheticStockGenerator
from data.feature_extractor import RiskFeatureExtractor
from models.deep_neuro_fuzzy_net import DeepNeuroFuzzyRiskModel


class TestPipelineIntegration(unittest.TestCase):

    def test_full_stock_prediction_pipeline(self):
        # 1. Generate data
        df = SyntheticStockGenerator.generate_stock_data(ticker="AAPL_BENCHMARK", days=120, seed=42)
        self.assertIsNotNone(df)

        # 2. Extract metrics
        extractor = RiskFeatureExtractor()
        metrics = extractor.extract_metrics_from_df(df)
        self.assertIn('sharpe', metrics)

        # 3. Vector feature extraction
        input_vec = extractor.metrics_to_feature_vector(metrics)
        self.assertEqual(len(input_vec), 4)

        # 4. Predict risk using Deep Neuro-Fuzzy Net
        model = DeepNeuroFuzzyRiskModel()
        risk_score, category, predicted_mdd, fuzzy_labels = model.predict_risk(input_vec)

        # 5. Assert valid outputs
        self.assertGreaterEqual(risk_score, 0.0)
        self.assertLessEqual(risk_score, 100.0)
        self.assertIn(category, ["Low", "Moderate", "High", "Extreme"])
        self.assertGreaterEqual(predicted_mdd, 0.0)
        self.assertLessEqual(predicted_mdd, 1.0)
        self.assertIsNotNone(fuzzy_labels)


if __name__ == '__main__':
    unittest.main()
