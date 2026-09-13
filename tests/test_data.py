"""
Unit tests for Synthetic Stock Generator and Stock DataLoader.
"""

import unittest
import pandas as pd
from data.data_loader import SyntheticStockGenerator, StockDataLoader
from data.feature_extractor import RiskFeatureExtractor


class TestDataModule(unittest.TestCase):

    def test_synthetic_stock_generator(self):
        df = SyntheticStockGenerator.generate_stock_data(ticker="TEST_SYNTH", days=100, seed=123)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        self.assertTrue(required_cols.issubset(df.columns))
        # Ensure prices are positive numbers
        self.assertTrue((df['Close'] > 0).all())

    def test_stock_data_loader_offline_mode(self):
        loader = StockDataLoader(use_offline_fallback=True)
        # Fetch invalid ticker to force fallback to synthetic generator
        df, is_synthetic = loader.fetch_stock_data("NONEXISTENT_TICKER_XYZ_99", period="1y")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(is_synthetic)
        self.assertGreater(len(df), 30)

    def test_feature_extractor(self):
        extractor = RiskFeatureExtractor()
        df = SyntheticStockGenerator.generate_stock_data(days=100, seed=42)
        metrics = extractor.extract_metrics_from_df(df)
        
        vec = extractor.metrics_to_feature_vector(metrics)
        self.assertEqual(len(vec), 4)
        
        norm_vec = extractor.normalize_feature_vector(vec)
        self.assertEqual(len(norm_vec), 4)
        self.assertTrue((norm_vec >= 0.0).all() and (norm_vec <= 1.0).all())


if __name__ == '__main__':
    unittest.main()
