"""
Unit tests for Financial Risk Metrics calculation utilities.
"""

import unittest
import numpy as np
import pandas as pd
from utils.financial_metrics import (
    calculate_historical_volatility,
    calculate_implied_volatility_proxy,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_value_at_risk,
    calculate_all_risk_metrics
)


class TestFinancialMetrics(unittest.TestCase):

    def setUp(self):
        # Create deterministic synthetic price series
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100)
        # Price starting at 100 with known drawdown and volatility
        prices = 100.0 * np.cumprod(1.0 + np.random.normal(0.001, 0.02, 100))
        # Force a peak at day 30 and trough at day 60 for explicit drawdown test
        prices[30] = 150.0
        prices[60] = 100.0
        
        self.prices_series = pd.Series(prices, index=dates)
        self.df_ohlc = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': 1000000
        }, index=dates)

    def test_calculate_max_drawdown(self):
        mdd = calculate_max_drawdown(self.prices_series)
        self.assertIsInstance(mdd, float)
        self.assertGreaterEqual(mdd, 0.0)
        self.assertLessEqual(mdd, 1.0)
        # Expected drawdown between day 30 (150) and day 60 (100) is (150-100)/150 = 33.3%
        self.assertGreaterEqual(mdd, 0.30)

    def test_calculate_historical_volatility(self):
        hv = calculate_historical_volatility(self.prices_series, window=21)
        self.assertIsInstance(hv, float)
        self.assertGreater(hv, 0.0)

    def test_calculate_implied_volatility_proxy(self):
        iv = calculate_implied_volatility_proxy(self.df_ohlc)
        self.assertIsInstance(iv, float)
        self.assertGreater(iv, 0.0)

    def test_calculate_sharpe_ratio(self):
        sharpe = calculate_sharpe_ratio(self.prices_series)
        self.assertIsInstance(sharpe, float)

    def test_calculate_sortino_ratio(self):
        sortino = calculate_sortino_ratio(self.prices_series)
        self.assertIsInstance(sortino, float)

    def test_calculate_value_at_risk(self):
        var95 = calculate_value_at_risk(self.prices_series, alpha=0.95)
        self.assertIsInstance(var95, float)
        self.assertGreaterEqual(var95, 0.0)

    def test_calculate_all_risk_metrics(self):
        metrics = calculate_all_risk_metrics(self.df_ohlc)
        expected_keys = {'sharpe', 'drawdown', 'implied_vol', 'hist_vol', 'sortino', 'var_95'}
        self.assertEqual(set(metrics.keys()), expected_keys)


if __name__ == '__main__':
    unittest.main()
