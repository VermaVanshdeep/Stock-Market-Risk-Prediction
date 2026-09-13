"""
Feature Extraction Engine.
Extracts normalized financial feature vectors and historical sequences from price data
for Soft Computing Fuzzy Fuzzification and Deep Learning input layers.
Supports both PyTorch and pure NumPy fallback mode.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union
from utils.financial_metrics import calculate_all_risk_metrics

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class RiskFeatureExtractor:
    """
    Transforms stock market time-series into feature arrays and PyTorch Tensors / NumPy arrays.
    Features: [Sharpe Ratio, Maximum Drawdown, Implied Volatility, Historical Volatility]
    """

    def __init__(self):
        self.feature_bounds = {
            'sharpe': (-2.0, 4.0),
            'drawdown': (0.0, 0.8),
            'implied_vol': (0.05, 1.0),
            'hist_vol': (0.05, 1.0)
        }

    def extract_metrics_from_df(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extracts financial risk metrics dictionary from a stock DataFrame."""
        return calculate_all_risk_metrics(df)

    def metrics_to_feature_vector(self, metrics: Dict[str, float]) -> np.ndarray:
        """
        Converts metrics dictionary to 1D numpy array:
        [sharpe, drawdown, implied_vol, hist_vol]
        """
        vector = np.array([
            metrics.get('sharpe', 0.0),
            metrics.get('drawdown', 0.0),
            metrics.get('implied_vol', 0.2),
            metrics.get('hist_vol', 0.2)
        ], dtype=np.float32)
        return vector

    def normalize_feature_vector(self, vector: np.ndarray) -> np.ndarray:
        """Min-max normalizes feature vector to [0, 1] domain."""
        keys = ['sharpe', 'drawdown', 'implied_vol', 'hist_vol']
        norm_vec = np.zeros_like(vector, dtype=np.float32)

        for idx, key in enumerate(keys):
            min_val, max_val = self.feature_bounds[key]
            norm_val = (vector[idx] - min_val) / (max_val - min_val + 1e-8)
            norm_vec[idx] = np.clip(norm_val, 0.0, 1.0)

        return norm_vec

    def create_sequence_dataset(self, df: pd.DataFrame, sequence_length: int = 30):
        """
        Creates rolling sequence features for Deep Neural Network training.
        """
        prices = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
        log_rets = np.log(prices / prices.shift(1)).fillna(0.0).values
        
        hv_rolling = pd.Series(log_rets).rolling(21).std().fillna(0.0).values * np.sqrt(252)
        mdd_rolling = ((pd.Series(prices).cummax() - prices) / pd.Series(prices).cummax()).fillna(0.0).values
        
        features = np.column_stack([
            log_rets,
            hv_rolling,
            mdd_rolling,
            hv_rolling * 1.15
        ])

        num_samples = len(df) - sequence_length
        if num_samples <= 0:
            raise ValueError(f"DataFrame length ({len(df)}) is smaller than sequence_length ({sequence_length}).")

        X_list = []
        y_list = []

        for i in range(num_samples):
            seq = features[i : i + sequence_length]
            future_prices = prices.values[i + sequence_length : min(len(prices), i + sequence_length + 10)]
            curr_price = prices.values[i + sequence_length - 1]
            future_max_dd = (curr_price - np.min(future_prices)) / curr_price if len(future_prices) > 0 else 0.0
            
            target_risk_score = min(100.0, max(0.0, future_max_dd * 200.0 + features[i + sequence_length - 1, 1] * 100.0))
            
            X_list.append(seq)
            y_list.append(target_risk_score)

        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)[:, np.newaxis]

        if TORCH_AVAILABLE:
            return torch.tensor(X_arr), torch.tensor(y_arr)
        return X_arr, y_arr
