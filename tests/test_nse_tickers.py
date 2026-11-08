"""
Tests for NSE Indian stock support in Stock Market Risk Prediction.
Validates mappings, name resolution, suggestions, yfinance fetching,
rejection of synthetic fallback for NSE tickers, and prediction pipeline.
"""

from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from config.settings import (
    COMPANY_NAME_MAP,
    DEFAULT_STOCKS,
    NSE_FEATURED_STOCKS,
    TICKER_EXCHANGE,
)
from data.data_loader import StockDataLoader
from data.feature_extractor import RiskFeatureExtractor
from models.deep_neuro_fuzzy_net import DeepNeuroFuzzyRiskModel
from ui import services

TARGET_NSE_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "TATAMOTORS.NS",
]


def test_nse_stocks_presence_in_config():
    """All 8 target NSE tickers must exist in NSE_FEATURED_STOCKS and COMPANY_NAME_MAP."""
    assert len(NSE_FEATURED_STOCKS) == 8
    for ticker in TARGET_NSE_TICKERS:
        assert ticker in NSE_FEATURED_STOCKS, f"{ticker} missing from NSE_FEATURED_STOCKS"
        assert ticker in COMPANY_NAME_MAP, f"{ticker} missing from COMPANY_NAME_MAP"
        assert TICKER_EXCHANGE.get(ticker) == "NSE", f"{ticker} must map to exchange 'NSE'"


def test_nse_stocks_in_default_stocks():
    """All 8 target NSE tickers must be present in DEFAULT_STOCKS."""
    for ticker in TARGET_NSE_TICKERS:
        found = any(f"({ticker})" in s for s in DEFAULT_STOCKS)
        assert found, f"Ticker {ticker} not found in DEFAULT_STOCKS"


def test_nse_ticker_resolution_exact():
    """Resolving the exact .NS ticker returns the .NS ticker and full company name."""
    for ticker in TARGET_NSE_TICKERS:
        res = services.resolve_free_text(ticker)
        assert res["ticker"] == ticker
        assert res["name"] == COMPANY_NAME_MAP[ticker]
        assert res["exchange"] == "NSE"


def test_nse_ticker_resolution_aliases():
    """Resolving common aliases and natural language names maps to the correct .NS ticker."""
    alias_map = {
        "reliance": "RELIANCE.NS",
        "ril": "RELIANCE.NS",
        "Reliance Industries": "RELIANCE.NS",
        "tcs": "TCS.NS",
        "Tata Consultancy Services": "TCS.NS",
        "infy": "INFY.NS",
        "infosys": "INFY.NS",
        "hdfc": "HDFCBANK.NS",
        "hdfc bank": "HDFCBANK.NS",
        "hdfcbank": "HDFCBANK.NS",
        "icici": "ICICIBANK.NS",
        "icici bank": "ICICIBANK.NS",
        "icicibank": "ICICIBANK.NS",
        "sbi": "SBIN.NS",
        "sbin": "SBIN.NS",
        "state bank": "SBIN.NS",
        "state bank of india": "SBIN.NS",
        "itc": "ITC.NS",
        "tata motors": "TATAMOTORS.NS",
        "tatamotors": "TATAMOTORS.NS",
    }
    for query, expected_ticker in alias_map.items():
        res = services.resolve_free_text(query)
        assert res["ticker"] == expected_ticker, f"Query '{query}' resolved to {res['ticker']}, expected {expected_ticker}"


def test_nse_parenthesized_wrapper_resolution():
    """Formatted company strings like 'Tata Motors Ltd. (TATAMOTORS.NS)' resolve correctly."""
    for ticker in TARGET_NSE_TICKERS:
        label = f"{COMPANY_NAME_MAP[ticker]} ({ticker})"
        res = services.resolve_free_text(label)
        assert res["ticker"] == ticker, f"Label '{label}' resolved to {res['ticker']}, expected {ticker}"


def test_nse_suggestions():
    """Autocomplete suggestions return the appropriate .NS ticker when queried."""
    # When typing 'reliance'
    sugg_rel = services.company_suggestions("reliance")
    assert any(s["ticker"] == "RELIANCE.NS" for s in sugg_rel)

    # When typing 'tcs'
    sugg_tcs = services.company_suggestions("tcs")
    assert any(s["ticker"] == "TCS.NS" for s in sugg_tcs)

    # When typing 'hdfc'
    sugg_hdfc = services.company_suggestions("hdfc")
    assert any(s["ticker"] == "HDFCBANK.NS" for s in sugg_hdfc)

    # When typing 'nse'
    sugg_nse = services.company_suggestions("nse")
    assert any(s["ticker"].endswith(".NS") for s in sugg_nse)


def test_nse_no_synthetic_fallback():
    """
    NSE tickers must NEVER silently fall back to synthetic data on failure.
    A clear ValueError must be raised instead.
    """
    loader = StockDataLoader(use_offline_fallback=True)

    # Mock yf.download to raise an exception
    with patch("data.data_loader.yf.download", side_effect=RuntimeError("Connection reset")):
        with pytest.raises(ValueError) as excinfo:
            loader.fetch_stock_data("RELIANCE.NS")
        assert "Synthetic fallback is disabled for NSE stocks" in str(excinfo.value)

    # If yf returns empty dataframe
    with patch("data.data_loader.yf.download", return_value=pd.DataFrame()):
        with pytest.raises(ValueError) as excinfo:
            loader.fetch_stock_data("TCS.NS")
        assert "Synthetic fallback is disabled for NSE stocks" in str(excinfo.value)


def test_us_stocks_still_use_synthetic_fallback_when_offline():
    """US stocks should retain their offline fallback behavior without regression."""
    loader = StockDataLoader(use_offline_fallback=True)
    with patch("data.data_loader.yf.download", side_effect=RuntimeError("Simulated offline")):
        df, is_synthetic = loader.fetch_stock_data("AAPL")
        assert is_synthetic is True
        assert not df.empty
        assert len(df) >= 30


def test_feature_extraction_and_prediction_on_nse_data():
    """Verify metrics calculation and hybrid prediction on simulated INR-priced stock data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    # Price path around INR 2500
    prices = 2500 + np.cumsum(np.random.normal(1.0, 20.0, 100))
    df = pd.DataFrame({
        "Open": prices * 0.99,
        "High": prices * 1.02,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": np.random.randint(100000, 5000000, size=100)
    }, index=dates)

    extractor = RiskFeatureExtractor()
    metrics = extractor.extract_metrics_from_df(df)

    assert "sharpe" in metrics
    assert "drawdown" in metrics
    assert "implied_vol" in metrics
    assert "hist_vol" in metrics
    assert not np.isnan(metrics["sharpe"])
    assert not np.isnan(metrics["drawdown"])
    assert not np.isnan(metrics["implied_vol"])
    assert not np.isnan(metrics["hist_vol"])

    vector = extractor.metrics_to_feature_vector(metrics)
    model = services.get_model()
    score, category, pred_mdd, fuzzy_labels = model.predict_risk(vector)

    assert 0.0 <= score <= 100.0
    assert category in ["Low", "Moderate", "High", "Extreme"]
    assert len(fuzzy_labels) == 4

