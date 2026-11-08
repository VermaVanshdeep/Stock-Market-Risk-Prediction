"""
Tests for Dashboard Quick-Select Stocks (10 demonstration tickers)
and Data Lineage (live / cached / synthetic).
"""

import pytest
import numpy as np
import pandas as pd
from config.settings import FEATURED_STOCKS, COMPANY_NAME_MAP
from ui import services


REQUIRED_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "WMT", "XOM", "TSLA"]


def test_quick_select_stocks_presence_and_order():
    """Verify all 10 faculty demonstration stocks are present in FEATURED_STOCKS."""
    assert len(FEATURED_STOCKS) >= 10
    for ticker in REQUIRED_TICKERS:
        assert ticker in FEATURED_STOCKS[:10], f"Expected {ticker} in the first 10 FEATURED_STOCKS"
    assert FEATURED_STOCKS[:10] == REQUIRED_TICKERS


def test_company_name_resolution():
    """Verify company names resolve accurately for all 10 stocks and common queries."""
    expected_names = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "NVDA": "NVIDIA Corporation",
        "GOOGL": "Alphabet Inc. Class A",
        "AMZN": "Amazon.com Inc.",
        "META": "Meta Platforms Inc.",
        "JPM": "JPMorgan Chase",
        "WMT": "Walmart",
        "XOM": "ExxonMobil",
        "TSLA": "Tesla",
    }
    for ticker, expected_name in expected_names.items():
        res = services.resolve_free_text(ticker)
        assert res["ticker"] == ticker
        assert res["name"] == expected_name


def test_alias_and_search_resolution():
    """Verify natural language search queries resolve to the correct stock."""
    query_mapping = {
        "jpmorgan": "JPM",
        "JPMorgan Chase": "JPM",
        "walmart": "WMT",
        "Walmart": "WMT",
        "exxon": "XOM",
        "ExxonMobil": "XOM",
        "Exxon Mobil": "XOM",
        "tesla": "TSLA",
        "Tesla": "TSLA",
        "nvidia": "NVDA",
        "NVIDIA": "NVDA",
        "microsoft": "MSFT",
        "Microsoft": "MSFT",
        "amazon": "AMZN",
        "Amazon": "AMZN",
        "meta": "META",
        "apple": "AAPL",
    }
    for query, expected_ticker in query_mapping.items():
        res = services.resolve_free_text(query)
        assert res["ticker"] == expected_ticker, f"Query '{query}' resolved to {res['ticker']}, expected {expected_ticker}"


def test_fetch_stock_with_meta_contract():
    """Verify fetch_stock_with_meta returns (df, status_code, status_label)."""
    df, status_code, status_label = services.fetch_stock_with_meta("AAPL", "6mo")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) >= 30
    assert status_code in ["live", "cached", "synthetic"]
    assert isinstance(status_label, str)
    assert len(status_label) > 0


def test_financial_metrics_recalculation_per_stock():
    """Verify financial features (Sharpe, MDD, IV, HV) are computed distinctly per stock."""
    extractor = services.get_extractor()
    metrics_by_ticker = {}
    for ticker in ["NVDA", "JPM", "WMT", "XOM"]:
        df, _ = services.fetch_stock(ticker, "6mo")
        m = extractor.extract_metrics_from_df(df)
        metrics_by_ticker[ticker] = m
        assert "sharpe" in m
        assert "drawdown" in m
        assert "implied_vol" in m
        assert "hist_vol" in m

    # Verify they are not identical clones
    sharpes = [m["sharpe"] for m in metrics_by_ticker.values()]
    assert len(set(np.round(sharpes, 2))) > 1, "Stocks should have distinct Sharpe ratios"


def test_invalid_ticker_handling():
    """Verify invalid ticker falls back to synthetic or handles gracefully without crashing."""
    df, is_synthetic = services.fetch_stock("INVALID_TICKER_999XYZ", "6mo")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert is_synthetic is True
