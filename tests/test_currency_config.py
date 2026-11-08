"""
Tests for Central Currency Configuration and Plotly Chart Monetary Formatting.
Validates get_currency_config() for NSE/BSE and US tickers,
as well as Plotly chart layouts, hovertemplates, and tick prefixes.
"""

import numpy as np
import pandas as pd
import pytest

from config.settings import get_currency_config
from ui import charts


def test_get_currency_config_nse_and_bse():
    """Verify that .NS and .BO tickers resolve to INR, ₹, en_IN."""
    for ticker in ["TCS.NS", "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.BO", "TEST.NS"]:
        cfg = get_currency_config(ticker)
        assert cfg["code"] == "INR", f"{ticker} code should be INR"
        assert cfg["symbol"] == "₹", f"{ticker} symbol should be ₹"
        assert cfg["locale"] == "en_IN", f"{ticker} locale should be en_IN"


def test_get_currency_config_us_tickers():
    """Verify that US tickers resolve to USD, $, en_US."""
    for ticker in ["AAPL", "MSFT", "NVDA", "SPY", "TSLA", "GOOGL"]:
        cfg = get_currency_config(ticker)
        assert cfg["code"] == "USD", f"{ticker} code should be USD"
        assert cfg["symbol"] == "$", f"{ticker} symbol should be $"
        assert cfg["locale"] == "en_US", f"{ticker} locale should be en_US"


def _create_sample_df(n=50, base_price=3000.0):
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.normal(0, 10, n))
    return pd.DataFrame({
        "Open": prices * 0.99,
        "High": prices * 1.01,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": [100000] * n,
    }, index=dates)


def test_tcs_chart_formatting_has_inr_and_never_usd():
    """
    Mandatory verification for TCS.NS:
    Every monetary display (axis tickprefix, hovertemplate, annotation) must use ₹, never $.
    """
    df = _create_sample_df(base_price=3500.0)
    metrics = {"drawdown": 0.15, "implied_vol": 0.22, "hist_vol": 0.25, "sharpe": 1.2}

    # 1. Line chart
    fig_line = charts.price_drawdown_chart(df, "TCS.NS", metrics, chart_type="Line")

    # Check yaxis tickprefix on closing price subplot
    assert fig_line.layout.yaxis.tickprefix == "₹", "Yaxis tickprefix for TCS.NS must be ₹"
    assert fig_line.layout.yaxis.tickprefix != "$", "Yaxis tickprefix for TCS.NS must not be $"

    # Check Scatter hovertemplate
    scatter_trace = [t for t in fig_line.data if t.name == "Close"][0]
    assert "₹" in scatter_trace.hovertemplate, "Scatter hovertemplate must contain ₹"
    assert "$" not in scatter_trace.hovertemplate, "Scatter hovertemplate must not contain $"

    # Check subplot title
    subplot_titles = [ann.text for ann in fig_line.layout.annotations if "Closing Price" in ann.text]
    assert len(subplot_titles) > 0
    assert "₹" in subplot_titles[0]
    assert "$" not in subplot_titles[0]

    # Check price annotation
    price_annotations = [ann.text for ann in fig_line.layout.annotations if "Last:" in ann.text]
    assert len(price_annotations) > 0
    assert "₹" in price_annotations[0]
    assert "$" not in price_annotations[0]

    # 2. Candlestick chart
    fig_candle = charts.price_drawdown_chart(df, "TCS.NS", metrics, chart_type="Candlestick")
    candle_trace = [t for t in fig_candle.data if t.name == "TCS.NS"][0]
    assert "₹" in candle_trace.hovertemplate, "Candlestick hovertemplate must contain ₹"
    assert "$" not in candle_trace.hovertemplate, "Candlestick hovertemplate must not contain $"

    # 3. Drawdown axis (Requirement 6: no currency formatting on percentage/ratio metrics)
    assert getattr(fig_line.layout.yaxis2, "tickprefix", "") == ""
    assert fig_line.layout.yaxis2.ticksuffix == "%"


def test_aapl_chart_formatting_has_usd_and_never_inr():
    """
    Mandatory verification for AAPL:
    Every monetary display (axis tickprefix, hovertemplate, annotation) must use $, never ₹.
    """
    df = _create_sample_df(base_price=220.0)
    metrics = {"drawdown": 0.12, "implied_vol": 0.20, "hist_vol": 0.18, "sharpe": 1.5}

    # 1. Line chart
    fig_line = charts.price_drawdown_chart(df, "AAPL", metrics, chart_type="Line")

    # Check yaxis tickprefix on closing price subplot
    assert fig_line.layout.yaxis.tickprefix == "$", "Yaxis tickprefix for AAPL must be $"
    assert fig_line.layout.yaxis.tickprefix != "₹", "Yaxis tickprefix for AAPL must not be ₹"

    # Check Scatter hovertemplate
    scatter_trace = [t for t in fig_line.data if t.name == "Close"][0]
    assert "$" in scatter_trace.hovertemplate, "Scatter hovertemplate must contain $"
    assert "₹" not in scatter_trace.hovertemplate, "Scatter hovertemplate must not contain ₹"

    # Check subplot title
    subplot_titles = [ann.text for ann in fig_line.layout.annotations if "Closing Price" in ann.text]
    assert len(subplot_titles) > 0
    assert "$" in subplot_titles[0]
    assert "₹" not in subplot_titles[0]

    # Check price annotation
    price_annotations = [ann.text for ann in fig_line.layout.annotations if "Last:" in ann.text]
    assert len(price_annotations) > 0
    assert "$" in price_annotations[0]
    assert "₹" not in price_annotations[0]

    # 2. Candlestick chart
    fig_candle = charts.price_drawdown_chart(df, "AAPL", metrics, chart_type="Candlestick")
    candle_trace = [t for t in fig_candle.data if t.name == "AAPL"][0]
    assert "$" in candle_trace.hovertemplate, "Candlestick hovertemplate must contain $"
    assert "₹" not in candle_trace.hovertemplate, "Candlestick hovertemplate must not contain ₹"

    # 3. Drawdown axis
    assert getattr(fig_line.layout.yaxis2, "tickprefix", "") == ""
    assert fig_line.layout.yaxis2.ticksuffix == "%"

