"""
Stock Data Loader & Synthetic Stock Generator Engine.
Provides real-time market data downloading via yfinance with automatic offline fallback
to a high-fidelity synthetic price path generator using Geometric Brownian Motion (GBM)
and Merton Jump Diffusion with Heston stochastic volatility.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional
from rich.console import Console

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    yf = None
    YFINANCE_AVAILABLE = False

console = Console(force_terminal=True, legacy_windows=False)



class SyntheticStockGenerator:
    """
    Generates realistic synthetic stock OHLCV data using Merton Jump Diffusion (MJD)
    and stochastic volatility dynamics for offline test execution and benchmark testing.
    """

    @staticmethod
    def generate_stock_data(
        ticker: str = "SYNTH_STOCK",
        days: int = 500,
        start_price: float = 150.0,
        mu: float = 0.08,             # Annual drift (8%)
        sigma: float = 0.25,          # Volatility (25%)
        jump_lambda: float = 0.05,    # Jump frequency
        jump_mu: float = -0.05,       # Expected jump size (-5% drop)
        jump_sigma: float = 0.10,     # Jump volatility
        seed: Optional[int] = 42
    ) -> pd.DataFrame:
        """Generates synthetic OHLCV DataFrame."""
        if seed is not None:
            np.random.seed(seed)

        dt = 1.0 / 252.0  # 1 day in years
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')

        # Generate GBM + Jump components
        n = days
        prices = np.zeros(n)
        prices[0] = start_price

        # Heston Stochastic Volatility
        vol = np.zeros(n)
        vol[0] = sigma

        for t in range(1, n):
            # Volatility mean reversion
            vol[t] = np.abs(vol[t-1] + 2.0 * (sigma - vol[t-1]) * dt + 0.1 * np.sqrt(dt) * np.random.normal())
            
            # Jump component
            jump_occured = np.random.poisson(jump_lambda * dt) > 0
            jump_factor = np.exp(np.random.normal(jump_mu, jump_sigma)) if jump_occured else 1.0
            
            # Price step
            z = np.random.normal()
            growth = np.exp((mu - 0.5 * vol[t]**2) * dt + vol[t] * np.sqrt(dt) * z) * jump_factor
            prices[t] = max(1.0, prices[t-1] * growth)

        # Generate Open, High, Low, Close, Volume
        close_prices = prices
        high_prices = close_prices * (1.0 + np.abs(np.random.normal(0, 0.012, n)))
        low_prices = close_prices * (1.0 - np.abs(np.random.normal(0, 0.012, n)))
        open_prices = low_prices + (high_prices - low_prices) * np.random.uniform(0.1, 0.9, n)
        volume = np.random.randint(1000000, 50000000, size=n)

        df = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        }, index=dates)

        return df


class StockDataLoader:
    """
    DataLoader with automatic online/offline fallback handling.
    Attempts to download real ticker data via yfinance, falling back to synthetic generator on network failure.
    """

    def __init__(self, use_offline_fallback: bool = True):
        self.use_offline_fallback = use_offline_fallback

    def fetch_stock_data(self, ticker: str, period: str = "2y") -> Tuple[pd.DataFrame, bool]:
        """
        Fetches stock data for ticker.
        Returns (df, is_synthetic_flag)
        """
        ticker_clean = ticker.strip().upper()
        console.print(f"[bold cyan]Fetching market data for ticker '{ticker_clean}'...[/bold cyan]")

        try:
            if not YFINANCE_AVAILABLE:
                raise ValueError("yfinance library not loaded or unavailable.")
                
            df = yf.download(ticker_clean, period=period, progress=False)
            if df is not None and not df.empty and len(df) > 30:
                # Handle MultiIndex columns if yfinance returns them
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                console.print(f"[bold green]Successfully retrieved {len(df)} price bars for {ticker_clean} via yfinance![/bold green]")
                return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(), False
            else:
                raise ValueError("Empty or insufficient data returned from yfinance.")

        except Exception as e:
            console.print(f"[yellow]Network fetch note: {str(e)}.[/yellow]")
            if self.use_offline_fallback:

                console.print(f"[bold yellow]Switching to high-fidelity synthetic market data generator for {ticker_clean}...[/bold yellow]")
                # Vary synthetic parameters by ticker hash to give unique characteristics per stock
                seed = abs(hash(ticker_clean)) % 10000
                vol_map = {"TSLA": 0.45, "NVDA": 0.50, "AAPL": 0.22, "SPY": 0.15, "QQQ": 0.20}
                sigma = vol_map.get(ticker_clean, 0.30)
                
                df_synth = SyntheticStockGenerator.generate_stock_data(
                    ticker=ticker_clean,
                    days=500,
                    sigma=sigma,
                    seed=seed
                )
                return df_synth, True
            else:
                raise e
