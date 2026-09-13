"""
Financial Risk Metrics Calculation Utilities.
Computes Sharpe Ratio, Maximum Drawdown, Implied Volatility Proxy, Historical Volatility,
Sortino Ratio, and Value at Risk (VaR).
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple
from config.settings import RISK_FREE_RATE, ANNUALIZATION_FACTOR, HV_WINDOW, CONFIDENCE_LEVEL_VAR


def calculate_daily_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily percentage returns."""
    return prices.pct_change().fillna(0.0)


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily log returns."""
    return np.log(prices / prices.shift(1)).fillna(0.0)


def calculate_historical_volatility(prices: Union[pd.Series, np.ndarray], window: int = HV_WINDOW) -> float:
    """
    Calculate annualized Historical Volatility (HV) using log returns standard deviation.
    Formula: std(log_returns) * sqrt(252)
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    log_rets = calculate_log_returns(prices)
    if len(log_rets) < 2:
        return 0.0
    
    daily_std = log_rets.iloc[-window:].std() if len(log_rets) >= window else log_rets.std()
    if np.isnan(daily_std):
        return 0.0
    return float(daily_std * np.sqrt(ANNUALIZATION_FACTOR))


def calculate_implied_volatility_proxy(df: pd.DataFrame) -> float:
    """
    Computes Garman-Klass / Parkinson Volatility Proxy for Implied Volatility (IV)
    when High/Low/Open/Close prices are provided, or option proxy estimation.
    
    Garman-Klass Formula:
    sigma^2 = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
    """
    if not isinstance(df, pd.DataFrame):
        # Fallback for 1D price array
        prices = np.array(df)
        rets = np.diff(np.log(prices)) if len(prices) > 1 else np.array([0.0])
        # Implied Vol proxy as forward volatility estimator with leverage multiplier
        return float(np.std(rets) * np.sqrt(ANNUALIZATION_FACTOR) * 1.15)

    required_cols = {'High', 'Low', 'Open', 'Close'}
    if required_cols.issubset(df.columns):
        h = df['High']
        l = df['Low']
        c = df['Close']
        o = df['Open']
        
        # Garman-Klass Volatility
        term1 = 0.5 * (np.log(h / l)) ** 2
        term2 = (2 * np.log(2) - 1) * (np.log(c / o)) ** 2
        gk_vol = np.sqrt(np.maximum(0, term1 - term2))
        ann_iv = float(gk_vol.tail(21).mean() * np.sqrt(ANNUALIZATION_FACTOR))
        return float(np.nan_to_num(ann_iv, nan=0.20))
    else:
        # Fallback to Close price returns with forward option skew adjustment factor
        prices = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
        hv = calculate_historical_volatility(prices)
        # Typically IV carries a volatility premium over HV
        return float(hv * 1.18)


def calculate_max_drawdown(prices: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate Maximum Drawdown (MDD) as a positive decimal fraction (e.g. 0.25 = 25%).
    Formula: (Peak - Trough) / Peak
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    if len(prices) == 0:
        return 0.0
    
    cumulative_max = prices.cummax()
    drawdowns = (cumulative_max - prices) / cumulative_max
    max_dd = float(drawdowns.max())
    return float(np.nan_to_num(max_dd, nan=0.0))


def calculate_sharpe_ratio(prices_or_returns: Union[pd.Series, np.ndarray], is_returns: bool = False, rf: float = RISK_FREE_RATE) -> float:
    """
    Calculate Annualized Sharpe Ratio.
    Formula: (Annualized Return - Risk Free Rate) / Annualized Volatility
    """
    if isinstance(prices_or_returns, np.ndarray):
        prices_or_returns = pd.Series(prices_or_returns)
        
    if not is_returns:
        returns = calculate_daily_returns(prices_or_returns)
    else:
        returns = prices_or_returns

    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    mean_daily_ret = returns.mean()
    std_daily_ret = returns.std()
    
    ann_return = mean_daily_ret * ANNUALIZATION_FACTOR
    ann_vol = std_daily_ret * np.sqrt(ANNUALIZATION_FACTOR)
    
    if ann_vol == 0:
        return 0.0
    
    sharpe = (ann_return - rf) / ann_vol
    return float(np.nan_to_num(sharpe, nan=0.0))


def calculate_sortino_ratio(prices_or_returns: Union[pd.Series, np.ndarray], is_returns: bool = False, rf: float = RISK_FREE_RATE) -> float:
    """
    Calculate Annualized Sortino Ratio (Downside Risk adjusted return).
    Formula: (Annualized Return - Risk Free Rate) / Annualized Downside Volatility
    """
    if isinstance(prices_or_returns, np.ndarray):
        prices_or_returns = pd.Series(prices_or_returns)
        
    if not is_returns:
        returns = calculate_daily_returns(prices_or_returns)
    else:
        returns = prices_or_returns

    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 5.0  # High safety cap

    ann_return = returns.mean() * ANNUALIZATION_FACTOR
    downside_vol = downside_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)
    
    if downside_vol == 0:
        return 0.0
        
    sortino = (ann_return - rf) / downside_vol
    return float(np.nan_to_num(sortino, nan=0.0))


def calculate_value_at_risk(prices_or_returns: Union[pd.Series, np.ndarray], is_returns: bool = False, alpha: float = CONFIDENCE_LEVEL_VAR) -> float:
    """
    Calculate 1-Day Historical Value at Risk (VaR) at specified confidence level (e.g., 95%).
    Returns positive value representing potential max % loss.
    """
    if isinstance(prices_or_returns, np.ndarray):
        prices_or_returns = pd.Series(prices_or_returns)
        
    if not is_returns:
        returns = calculate_daily_returns(prices_or_returns)
    else:
        returns = prices_or_returns

    if len(returns) == 0:
        return 0.0

    var_percentile = np.percentile(returns, (1.0 - alpha) * 100)
    return float(abs(min(0.0, var_percentile)))


def calculate_all_risk_metrics(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Dict[str, float]:
    """
    Comprehensive pipeline extractor for all financial risk metrics.
    Returns dictionary with Sharpe, Max Drawdown, Implied Volatility, Historical Volatility,
    Sortino Ratio, and VaR.
    """
    if isinstance(data, pd.DataFrame):
        prices = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
        iv = calculate_implied_volatility_proxy(data)
    else:
        prices = pd.Series(data) if isinstance(data, np.ndarray) else data
        iv = calculate_implied_volatility_proxy(prices)

    sharpe = calculate_sharpe_ratio(prices)
    mdd = calculate_max_drawdown(prices)
    hv = calculate_historical_volatility(prices)
    sortino = calculate_sortino_ratio(prices)
    var95 = calculate_value_at_risk(prices)

    return {
        'sharpe': round(sharpe, 4),
        'drawdown': round(mdd, 4),
        'implied_vol': round(iv, 4),
        'hist_vol': round(hv, 4),
        'sortino': round(sortino, 4),
        'var_95': round(var95, 4)
    }
