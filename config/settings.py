"""
Global Configuration Settings for Stock Market Risk Prediction System.
Defines risk metric parameters, fuzzy set bounds, neural network hyperparameters,
and genetic algorithm configuration.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_cache"
DATA_DIR.mkdir(exist_ok=True)

# Financial Risk Parameters
RISK_FREE_RATE = 0.04           # 4% annual risk-free rate
ANNUALIZATION_FACTOR = 252       # Trading days per year
HV_WINDOW = 21                  # 21-day historical volatility window
DRAWDOWN_WINDOW = 252           # 1-year max drawdown window
CONFIDENCE_LEVEL_VAR = 0.95     # 95% Value at Risk confidence level

# Fuzzy Logic Membership Settings
FUZZY_INPUT_SPECS = {
    'sharpe': {
        'low': {'mean': 0.0, 'sigma': 0.5},
        'medium': {'mean': 1.0, 'sigma': 0.5},
        'high': {'mean': 2.0, 'sigma': 0.5}
    },
    'drawdown': {
        'low': {'mean': 0.05, 'sigma': 0.05},
        'medium': {'mean': 0.20, 'sigma': 0.08},
        'high': {'mean': 0.40, 'sigma': 0.10}
    },
    'implied_vol': {
        'low': {'mean': 0.12, 'sigma': 0.04},
        'medium': {'mean': 0.25, 'sigma': 0.06},
        'high': {'mean': 0.45, 'sigma': 0.10}
    },
    'hist_vol': {
        'low': {'mean': 0.12, 'sigma': 0.04},
        'medium': {'mean': 0.25, 'sigma': 0.06},
        'high': {'mean': 0.45, 'sigma': 0.10}
    }
}

# Risk Level Classification Categories
RISK_CATEGORIES = ["Low", "Moderate", "High", "Extreme"]
RISK_COLOR_MAP = {
    "Low": "blue",
    "Moderate": "blue",
    "High": "blue",
    "Extreme": "blue"
}

# Deep Neural Network Hyperparameters
MODEL_CONFIG = {
    'input_dim': 4,            # [Sharpe, MaxDrawdown, ImpliedVol, HistVol]
    'hidden_dim': 64,
    'bilstm_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.003,
    'epochs': 50,
    'batch_size': 32
}

# Soft Computing: Genetic Algorithm Hyperparameters
GA_CONFIG = {
    'population_size': 20,
    'generations': 15,
    'mutation_rate': 0.15,
    'crossover_rate': 0.8,
    'elite_count': 2
}

# Company Name Mapping & Benchmark Stock List
COMPANY_NAME_MAP = {
    "AAPL": "Apple Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corporation",
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq-100 ETF",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc."
}

DEFAULT_STOCKS = [
    "Apple Inc. (AAPL)",
    "Tesla Inc. (TSLA)",
    "NVIDIA Corporation (NVDA)",
    "S&P 500 ETF (SPY)",
    "Nasdaq-100 ETF (QQQ)",
    "Microsoft Corporation (MSFT)",
    "Amazon.com Inc. (AMZN)"
]
