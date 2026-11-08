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
    'batch_size': 32,
    'sequence_length': 20      # Metric window fed to the BiLSTM branch
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
    # Mega-cap technology
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. Class A",
    "GOOG": "Alphabet Inc. Class C",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla",
    "NFLX": "Netflix Inc.",
    "AMD": "Advanced Micro Devices Inc.",
    "INTC": "Intel Corporation",
    "CRM": "Salesforce Inc.",
    "ORCL": "Oracle Corporation",
    "ADBE": "Adobe Inc.",
    "CSCO": "Cisco Systems Inc.",
    "QCOM": "QUALCOMM Incorporated",
    "TXN": "Texas Instruments Inc.",
    "AVGO": "Broadcom Inc.",
    "MU": "Micron Technology Inc.",
    "IBM": "International Business Machines",
    "PLTR": "Palantir Technologies",
    "SHOP": "Shopify Inc.",
    "UBER": "Uber Technologies",
    "ABNB": "Airbnb Inc.",
    "SNOW": "Snowflake Inc.",
    "SQ": "Block Inc.",
    "PYPL": "PayPal Holdings",
    "INTU": "Intuit Inc.",
    "NOW": "ServiceNow Inc.",
    "PANW": "Palo Alto Networks",
    "CRWD": "CrowdStrike Holdings",
    "ASML": "ASML Holding N.V.",
    "TSM": "Taiwan Semiconductor Mfg.",
    "SAP": "SAP SE",
    "NVO": "Novo Nordisk A/S",
    # Consumer / retail / staples
    "WMT": "Walmart",
    "COST": "Costco Wholesale Corp.",
    "HD": "Home Depot Inc.",
    "LOW": "Lowe's Companies Inc.",
    "TGT": "Target Corporation",
    "MCD": "McDonald's Corporation",
    "SBUX": "Starbucks Corporation",
    "NKE": "NIKE Inc.",
    "DIS": "Walt Disney Company",
    "PG": "Procter & Gamble Co.",
    "KO": "Coca-Cola Company",
    "PEP": "PepsiCo Inc.",
    "PM": "Philip Morris International",
    "MO": "Altria Group Inc.",
    "MDLZ": "Mondelez International",
    "KMB": "Kimberly-Clark Corp.",
    "CL": "Colgate-Palmolive",
    # Healthcare / biotech
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group Inc.",
    "LLY": "Eli Lilly and Company",
    "PFE": "Pfizer Inc.",
    "ABBV": "AbbVie Inc.",
    "MRK": "Merck & Co. Inc.",
    "TMO": "Thermo Fisher Scientific",
    "AMGN": "Amgen Inc.",
    "GILD": "Gilead Sciences Inc.",
    "ISRG": "Intuitive Surgical Inc.",
    "VRTX": "Vertex Pharmaceuticals",
    "REGN": "Regeneron Pharmaceuticals",
    # Financials
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America Corp.",
    "WFC": "Wells Fargo & Company",
    "GS": "Goldman Sachs Group",
    "MS": "Morgan Stanley",
    "V": "Visa Inc.",
    "MA": "Mastercard Incorporated",
    "AXP": "American Express Co.",
    "BRK.B": "Berkshire Hathaway Inc.",
    "SCHW": "Charles Schwab Corp.",
    "BLK": "BlackRock Inc.",
    "SPGI": "S&P Global Inc.",
    # Energy / industrials / materials
    "XOM": "ExxonMobil",
    "CVX": "Chevron Corporation",
    "COP": "ConocoPhillips",
    "SLB": "Schlumberger N.V.",
    "BA": "Boeing Company",
    "CAT": "Caterpillar Inc.",
    "GE": "General Electric Company",
    "HON": "Honeywell International",
    "LMT": "Lockheed Martin Corp.",
    "RTX": "RTX Corporation",
    "UPS": "United Parcel Service",
    "DE": "Deere & Company",
    "LMTX": "Lockheed Martin Corp.",
    "MMM": "3M Company",
    "LIN": "Linde plc",
    "FCX": "Freeport-McMoRan Inc.",
    "NEM": "Newmont Corporation",
    # Utilities / real estate / comms
    "NEE": "NextEra Energy Inc.",
    "DUK": "Duke Energy Corporation",
    "SO": "Southern Company",
    "AMT": "American Tower Corp.",
    "PLD": "Prologis Inc.",
    "SPG": "Simon Property Group",
    "T": "AT&T Inc.",
    "VZ": "Verizon Communications",
    "CMCSA": "Comcast Corporation",
    "TMUS": "T-Mobile US Inc.",
    # Funds
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq-100 ETF",
    "IWM": "Russell 2000 ETF",
    "DIA": "Dow Jones Industrial Average ETF",
    "VOO": "Vanguard S&P 500 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "GLD": "SPDR Gold Shares",
    "ARKK": "ARK Innovation ETF",
    # International ADRs
    "SHEL": "Shell plc",
    "BP": "BP p.l.c.",
    "TOYOF": "Toyota Motor Corporation",
    "SONY": "Sony Group Corporation",
    "NKEKY": "NIKE Inc. ADR",
    "SAP": "SAP SE",
    "SNY": "Sanofi S.A.",
    "NVS": "Novartis AG",
    "TM": "Toyota Motor Corp.",
    "UL": "Unilever PLC",
    "HSBA": "HSBC Holdings plc",
    # Indian Stocks (NSE)
    "RELIANCE.NS": "Reliance Industries Ltd.",
    "TCS.NS": "Tata Consultancy Services Ltd.",
    "INFY.NS": "Infosys Ltd.",
    "HDFCBANK.NS": "HDFC Bank Ltd.",
    "ICICIBANK.NS": "ICICI Bank Ltd.",
    "SBIN.NS": "State Bank of India",
    "ITC.NS": "ITC Ltd.",
    "TATAMOTORS.NS": "Tata Motors Ltd.",
}

# Which Google Finance exchange carries each listing (used for live quotes).
TICKER_EXCHANGE = {
    "BRK.B": "NYSE", "BRK-B": "NYSE", "SPY": "NYSEARCA", "QQQ": "NASDAQ",
    "IWM": "NYSEARCA", "DIA": "NYSEARCA", "VOO": "NYSEARCA", "VTI": "NYSEARCA",
    "TLT": "NASDAQ", "GLD": "NYSEARCA", "ARKK": "NYSEARCA",
    "JPM": "NYSE", "V": "NYSE", "MA": "NYSE", "AXP": "NYSE", "GS": "NYSE",
    "MS": "NYSE", "BLK": "NYSE", "SPGI": "NYSE", "SCHW": "NYSE",
    "WMT": "NYSE", "HD": "NYSE", "LOW": "NYSE", "TGT": "NYSE", "MCD": "NYSE",
    "NKE": "NYSE", "PG": "NYSE", "KO": "NYSE", "PEP": "NYSE", "PM": "NYSE",
    "MO": "NYSE", "MDLZ": "NYSE", "KMB": "NYSE", "CL": "NYSE",
    "JNJ": "NYSE", "LLY": "NYSE", "PFE": "NYSE", "ABBV": "NYSE", "MRK": "NYSE",
    "TMO": "NYSE", "AMGN": "NYSE", "GILD": "NYSE", "ISRG": "NASDAQ",
    "VRTX": "NASDAQ", "REGN": "NASDAQ",
    "XOM": "NYSE", "CVX": "NYSE", "COP": "NYSE", "SLB": "NYSE",
    "BA": "NYSE", "CAT": "NYSE", "GE": "NYSE", "HON": "NYSE", "LMT": "NYSE",
    "RTX": "NYSE", "UPS": "NYSE", "DE": "NYSE", "MMM": "NYSE", "LIN": "NYSE",
    "FCX": "NYSE", "NEM": "NYSE",
    "NEE": "NYSE", "DUK": "NYSE", "SO": "NYSE",
    "AMT": "NYSE", "PLD": "NYSE", "SPG": "NYSE",
    "T": "NYSE", "VZ": "NYSE", "CMCSA": "NASDAQ", "TMUS": "NASDAQ",
    "BAC": "NYSE", "WFC": "NYSE", "SNY": "NYSE", "NVS": "NYSE",
    "SAP": "NYSE", "ASML": "NASDAQ", "NVO": "NYSE", "TM": "NYSE",
    "TSLA": "NASDAQ",
    # Indian Listings (NSE)
    "RELIANCE.NS": "NSE",
    "TCS.NS": "NSE",
    "INFY.NS": "NSE",
    "HDFCBANK.NS": "NSE",
    "ICICIBANK.NS": "NSE",
    "SBIN.NS": "NSE",
    "ITC.NS": "NSE",
    "TATAMOTORS.NS": "NSE",
}

DEFAULT_EXCHANGE = "NASDAQ"

# Some listings are quoted with punctuation but fetched with a dash.
TICKER_ALIASES = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}

# Popular presets shown in the dashboard quick-select picker (10 demonstration stocks).
FEATURED_STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    "JPM", "WMT", "XOM", "TSLA",
]

# Major Indian benchmark presets (NSE listed).
NSE_FEATURED_STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "TATAMOTORS.NS",
]

DEFAULT_STOCKS = [
    "Apple Inc. (AAPL)",
    "Microsoft Corporation (MSFT)",
    "NVIDIA Corporation (NVDA)",
    "Alphabet Inc. Class A (GOOGL)",
    "Amazon.com Inc. (AMZN)",
    "Meta Platforms Inc. (META)",
    "JPMorgan Chase (JPM)",
    "Walmart (WMT)",
    "ExxonMobil (XOM)",
    "Tesla (TSLA)",
    "S&P 500 ETF (SPY)",
    "Nasdaq-100 ETF (QQQ)",
    "Costco Wholesale Corp. (COST)",
    # Indian market selections
    "Reliance Industries Ltd. (RELIANCE.NS)",
    "Tata Consultancy Services Ltd. (TCS.NS)",
    "Infosys Ltd. (INFY.NS)",
    "HDFC Bank Ltd. (HDFCBANK.NS)",
    "ICICI Bank Ltd. (ICICIBANK.NS)",
    "State Bank of India (SBIN.NS)",
    "ITC Ltd. (ITC.NS)",
    "Tata Motors Ltd. (TATAMOTORS.NS)",
]


def get_currency_config(ticker: str) -> dict:
    """
    Central currency configuration resolver for market assets.
    Returns:
      - 'code': currency ISO code ('INR' or 'USD')
      - 'symbol': display currency symbol ('₹' or '$')
      - 'locale': formatting locale tag ('en_IN' or 'en_US')
    """
    sym = (ticker or "").strip().upper()
    if sym.endswith(".NS") or sym.endswith(".BO"):
        return {
            "code": "INR",
            "symbol": "₹",
            "locale": "en_IN",
        }
    return {
        "code": "USD",
        "symbol": "$",
        "locale": "en_US",
    }
