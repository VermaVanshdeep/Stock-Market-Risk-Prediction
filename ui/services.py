"""
Service layer for the Streamlit UI.

Thin, cache-aware wrappers around the existing data / model / optimizer engines so
that presentation code never has to know how the pipeline is wired together.
"""

import io
import json
import re
import urllib.request
import uuid
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import (
    COMPANY_NAME_MAP,
    DEFAULT_EXCHANGE,
    DEFAULT_STOCKS,
    FEATURED_STOCKS,
    RISK_CATEGORIES,
    TICKER_ALIASES,
    TICKER_EXCHANGE,
)
from data.data_loader import StockDataLoader
from data.feature_extractor import RiskFeatureExtractor
from models.deep_neuro_fuzzy_net import DeepNeuroFuzzyRiskModel
from models.fuzzy_layer import FEATURE_KEYS, TERM_LABELS, TORCH_AVAILABLE, build_rule_list

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Cached singletons                                                            #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_loader() -> StockDataLoader:
    return StockDataLoader(use_offline_fallback=True)


@st.cache_resource(show_spinner=False)
def get_extractor() -> RiskFeatureExtractor:
    return RiskFeatureExtractor()


@st.cache_resource(show_spinner=False)
def get_model() -> DeepNeuroFuzzyRiskModel:
    """
    Returns the process-wide model.

    Construction is seeded so an untrained model is deterministic, and trained
    weights are loaded from disk when they exist.
    """
    from models import train as trainer

    trainer.seed_everything()
    model = DeepNeuroFuzzyRiskModel()
    trainer.load_weights(model)
    return model


def model_is_trained() -> bool:
    from models import train as trainer

    return trainer.weights_exist()


def model_training_status() -> Dict[str, object]:
    """Describes whether the risk head is fitted, for display in the UI."""
    from models import train as trainer

    return {
        "trained": trainer.weights_exist(),
        "weights_path": str(trainer.WEIGHTS_PATH),
        "torch": trainer.TORCH_AVAILABLE,
        "universe": list(trainer.TRAINING_UNIVERSE),
    }


@st.cache_resource(show_spinner=False, max_entries=1)
def ensure_trained():
    """
    Trains and persists the model once per server process when weights are absent.

    This is what makes the headline risk score respond to input instead of
    sitting at a random constant. Cached with no arguments, so it runs at most
    once per server process.
    """
    from models import train as trainer

    model = get_model()
    if trainer.weights_exist():
        return model

    trainer.train_and_save(model, epochs=trainer.EPOCHS)
    return model


def retrain(epochs: int = 10, progress_callback=None) -> Dict[str, object]:
    """Re-runs training from scratch and overwrites the persisted weights."""
    from models import train as trainer

    trainer.seed_everything()
    model = get_model()
    history = trainer.train_and_save(
        model,
        epochs=int(epochs),
        progress_callback=progress_callback,
    )
    history["score_std"] = round(trainer.score_spread(model), 4)
    return history


def training_probe() -> Dict[str, float]:
    """Scores a spread of regimes so the UI can prove the model is responsive."""
    profiles = {
        "Defensive": (2.0, 0.05, 0.12, 0.10),
        "Balanced": (1.2, 0.15, 0.25, 0.20),
        "Stressed": (0.1, 0.40, 0.55, 0.50),
        "Crisis": (-1.2, 0.65, 0.85, 0.75),
    }
    results = {}
    for label, values in profiles.items():
        vector = np.asarray(values, dtype=np.float32)
        score, category, _, _ = get_model().predict_risk(vector)
        results[label] = {"score": float(score), "category": category}

    scores = [v["score"] for v in results.values()]
    return {
        "profiles": results,
        "min": min(scores),
        "max": max(scores),
        "spread": max(scores) - min(scores),
    }


# --------------------------------------------------------------------------- #
# Data access                                                                  #
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stock(ticker: str, period: str) -> Tuple[pd.DataFrame, bool]:
    """Returns ``(ohlc_frame, used_synthetic_fallback)`` for a ticker."""
    return get_loader().fetch_stock_data(ticker, period=period)


def clear_market_cache() -> None:
    fetch_stock.clear()
    fetch_google_sheet.clear()


# --------------------------------------------------------------------------- #
# Google Sheets source                                                        #
# --------------------------------------------------------------------------- #
SHEET_HELP = (
    "Share your sheet as **File › Share › Anyone with the link › Viewer**, then paste "
    "either the full URL, the export URL, or just the sheet ID from the URL."
)

_DATE_KEYS = ("date", "datetime", "timestamp", "time", "day", "period")
_CLOSE_KEYS = ("close", "adjclose", "adjustedclose", "closeprice", "lastclose", "price", "last", "value")
_OPEN_KEYS = ("open", "openprice")
_HIGH_KEYS = ("high", "highprice")
_LOW_KEYS = ("low", "lowprice")
_VOLUME_KEYS = ("volume", "vol", "shares")


def _normalise_sheet_url(source: str) -> str:
    """Turns a pasted link or bare sheet ID into a downloadable CSV URL."""
    text = (source or "").strip()
    if not text:
        raise ValueError("Paste a Google Sheets link or ID first.")

    if text.startswith("http") and "export?format=csv" in text:
        return text
    if text.startswith("http") and "output=csv" in text:
        return text

    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", text)
    if not match:
        if re.fullmatch(r"[a-zA-Z0-9-_]{20,}", text):
            match = None
            sheet_id = text
        else:
            raise ValueError(
                "That does not look like a Google Sheets link or ID. Expected a URL "
                "containing /spreadsheets/d/<ID>/edit or a bare sheet ID."
            )
    else:
        sheet_id = match.group(1)

    tab = re.search(r"[#&]gid=(\d+)", text)
    gid = f"&gid={tab.group(1)}" if tab else ""
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv{gid}"


def _match_column(columns: List[str], keys) -> Optional[str]:
    lookup = {str(c).strip().lower().replace(" ", "").replace("_", ""): str(c).strip()
              for c in columns}
    for key in keys:
        if key in lookup:
            return lookup[key]
    return None


def _to_datetime(series: "pd.Series") -> "pd.Series":
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
    return parsed


def normalise_ohlcv(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Coerces an arbitrary sheet into the OHLCV frame the pipeline expects.

    Returns ``(frame, notes)`` where notes describe any columns that were filled
    in, so the UI can be transparent about what was derived versus supplied.
    """
    frame = raw.copy()
    frame.columns = [str(c).strip() for c in frame.columns]

    notes: List[str] = []

    date_col = _match_column(list(frame.columns), _DATE_KEYS) or frame.columns[0]
    index = _to_datetime(frame[date_col])
    if index.isna().all():
        raise ValueError(
            f"Could not read dates from column '{date_col}'. Use YYYY-MM-DD values."
        )

    close_col = _match_column(list(frame.columns), _CLOSE_KEYS)
    if close_col is None:
        raise ValueError(
            "No closing price column found. Name one of: Close, Adj Close, Price or Value."
        )

    close = pd.to_numeric(frame[close_col], errors="coerce")

    def optional(keys) -> "pd.Series":
        col = _match_column(list(frame.columns), keys)
        if col is None:
            return None
        return pd.to_numeric(frame[col], errors="coerce")

    open_col = optional(_OPEN_KEYS)
    high_col = optional(_HIGH_KEYS)
    low_col = optional(_LOW_KEYS)
    volume_col = optional(_VOLUME_KEYS)

    out = pd.DataFrame({
        "Open": open_col if open_col is not None else close,
        "High": high_col if high_col is not None else close,
        "Low": low_col if low_col is not None else close,
        "Close": close,
        "Volume": volume_col if volume_col is not None else 0,
    })
    out.index = pd.DatetimeIndex(index.to_numpy())

    for name, present in (("Open", open_col is not None),
                          ("High", high_col is not None),
                          ("Low", low_col is not None)):
        if not present:
            notes.append(f"{name} derived from Close")

    if volume_col is None:
        notes.append("Volume defaulted to zero")

    if high_col is None and low_col is None:
        notes.append("implied volatility will read zero")

    out = out[out["Close"].notna()].sort_index()
    if len(out) < 30:
        raise ValueError(
            f"Only {len(out)} usable price rows found — at least 30 are needed to "
            "compute 21-day volatility and drawdown metrics."
        )

    out.index.name = "Date"
    return out, notes


@st.cache_data(ttl=900, show_spinner=False)
def fetch_google_sheet(source: str) -> Tuple[pd.DataFrame, List[str]]:
    """Loads and normalises an OHLCV sheet straight from Google Sheets."""
    url = _normalise_sheet_url(source)

    try:
        raw = pd.read_csv(url)
    except Exception as exc:
        raise ValueError(
            f"Google returned an error for that sheet ({exc}). Make sure it is shared "
            "as 'Anyone with the link' and that the first tab is published as CSV."
        ) from exc

    if raw.empty or len(raw.columns) < 2:
        raise ValueError("That sheet appears to be empty, or has fewer than two columns.")

    return normalise_ohlcv(raw)


def company_choices() -> List[str]:
    return list(DEFAULT_STOCKS)


# --------------------------------------------------------------------------- #
# Google integration: company-name resolution + live quote                   #
# --------------------------------------------------------------------------- #
GOOGLE_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@st.cache_data(ttl=3600, show_spinner=False)
def google_suggest(query: str, limit: int = 6) -> List[str]:
    """
    Queries Google's public suggestion endpoint for company-name autocomplete.

    Returns an empty list when offline rather than raising, so the UI can fall
    back to the built-in company map.
    """
    text = (query or "").strip()
    if len(text) < 2:
        return []

    url = ("https://suggestqueries.google.com/complete/search"
           f"?client=firefox&hl=en&q={quote_plus(text)}")
    try:
        request = urllib.request.Request(url, headers=GOOGLE_UA)
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
        return [str(item) for item in payload[1]][:limit]
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def google_finance_quote(ticker: str, exchange: Optional[str] = None) -> Dict[str, object]:
    """
    Reads a live quote from Google Finance.

    Returns ``{}`` when the page cannot be read or parsed, so callers never have
    to guard against exceptions. Fields: name, price, currency, exchange,
    source_url, retrieved_at.
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return {}

    exch = (exchange or TICKER_EXCHANGE.get(symbol) or DEFAULT_EXCHANGE).upper()
    url = f"https://www.google.com/finance/quote/{symbol}:{exch}?hl=en"

    try:
        request = urllib.request.Request(url, headers=GOOGLE_UA)
        with urllib.request.urlopen(request, timeout=12) as response:
            html = response.read().decode("utf-8", "replace")
    except Exception:
        return {}

    # Confirm Google actually resolved this listing before trusting the numbers.
    title = re.search(r"<title>([^<]*)</title>", html)
    title_text = title.group(1) if title else ""
    if f"({symbol})" not in title_text:
        return {}

    name = re.search(r'<div class="gO24Ff">([^<]+)</div>', html)
    money = re.findall(r"\$[\d,]+\.\d{2}", html)

    price = None
    if money:
        try:
            candidate = float(money[0].replace("$", "").replace(",", ""))
            # Sanity gate: rejects page furniture and index widgets.
            if 0.01 <= candidate <= 1_000_000:
                price = candidate
        except ValueError:
            price = None

    return {
        "ticker": symbol,
        "exchange": exch,
        "name": unescape(name.group(1)) if name else COMPANY_NAME_MAP.get(symbol, symbol),
        "price": price,
        "currency": "USD",
        "source_url": url,
        "retrieved_at": datetime.now().strftime("%H:%M:%S"),
    }


def exchange_for(ticker: str) -> str:
    return TICKER_EXCHANGE.get(ticker.strip().upper(), DEFAULT_EXCHANGE)


def fetch_symbol(ticker: str) -> str:
    """Maps a display ticker onto the symbol the market data provider expects."""
    symbol = (ticker or "").strip().upper()
    return TICKER_ALIASES.get(symbol, symbol)


# --------------------------------------------------------------------------- #
# Autocomplete suggestions for the search box                                  #
# --------------------------------------------------------------------------- #
def _normalise(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def company_suggestions(query: str, limit: int = 8) -> List[Dict[str, str]]:
    """
    Builds the dropdown shown while typing a company name.

    Results are ranked: exact ticker, then company-name prefix, then substring,
    and finally live Google suggestions. Each entry carries the ticker it will
    resolve to so clicking a row is unambiguous.
    """
    text = (query or "").strip()
    if len(text) < 1:
        return []

    needle = _normalise(text)
    upper = text.upper()
    seen: set = set()
    results: List[Dict[str, str]] = []

    def add(ticker: str, label: str, source: str) -> None:
        if ticker in seen or len(results) >= limit:
            return
        seen.add(ticker)
        results.append({"ticker": ticker, "label": label, "source": source})

    ranked = []
    for symbol, company in COMPANY_NAME_MAP.items():
        company_norm = _normalise(company)
        sym_norm = symbol.replace(".", "").replace("-", "").lower()

        if upper == symbol:
            rank = 0
        elif needle and company_norm.startswith(needle):
            rank = 1
        elif needle and re.fullmatch(r"[a-z0-9]+", needle) and sym_norm.startswith(needle):
            rank = 2
        elif needle and re.search(rf"\b{re.escape(needle)}\b", company_norm):
            rank = 3
        elif needle and len(needle) >= 3 and needle in company_norm:
            rank = 4
        else:
            continue

        ranked.append((rank, len(company_norm), symbol, company))

    ranked.sort()
    for _, _, symbol, company in ranked:
        add(symbol, f"{company}", "matched")

    # Live Google suggestions, kept only when they resolve to a known listing
    # so that every row in the dropdown is actionable.
    if len(results) < limit:
        wanted = {r["label"].lower() for r in results}
        for suggestion in google_suggest(text):
            if len(results) >= limit:
                break
            low = suggestion.lower()
            if low in wanted:
                continue

            found = re.search(r"\(([A-Za-z.\-]{1,6})\)", suggestion)
            if found:
                ticker = found.group(1).upper()
                if ticker in COMPANY_NAME_MAP:
                    add(ticker, COMPANY_NAME_MAP[ticker], "google")
                continue

            for symbol, company in COMPANY_NAME_MAP.items():
                if _normalise(company) == low:
                    add(symbol, company, "google")
                    break

    return results


def resolve_free_text(text: str) -> Dict[str, object]:
    """
    Resolves free-typed text into a ticker.

    Order of attempts:
      1. exact ticker symbol in the built-in map
      2. company-name / substring match against the built-in map
      3. Google Suggest, picking the first suggestion that names a known symbol
      4. graceful fallback that passes the text through as a raw ticker

    Returns a dict with ticker, name, exchange, matched_by and suggestions.
    """
    clean = (text or "").strip()
    upper = clean.upper()
    normalised = re.sub(r"[^a-z0-9 ]", " ", clean.lower())
    normalised = re.sub(r"\s+", " ", normalised).strip()

    def pack(symbol: str, matched_by: str) -> Dict[str, object]:
        return {
            "ticker": symbol,
            "name": COMPANY_NAME_MAP.get(symbol, symbol),
            "exchange": exchange_for(symbol),
            "matched_by": matched_by,
        }

    if not clean:
        return {"ticker": "", "name": "", "exchange": DEFAULT_EXCHANGE,
                "matched_by": "empty", "suggestions": []}

    # Strip a trailing "(TICKER)" wrapper such as "Apple Inc. (AAPL)".
    wrapped = re.search(r"\(([A-Za-z.\-]{1,6})\)\s*$", clean)
    if wrapped:
        candidate = wrapped.group(1).upper()
        if candidate in COMPANY_NAME_MAP:
            return {**pack(candidate, "exact ticker"), "suggestions": []}

    if upper in COMPANY_NAME_MAP:
        return {**pack(upper, "exact ticker"), "suggestions": []}

    # Longest company-name match first so "Bank of America" beats "Bank".
    matches = []
    for symbol, company in COMPANY_NAME_MAP.items():
        company_norm = re.sub(r"[^a-z0-9 ]", " ", company.lower())
        company_norm = re.sub(r"\s+", " ", company_norm).strip()
        if not company_norm:
            continue
        if company_norm in normalised:
            matches.append((len(company_norm), symbol, "company name"))
        elif normalised and normalised in company_norm:
            matches.append((len(normalised), symbol, "partial name"))
        elif re.search(rf"\b{re.escape(symbol.replace('.', ''))}\b",
                       normalised.replace(" ", " "), re.IGNORECASE):
            # Whole-token match only, so "ZZZQQQ" never resolves to QQQ.
            matches.append((len(symbol), symbol, "symbol in text"))

    suggestions = google_suggest(clean) if normalised else []

    if matches:
        matches.sort(reverse=True)
        return {**pack(matches[0][1], matches[0][2]), "suggestions": suggestions}

    # Google Suggest: look for a suggestion containing a known symbol.
    for suggestion in suggestions:
        found = re.search(r"\(([A-Za-z.\-]{1,6})\)", suggestion)
        if found:
            candidate = found.group(1).upper()
            if candidate in COMPANY_NAME_MAP:
                return {**pack(candidate, "google suggest"), "suggestions": suggestions}
        words = re.sub(r"[^a-z ]", " ", suggestion.lower()).split()
        for word in words:
            if word.upper() in COMPANY_NAME_MAP:
                return {**pack(word.upper(), "google suggest"), "suggestions": suggestions}

    return {
        "ticker": upper,
        "name": f"Company ({upper})",
        "exchange": exchange_for(upper),
        "matched_by": "unresolved — treated as a raw ticker",
        "suggestions": suggestions,
    }


def resolve_company(user_input: str) -> Tuple[str, str]:
    """Parses 'Apple' / 'AAPL' / 'Apple Inc. (AAPL)' into (TICKER, Full Name (TICKER))."""
    clean = user_input.strip()
    upper = clean.upper()

    if upper in COMPANY_NAME_MAP:
        return upper, f"{COMPANY_NAME_MAP[upper]} ({upper})"

    for symbol, name in COMPANY_NAME_MAP.items():
        if clean.lower() in name.lower() or symbol.lower() in clean.lower():
            return symbol, f"{name} ({symbol})"

    # Strip a trailing "(TICKER)" wrapper used by the preset labels.
    if clean.endswith(")") and "(" in clean:
        inner = clean.split("(")[-1].replace(")", "").strip().upper()
        if inner in COMPANY_NAME_MAP:
            return inner, f"{COMPANY_NAME_MAP[inner]} ({inner})"

    return upper, f"Company ({upper})"


# --------------------------------------------------------------------------- #
# Prediction                                                                   #
# --------------------------------------------------------------------------- #
def metrics_to_vector(metrics: Dict[str, float]) -> np.ndarray:
    return get_extractor().metrics_to_feature_vector(metrics)


def predict_risk(vector: np.ndarray, sequence: Optional[np.ndarray] = None) -> Dict[str, object]:
    """
    Runs the hybrid model and returns a flat, UI-ready result dictionary.

    ``sequence`` is an optional ``[time_steps, 4]`` metric window; supplying it
    lets the trained BiLSTM branch contribute instead of the broadcast fallback.
    """
    score, category, pred_mdd, fuzzy_labels = get_model().predict_risk(vector, sequence)
    return {
        "score": float(score),
        "category": category if category in RISK_CATEGORIES else RISK_CATEGORIES[0],
        "predicted_mdd": float(pred_mdd),
        "fuzzy_labels": fuzzy_labels or {},
        "vector": np.asarray(vector, dtype=np.float32),
    }


def sequence_from_frame(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Builds the trailing metric window the BiLSTM expects from an OHLCV frame.

    Returns None when there is not enough history to fill the window.
    """
    from models import train as trainer

    try:
        metrics = trainer.rolling_metric_frame(df).dropna()
        if len(metrics) < trainer.SEQUENCE_LENGTH:
            return None
        return metrics.to_numpy(dtype=np.float32)[-trainer.SEQUENCE_LENGTH:]
    except Exception:
        return None


def _uses_torch_engine(engine) -> bool:
    """True when the active fuzzy engine is the PyTorch module variant."""
    if not TORCH_AVAILABLE:
        return False
    import torch

    return isinstance(engine, torch.nn.Module)


def membership_degrees(vector: np.ndarray) -> np.ndarray:
    """Gaussian membership degrees, shape ``(4 features, 3 terms)``."""
    engine = get_model().fuzzy_engine

    if _uses_torch_engine(engine):
        import torch

        with torch.no_grad():
            mu = engine.compute_membership(
                torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
            )
        return mu.squeeze(0).detach().cpu().numpy()

    return np.asarray(engine.compute_membership(np.asarray(vector, dtype=np.float32))[0])


def class_probabilities(vector: np.ndarray) -> Dict[str, float]:
    """Softmax probabilities over the four risk categories."""
    model = get_model()

    if TORCH_AVAILABLE:
        import torch
        import torch.nn.functional as F

        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                _, logits, _, _ = model.forward(
                    torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
                )
            probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
            return {cat: float(probs[i]) for i, cat in enumerate(RISK_CATEGORIES)}

    _, logits, _, _ = model.forward(np.asarray(vector, dtype=np.float32))
    raw = np.asarray(logits, dtype=float).reshape(-1)
    shifted = raw - raw.max()
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    return {cat: float(probs[i]) for i, cat in enumerate(RISK_CATEGORIES)}

def rule_firing_strengths(vector: np.ndarray) -> np.ndarray:
    """Normalized TSK rule firing weights for the 16 fuzzy rules."""
    engine = get_model().fuzzy_engine

    if TORCH_AVAILABLE:
        import torch

        if isinstance(engine, torch.nn.Module):
            with torch.no_grad():
                _, w_norm = engine.forward(
                    torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
                )
            return w_norm.squeeze(0).detach().cpu().numpy()

    _, w_norm = engine.forward(np.asarray(vector, dtype=np.float32))
    return np.asarray(w_norm, dtype=np.float64).reshape(-1)


def rule_catalog() -> List[Dict[str, str]]:
    """Human readable antecedents for each of the 16 fuzzy rules."""
    catalog = []
    for rule_id, (s, d, iv, hv) in enumerate(build_rule_list(16), start=1):
        catalog.append({
            "id": rule_id,
            "sharpe": TERM_LABELS[s],
            "drawdown": TERM_LABELS[d],
            "implied_vol": TERM_LABELS[iv],
            "hist_vol": TERM_LABELS[hv],
        })
    return catalog


def describe_rule(entry: Dict[str, str]) -> str:
    return (
        f"IF Sharpe is {entry['sharpe']} AND Drawdown is {entry['drawdown']} "
        f"AND IV is {entry['implied_vol']} AND HV is {entry['hist_vol']}"
    )


FEATURE_LABELS = {
    "sharpe": "Sharpe Ratio",
    "drawdown": "Max Drawdown",
    "implied_vol": "Implied Volatility",
    "hist_vol": "Historical Volatility",
}


# --------------------------------------------------------------------------- #
# Genetic algorithm                                                            #
# --------------------------------------------------------------------------- #
def run_genetic_algorithm(
    population_size: int,
    generations: int,
    seed: int = 42,
    progress_callback=None,
) -> Dict[str, object]:
    from models.genetic_optimizer import GeneticAlgorithmOptimizer

    rng = np.random.default_rng(seed)
    X_val = rng.uniform(0.05, 0.5, size=(30, 4))
    y_val = rng.uniform(10, 90, size=30)

    ga = GeneticAlgorithmOptimizer(
        population_size=population_size,
        generations=generations,
    )
    best_chromosome, history = ga.optimize(X_val, y_val, progress_callback=progress_callback)

    return {
        "chromosome": np.asarray(best_chromosome, dtype=float),
        "history": [float(h) for h in history],
        "population_size": population_size,
        "generations": generations,
        "seed": seed,
    }


# --------------------------------------------------------------------------- #
# Test suite                                                                   #
# --------------------------------------------------------------------------- #
def run_tests() -> Dict[str, object]:
    from test_runner import run_all_tests_detailed

    return run_all_tests_detailed()


# --------------------------------------------------------------------------- #
# PDF report                                                                   #
# --------------------------------------------------------------------------- #
PDF_FILENAME = "Stock_Market_Risk_Prediction_Report.pdf"


def build_pdf_bytes() -> bytes:
    """Builds the research report and returns it as PDF bytes."""
    from generate_pdf_report import build_pdf_report

    temp_name = f"_ui_report_{uuid.uuid4().hex}.pdf"
    build_pdf_report(filename=temp_name)
    temp_path = ROOT / temp_name
    try:
        return temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)


def build_pdf_on_disk() -> Path:
    from generate_pdf_report import build_pdf_report

    build_pdf_report(filename=PDF_FILENAME)
    return ROOT / PDF_FILENAME


# --------------------------------------------------------------------------- #
# Documentation                                                                #
# --------------------------------------------------------------------------- #
def load_markdown(name: str) -> str:
    path = ROOT / name
    if not path.exists():
        return f"_{name} not found._"
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Engine status                                                                #
# --------------------------------------------------------------------------- #
def engine_status() -> List[Dict[str, str]]:
    from data.data_loader import YFINANCE_AVAILABLE
    from models.fuzzy_layer import TORCH_AVAILABLE as LAYER_TORCH

    try:
        import torch

        torch_label = f"PyTorch {torch.__version__}" if LAYER_TORCH else "NumPy fallback"
    except Exception:
        torch_label = "NumPy fallback"

    try:
        import streamlit

        streamlit_label = f"Streamlit {streamlit.__version__}"
    except Exception:
        streamlit_label = "Streamlit"

    return [
        {"name": "Deep engine", "value": torch_label},
        {"name": "Market data", "value": "yfinance live" if YFINANCE_AVAILABLE else "Synthetic only"},
        {"name": "Web interface", "value": streamlit_label},
    ]


def capture_stdout(func, *args, **kwargs):
    """Runs ``func`` while capturing stdout/stderr, returning ``(result, captured)``."""
    import sys

    buffer = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = buffer, buffer
        result = func(*args, **kwargs)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return result, buffer.getvalue()
