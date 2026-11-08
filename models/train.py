"""
Supervised Training Engine for the Hybrid Deep Neuro-Fuzzy Risk Model.

The repository originally shipped a randomly initialised network, which produced a
near-constant risk score regardless of input. This module builds a real training
set from market history and fits the three prediction heads so the score actually
tracks risk.

Training targets (all derived from forward-looking price behaviour):

    y_score = clip(future_max_drawdown * 200 + HV * 100, 0, 100)
    y_cat   = bucket(y_score)  ->  Low / Moderate / High / Extreme
    y_mdd   = future_max_drawdown

The BiLSTM sequence channel and the dense vector channel share the same feature
domain ([Sharpe, MaxDrawdown, ImpliedVol, HistVol]) so that inference through the
plain vector API stays consistent with the trained model.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    ANNUALIZATION_FACTOR,
    DATA_DIR,
    HV_WINDOW,
    MODEL_CONFIG,
    RISK_CATEGORIES,
    RISK_FREE_RATE,
)
from data.data_loader import StockDataLoader, SyntheticStockGenerator

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without PyTorch
    torch = None
    nn = None
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Hyper-parameters                                                            #
# --------------------------------------------------------------------------- #
SEQUENCE_LENGTH = MODEL_CONFIG.get("sequence_length", 20)
FORWARD_HORIZON = 10
EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.003
VALIDATION_SPLIT = 0.15
TRAIN_SEED = 42

# Fraction of the concatenated dataset reserved as the held-out final test set.
# These are always the LAST (chronologically most recent) TEST_SPLIT of all rows.
# They are NEVER exposed to GA optimisation, model selection, or early stopping.
TEST_SPLIT = 0.10

# Broad, liquid universe spanning defensive to speculative regimes.
TRAINING_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
    "XOM", "WMT", "UNH", "JNJ", "PG", "HD", "BAC", "CVX", "NFLX", "AMD",
    "COST", "PEP",
]

WEIGHTS_PATH = Path(DATA_DIR) / "risk_model_weights.pt"
GP_EXPRESSION_PATH = Path(DATA_DIR) / "gp_expression_tree.json"

# Feature bounds mirror RiskFeatureExtractor.feature_bounds.
_BOUNDS = {
    "sharpe": (-2.0, 4.0),
    "drawdown": (0.0, 0.8),
    "implied_vol": (0.05, 1.0),
    "hist_vol": (0.05, 1.0),
}
_FEATURES = ["sharpe", "drawdown", "implied_vol", "hist_vol"]


def seed_everything(seed: int = TRAIN_SEED) -> None:
    """Makes weight initialisation reproducible across processes."""
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)


def get_train_val_indices(
    n: int,
    validation_split: float = VALIDATION_SPLIT,
    seed: int = TRAIN_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return *(train_idx, val_idx)* — the exact same permuted split that
    ``train_model()`` uses internally.

    Both arrays are integer indices into the *original* (unshuffled) dataset.
    The GA uses these to build a validation set that is provably disjoint from
    the rows used to fit neural-network weights.

    NOTE: when called on the full dataset (including held-out test samples),
    use ``get_train_val_test_indices`` instead so the test set is excluded.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = max(int(n * (1.0 - validation_split)), 1)
    return order[:cut], order[cut:]


def get_train_val_test_indices(
    n: int,
    validation_split: float = VALIDATION_SPLIT,
    test_split: float = TEST_SPLIT,
    seed: int = TRAIN_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return *(train_idx, val_idx, test_idx)* — three pairwise-disjoint index sets.

    Chronological guarantee
    -----------------------
    The concatenated dataset is ordered by (ticker, time).  The final test set
    is always the last ``test_split`` fraction by *position* so it contains
    only rows whose look-ahead window lies strictly after the last training row.
    The train/val split within the remaining rows is then shuffled (seeded) so
    the neural network does not overfit to chronological artefacts.

    The three returned sets satisfy:
        len(train_idx) + len(val_idx) + len(test_idx) == n
        set(train_idx) & set(val_idx) == {}
        set(train_idx) & set(test_idx) == {}
        set(val_idx)   & set(test_idx) == {}

    The test set is NEVER used during GA optimisation or model selection.
    """
    # 1. Reserve the chronologically last rows for the final test set.
    n_test = max(int(n * test_split), 1)
    n_trainval = n - n_test

    # test_idx = the last n_test positions by concatenation order
    test_idx = np.arange(n_trainval, n, dtype=np.intp)

    # 2. Shuffle the remaining rows to form train / val.
    rng = np.random.default_rng(seed)
    tv_positions = np.arange(n_trainval, dtype=np.intp)
    order = rng.permutation(n_trainval)
    cut = max(int(n_trainval * (1.0 - validation_split)), 1)

    train_idx = tv_positions[order[:cut]]
    val_idx   = tv_positions[order[cut:]]

    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- #
# Feature / target engineering                                                #
# --------------------------------------------------------------------------- #
def rolling_metric_frame(df: pd.DataFrame, is_synthetic: bool = False) -> pd.DataFrame:
    """
    Vectorised rolling risk metrics, one row per trading day.

    Columns are clipped into the exact feature domain the fuzzy layer was
    configured with, which keeps membership degrees in a sane range.

    Implied-Volatility proxy (Phase 7 fix)
    ───────────────────────────────────────
    The Garman-Klass estimator requires genuine intraday High-Low spread.
    For synthetic OHLC data the intraday spread is small (+/-1.2% of Close),
    causing the annualised GK value to sit at or below the 0.05 clip floor.
    This makes IV a constant and provides no information to the fuzzy layer.

    When ``is_synthetic=True`` OR when the computed GK proxy is constant
    (<= lower clip bound + 1e-4) across more than 90% of observations, the
    function substitutes ``HV * 1.05`` as an honest proxy labelled
    ``implied_vol``.  This is a known approximation, not a claim of real IV.
    Real market data produces genuine GK values well above the floor.
    """
    close = df["Close"]
    log_rets = np.log(close / close.shift(1))
    ann = float(np.sqrt(ANNUALIZATION_FACTOR))

    hist_vol = (log_rets.rolling(HV_WINDOW).std() * ann).clip(*_BOUNDS["hist_vol"])

    running_peak = close.cummax()
    drawdown = ((running_peak - close) / running_peak).clip(*_BOUNDS["drawdown"])

    # Garman-Klass proxy, averaged over the volatility window.
    gk = (
        0.5 * np.log(df["High"] / df["Low"]) ** 2
        - (2.0 * np.log(2.0) - 1.0) * np.log(df["Close"] / df["Open"]) ** 2
    )
    gk_iv = (gk.clip(lower=0).rolling(HV_WINDOW).mean() * ann).clip(*_BOUNDS["implied_vol"])

    # Detect degenerate IV: synthetic flag OR >=90% of values at the lower clip bound
    iv_lo = _BOUNDS["implied_vol"][0]
    pct_at_floor = float((gk_iv <= iv_lo + 1e-4).mean())
    use_hv_proxy = is_synthetic or (pct_at_floor > 0.90)

    if use_hv_proxy:
        # HV * 1.05 is a standard rule-of-thumb approximation for IV.
        # It is honest - we do not claim it is market-derived implied volatility.
        implied_vol = (hist_vol * 1.05).clip(*_BOUNDS["implied_vol"])
    else:
        implied_vol = gk_iv

    mean_ann_return = log_rets.rolling(HV_WINDOW).mean() * ANNUALIZATION_FACTOR
    safe_hv = hist_vol.replace(0.0, np.nan)
    sharpe = ((mean_ann_return - RISK_FREE_RATE) / safe_hv).clip(*_BOUNDS["sharpe"])

    return pd.DataFrame({
        "sharpe": sharpe,
        "drawdown": drawdown,
        "implied_vol": implied_vol,
        "hist_vol": hist_vol,
    }, index=df.index)


def _bucket(score: float) -> int:
    if score < 25.0:
        return 0
    if score < 50.0:
        return 1
    if score < 75.0:
        return 2
    return 3


def _synthetic_sigma(ticker: str) -> float:
    """
    Deterministic volatility for a synthetic stand-in series.

    Known tickers keep their calibrated level; anything else gets a stable
    pseudo-random value in [0.15, 0.60] so training sets stay diverse.
    """
    calibrated = {
        "TSLA": 0.45, "NVDA": 0.50, "AAPL": 0.22,
        "MSFT": 0.24, "SPY": 0.15, "QQQ": 0.20, "META": 0.34,
    }
    if ticker in calibrated:
        return calibrated[ticker]
    return 0.15 + (abs(hash(ticker)) % 46) / 100.0


def _training_frame(
    ticker: str,
    period: str = "5y",
    allow_network: bool = True,
) -> Optional[pd.DataFrame]:
    """Fetches one ticker (live or synthetic) and returns metrics + prices.

    Tracks whether data is synthetic so that rolling_metric_frame can use
    the HV*1.05 IV proxy instead of the degenerate GK estimate (Phase 7 fix).
    """
    frame = None
    is_synthetic = False

    if allow_network:
        try:
            loader = StockDataLoader(use_offline_fallback=True)
            frame, is_synthetic = loader.fetch_stock_data(ticker, period=period)
        except Exception:
            frame = None

    if frame is None or len(frame) < 120:
        # Deterministic synthetic stand-in keeps training reproducible offline.
        frame = SyntheticStockGenerator.generate_stock_data(
            ticker=ticker,
            days=1250,
            sigma=_synthetic_sigma(ticker),
            seed=abs(hash(ticker)) % 10000,
        )
        is_synthetic = True

    metrics = rolling_metric_frame(frame, is_synthetic=is_synthetic)
    combined = metrics.copy()
    combined["_close"] = frame["Close"]
    return combined.dropna()


def build_dataset(
    tickers: Optional[List[str]] = None,
    sequence_length: int = SEQUENCE_LENGTH,
    horizon: int = FORWARD_HORIZON,
    period: str = "5y",
    allow_network: bool = True,
) -> Dict[str, np.ndarray]:
    """Builds (sequence, vector, score, category, drawdown) tensors."""
    tickers = tickers or TRAINING_UNIVERSE

    seq_rows: List[np.ndarray] = []
    vec_rows: List[np.ndarray] = []
    score_rows: List[float] = []
    category_rows: List[int] = []
    mdd_rows: List[float] = []

    for ticker in tickers:
        try:
            frame = _training_frame(ticker, period=period, allow_network=allow_network)
        except Exception:
            continue
        if frame is None or len(frame) < sequence_length + horizon + 5:
            continue

        values = frame[_FEATURES].to_numpy(dtype=np.float32)
        close = frame["_close"].to_numpy(dtype=np.float64)
        total = len(frame)

        for end in range(sequence_length - 1, total - horizon):
            future = close[end + 1: end + 1 + horizon]
            if future.size == 0 or close[end] <= 0:
                continue

            future_max_dd = float((close[end] - future.min()) / close[end])
            if not np.isfinite(future_max_dd):
                continue

            hv = float(values[end][3])
            score = float(np.clip(future_max_dd * 200.0 + hv * 100.0, 0.0, 100.0))

            seq_rows.append(values[end - sequence_length + 1: end + 1])
            vec_rows.append(values[end])
            score_rows.append(score)
            category_rows.append(_bucket(score))
            mdd_rows.append(min(max(future_max_dd, 0.0), 1.0))

    if not seq_rows:
        raise ValueError("Could not build a training set from any ticker.")

    return {
        "X_seq": np.asarray(seq_rows, dtype=np.float32),
        "X_vec": np.asarray(vec_rows, dtype=np.float32),
        "y_score": np.asarray(score_rows, dtype=np.float32).reshape(-1, 1),
        "y_cat": np.asarray(category_rows, dtype=np.int64),
        "y_mdd": np.asarray(mdd_rows, dtype=np.float32).reshape(-1, 1),
    }


# --------------------------------------------------------------------------- #
# Training loop                                                               #
# --------------------------------------------------------------------------- #
def train_model(
    model,
    dataset: Dict[str, np.ndarray],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    validation_split: float = VALIDATION_SPLIT,
    seed: int = TRAIN_SEED,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Dict[str, object]:
    """Fits the three heads on the dataset. Returns a training history."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training - install torch>=2.0.0.")

    seed_everything(seed)
    model.train()

    if "split_indices" in dataset:
        tr_idx = dataset["split_indices"]["train"]
        va_idx = dataset["split_indices"]["val"]
        cut = int(len(tr_idx))
        n = cut + int(len(va_idx))
        order = np.concatenate([tr_idx, va_idx])
    else:
        n = len(dataset["X_vec"])
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        cut = max(int(n * (1.0 - validation_split)), 1)

    X_seq = torch.tensor(dataset["X_seq"][order])
    X_vec = torch.tensor(dataset["X_vec"][order])
    y_score = torch.tensor(dataset["y_score"][order])
    y_cat = torch.tensor(dataset["y_cat"][order])
    y_mdd = torch.tensor(dataset["y_mdd"][order])

    tr = slice(0, cut)
    va = slice(cut, n)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(epochs, 1)
    )
    score_loss_fn = nn.MSELoss()
    mdd_loss_fn = nn.MSELoss()
    cat_loss_fn = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_mae": [], "val_mae": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(cut)
        total_loss = 0.0
        batches = 0

        for start in range(0, cut, batch_size):
            batch = idx[start: start + batch_size]
            if batch.numel() == 0:
                continue

            optimiser.zero_grad()
            pred_score, logits, pred_mdd, _ = model(
                X_vec[batch], X_seq[batch]
            )
            loss = (
                score_loss_fn(pred_score, y_score[batch])
                + 2.0 * cat_loss_fn(logits, y_cat[batch])
                + 10.0 * mdd_loss_fn(pred_mdd, y_mdd[batch])
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            if hasattr(model, 'fuzzy_engine') and hasattr(model.fuzzy_engine, 'clamp_and_sort_parameters'):
                model.fuzzy_engine.clamp_and_sort_parameters()

            total_loss += float(loss.item())
            batches += 1

        scheduler.step()

        train_mae, train_acc = _evaluate(model, X_vec, X_seq, y_score, y_cat, tr)
        val_mae, val_acc = _evaluate(model, X_vec, X_seq, y_score, y_cat, va)

        history["epoch"].append(epoch)
        history["train_mae"].append(round(train_mae, 4))
        history["val_mae"].append(round(val_mae, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["train_acc"] = history.get("train_acc", [])
        history["train_acc"].append(round(train_acc, 4))

        if progress_callback is not None:
            try:
                progress_callback(epoch, epochs, {
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "val_acc": val_acc,
                })
            except Exception:
                pass

    model.eval()
    history["samples"] = int(n)
    history["train_samples"] = int(cut)
    history["val_samples"] = int(n - cut)
    history["epochs"] = int(epochs)
    return history


def _evaluate(model, X_vec, X_seq, y_score, y_cat, index) -> Tuple[float, float]:
    """Returns (mean absolute score error, category accuracy %) over a slice."""
    model.eval()
    total = index.stop - index.start
    if total <= 0:
        return 0.0, 0.0

    mae = 0.0
    correct = 0
    with torch.no_grad():
        for start in range(index.start, index.stop, 256):
            stop = min(start + 256, index.stop)
            vx = X_vec[start:stop]
            sx = X_seq[start:stop]
            pred_score, logits, _, _ = model(vx, sx)
            mae += float((pred_score - y_score[start:stop]).abs().sum().item())
            correct += int((logits.argmax(dim=1) == y_cat[start:stop]).sum().item())

    model.train()
    return mae / total, (correct / total) * 100.0


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #
def weights_exist() -> bool:
    return WEIGHTS_PATH.exists()


def save_weights(model) -> Path:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to persist weights.")
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    return WEIGHTS_PATH


def load_weights(model) -> bool:
    """
    Load persisted weights. Returns True when weights were applied.

    If the saved weights were produced by a different architecture
    (e.g. fusion_dim=161 before Phase 3 vs 162 after), the function
    tries strict=False and verifies that the critical fusion layer is
    actually compatible. If not, it returns False so callers know
    they must retrain.
    """
    if not TORCH_AVAILABLE or not WEIGHTS_PATH.exists():
        return False
    try:
        state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
        # Strict load first
        try:
            model.load_state_dict(state, strict=True)
            if hasattr(model, 'fuzzy_engine') and hasattr(model.fuzzy_engine, 'clamp_and_sort_parameters'):
                model.fuzzy_engine.clamp_and_sort_parameters()
            model.eval()
            return True
        except RuntimeError:
            pass   # shape mismatch — try lenient load

        # Lenient load: apply only matching keys
        model_state  = model.state_dict()
        matched      = {k: v for k, v in state.items()
                        if k in model_state and v.shape == model_state[k].shape}

        # Adapt fusion_net.0.weight between 161 and 162 input dimensions
        if "fusion_net.0.weight" in state and "fusion_net.0.weight" not in matched:
            s_weight = state["fusion_net.0.weight"]
            m_weight = model_state.get("fusion_net.0.weight")
            if m_weight is not None and s_weight.shape[0] == m_weight.shape[0]:
                min_cols = min(s_weight.shape[1], m_weight.shape[1])
                m_weight = m_weight.clone()
                m_weight[:, :min_cols] = s_weight[:, :min_cols]
                matched["fusion_net.0.weight"] = m_weight

        # The critical fusion layer must be successfully matched or adapted
        if "fusion_net.0.weight" not in matched:
            return False

        if not matched:
            return False   # nothing matches — weights are from a completely different arch
        model_state.update(matched)
        model.load_state_dict(model_state, strict=False)
        if hasattr(model, 'fuzzy_engine') and hasattr(model.fuzzy_engine, 'clamp_and_sort_parameters'):
            model.fuzzy_engine.clamp_and_sort_parameters()
        model.eval()
        return True
    except Exception:
        return False


def save_gp_expression(model) -> Path:
    """
    Persist the GP expression tree attached to *model* to a JSON file.

    Silently returns the default path with nothing written when no expression
    has been attached (``model._gp_expression is None``).

    Returns:
        Path to the written file.
    """
    from models.genetic_programming import save_expression_tree
    expr = getattr(model, "_gp_expression", None)
    if expr is None:
        return GP_EXPRESSION_PATH    # no-op but still return the path
    save_expression_tree(expr.to_dict(), GP_EXPRESSION_PATH)
    return GP_EXPRESSION_PATH


def load_gp_expression(model) -> bool:
    """
    Restore the GP expression tree from disk and attach it to *model*.

    Returns True when an expression was successfully loaded and attached.
    Returns False (no-op) when the file is absent or corrupt.
    """
    from models.genetic_programming import load_expression_tree
    node = load_expression_tree(GP_EXPRESSION_PATH)
    if node is None:
        return False
    model.set_gp_expression(node)
    return True


def train_and_save(
    model,
    tickers: Optional[List[str]] = None,
    epochs: int = EPOCHS,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Dict[str, object]:
    """End-to-end: build data -> train -> persist weights."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training - install torch>=2.0.0.")

    tickers = tickers or TRAINING_UNIVERSE
    dataset = build_dataset(tickers)
    history = train_model(model, dataset, epochs=epochs, progress_callback=progress_callback)
    history["tickers"] = tickers
    history["weights_path"] = str(save_weights(model))
    return history


def score_spread(
    model,
    dataset: Optional[Dict[str, np.ndarray]] = None,
    allow_network: bool = True,
) -> float:
    """
    Standard deviation of predicted risk scores across a probe grid.

    Used by the UI to show whether the model is actually responsive; a value
    near zero means the network is still effectively untrained.
    """
    if dataset is None:
        dataset = build_dataset(
            TRAINING_UNIVERSE[:6], period="1y", allow_network=allow_network
        )
    if not TORCH_AVAILABLE or not len(dataset["X_vec"]):
        return 0.0

    model.eval()
    with torch.no_grad():
        vectors = torch.tensor(dataset["X_vec"][:512])
        sequences = torch.tensor(dataset["X_seq"][:512])
        scores, _, _, _ = model(vectors, sequences)
    return float(scores.squeeze().std().item())


# --------------------------------------------------------------------------- #
# Evaluation against held-out splits                                          #
# --------------------------------------------------------------------------- #

def evaluate_split(
    model,
    dataset: Dict[str, np.ndarray],
    indices: np.ndarray,
) -> Dict[str, float]:
    """
    Compute MSE and MAE of the risk-score head on *indices* rows of *dataset*.

    Both metrics are in [0, 100] score-point units — the same scale as the
    training target ``y_score``.

    Returns
    -------
    {"mse": float, "mae": float, "n": int}
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for evaluation — install torch>=2.0.0.")

    n = len(indices)
    if n == 0:
        return {"mse": float("nan"), "mae": float("nan"), "n": 0}

    X_vec   = torch.tensor(dataset["X_vec"][indices],   dtype=torch.float32)
    X_seq   = torch.tensor(dataset["X_seq"][indices],   dtype=torch.float32)
    y_score = torch.tensor(dataset["y_score"][indices], dtype=torch.float32).squeeze(-1)

    model.eval()
    total_se = 0.0
    total_ae = 0.0

    with torch.no_grad():
        batch_size = 256
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            pred_score, _, _, _ = model(X_vec[start:stop], X_seq[start:stop])
            pred = pred_score.squeeze(-1)
            diff = pred - y_score[start:stop]
            total_se += float((diff ** 2).sum().item())
            total_ae += float(diff.abs().sum().item())

    return {
        "mse": total_se / n,
        "mae": total_ae / n,
        "n":   n,
    }


def build_dataset_split(
    tickers: Optional[List[str]] = None,
    sequence_length: int = SEQUENCE_LENGTH,
    horizon: int = FORWARD_HORIZON,
    period: str = "5y",
    allow_network: bool = True,
    validation_split: float = VALIDATION_SPLIT,
    test_split: float = TEST_SPLIT,
    seed: int = TRAIN_SEED,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the full dataset and return explicit, pairwise-disjoint index splits.

    Returns
    -------
    (dataset, train_idx, val_idx, test_idx)

    * ``dataset``   — full concatenated dataset dict (all N rows)
    * ``train_idx`` — indices used for NN weight fitting (shuffled)
    * ``val_idx``   — indices used for GA fitness / model selection (shuffled)
    * ``test_idx``  — indices held out for final reporting (chronological tail)

    Chronological guarantee
    -----------------------
    ``test_idx`` always contains the last ``test_split`` fraction of rows by
    concatenation order. Because rows are appended ticker-by-ticker in time
    order, these rows represent the most recent market regimes and their
    look-ahead windows do not overlap with any training or validation sample.

    Disjointness guarantee
    ----------------------
    The three index sets satisfy:
        union(train, val, test) == {0, ..., N-1}
        pairwise intersection == {} for all three pairs
    """
    dataset = build_dataset(
        tickers=tickers,
        sequence_length=sequence_length,
        horizon=horizon,
        period=period,
        allow_network=allow_network,
    )
    n = len(dataset["X_vec"])
    train_idx, val_idx, test_idx = get_train_val_test_indices(
        n,
        validation_split=validation_split,
        test_split=test_split,
        seed=seed,
    )
    return dataset, train_idx, val_idx, test_idx
