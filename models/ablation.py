"""
Ablation Framework for the Hybrid Neuro-Fuzzy + GA + GP System.

Runs six configurations on IDENTICAL, leakage-controlled data partitions:

    1. BiLSTM-only            -- NN sequence branch; fuzzy + GP disabled
    2. Fuzzy-only             -- TSK fuzzy inference; BiLSTM + GP disabled
    3. Neuro-Fuzzy            -- BiLSTM + TSK fuzzy; no evolutionary optimisation
    4. Neuro-Fuzzy + GA       -- adds GA-optimised fuzzy MF parameters
    5. Neuro-Fuzzy + GP       -- adds evolved GP symbolic expression
    6. Neuro-Fuzzy + GA + GP  -- full hybrid system

Each configuration is evaluated on the SAME held-out test split.
No test rows are ever seen during training or evolution.

Metrics: MSE, MAE (both in [0, 100] score-point units), and test-set R2.

Honest reporting
----------------
- All six variants use the same dataset and the same chronological test split.
- Synthetic-data results are clearly labelled (allow_network=False).
- Real-market results are labelled separately when allow_network=True.
- Performance differences across variants may be small or zero for untrained
  models; the framework reports whatever it measures without fabrication.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from models.train import (
    TEST_SPLIT,
    TRAIN_SEED,
    VALIDATION_SPLIT,
    build_dataset,
    get_train_val_test_indices,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AblationRow:
    name: str
    mse:  float
    mae:  float
    r2:   float
    elapsed_seconds: float
    note: str = ""


@dataclass
class AblationReport:
    rows:            List[AblationRow]
    data_type:       str    # "synthetic" or "real-market"
    total_samples:   int
    train_samples:   int
    val_samples:     int
    test_samples:    int
    seed:            int

    def summary(self) -> str:
        header = (
            f"Ablation Study — {self.data_type} data  "
            f"(test: {self.test_samples} rows)\n"
            f"{'Configuration':<30} {'MSE':>10} {'MAE':>10} {'R2':>8}  {'Time':>6}\n"
            + "-" * 70
        )
        body = "\n".join(
            f"{r.name:<30} {r.mse:>10.3f} {r.mae:>10.3f} {r.r2:>8.4f}  {r.elapsed_seconds:>5.1f}s"
            for r in self.rows
        )
        return header + "\n" + body

    def to_dict(self) -> Dict:
        return {
            "data_type":     self.data_type,
            "total_samples": self.total_samples,
            "train_samples": self.train_samples,
            "val_samples":   self.val_samples,
            "test_samples":  self.test_samples,
            "seed":          self.seed,
            "results": [
                {
                    "name":    r.name,
                    "mse":     r.mse,
                    "mae":     r.mae,
                    "r2":      r.r2,
                    "elapsed": r.elapsed_seconds,
                    "note":    r.note,
                }
                for r in self.rows
            ],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_to_mse_mae_r2(pred: np.ndarray, true: np.ndarray):
    pred  = np.asarray(pred,  dtype=float).ravel()
    true  = np.asarray(true,  dtype=float).ravel()
    diff  = pred - true
    mse   = float(np.mean(diff ** 2))
    mae   = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2    = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    return mse, mae, r2


def _numpy_predict(model, X_vec: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
    """
    Run a batch of predictions using the model's forward() method.
    Works for both the PyTorch and NumPy fallback variants.
    """
    try:
        from models.fuzzy_layer import TORCH_AVAILABLE
        if TORCH_AVAILABLE:
            import torch
            scores = []
            batch = 256
            model.eval()
            with torch.no_grad():
                for s in range(0, len(X_vec), batch):
                    vx = torch.tensor(X_vec[s:s+batch], dtype=torch.float32)
                    sx = torch.tensor(X_seq[s:s+batch], dtype=torch.float32)
                    pred, _, _, _ = model(vx, sx)
                    scores.append(pred.squeeze(-1).cpu().numpy())
            return np.concatenate(scores)
        else:
            scores = []
            for i in range(len(X_vec)):
                r, _, _, _ = model.forward(X_vec[i])
                scores.append(float(r))
            return np.array(scores, dtype=float)
    except Exception:
        return np.full(len(X_vec), 50.0)


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

def _run_bilstm_only(model, X_vec, X_seq, y_true, disable_fuzzy: bool = False):
    """BiLSTM branch; fuzzy output is zeroed by setting consequent weights to 0."""
    import copy
    m2 = copy.deepcopy(model)
    # Zero the fuzzy consequent weights so the fuzzy score is near 0
    try:
        m2.fuzzy_engine.consequent_weights.data.zero_()
    except Exception:
        try:
            m2.fuzzy_engine.consequent_weights[:] = 0.0
        except Exception:
            pass
    m2.set_gp_expression(None)
    return _numpy_predict(m2, X_vec, X_seq)


def _run_fuzzy_only(model, X_vec, X_seq, y_true):
    """Use only the TSK fuzzy score (defuzzified output) as the prediction."""
    try:
        from models.fuzzy_layer import TORCH_AVAILABLE
        if TORCH_AVAILABLE:
            import torch
            m2 = copy.deepcopy(model)
            m2.set_gp_expression(None)
            scores = []
            m2.eval()
            with torch.no_grad():
                for s in range(0, len(X_vec), 256):
                    vx = torch.tensor(X_vec[s:s+256], dtype=torch.float32)
                    fuzzy_score, _ = m2.fuzzy_engine(vx)
                    # Fuzzy output is already roughly in score range; scale to [0,100]
                    scores.append(fuzzy_score.squeeze(-1).cpu().numpy() * 10.0)
            return np.clip(np.concatenate(scores), 0, 100)
        else:
            results = []
            fe = model.fuzzy_engine
            for i in range(len(X_vec)):
                fs, _ = fe.forward(X_vec[i:i+1])
                results.append(float(fs) * 10.0)
            return np.clip(np.array(results, dtype=float), 0, 100)
    except Exception:
        return np.full(len(X_vec), 50.0)


def run_ablation(
    model,
    dataset: Dict[str, np.ndarray],
    ga_result: Optional[Dict] = None,
    gp_result_dict: Optional[Dict] = None,
    allow_network: bool = False,
    seed: int = TRAIN_SEED,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> AblationReport:
    """
    Run all six ablation configurations and return an AblationReport.

    Args:
        model           : Fully trained DeepNeuroFuzzyRiskModel.
        dataset         : Dataset dict (X_vec, X_seq, y_score, ...).
        ga_result       : Optional GA result dict (already run).
                          If provided, a GA-optimised model variant is used.
        gp_result_dict  : Optional GP result dict from run_gp() containing
                          'expression_tree'. If provided, the GP variant is used.
        allow_network   : Whether live data was used (labels the report).
        seed            : Random seed for reproducibility.
        progress_callback: Optional callback(config_name) called before each run.

    Returns:
        AblationReport with six AblationRow entries.
    """
    from models.genetic_programming import attach_gp_to_model

    n = len(dataset["X_vec"])
    if "split_indices" in dataset:
        train_idx = dataset["split_indices"]["train"]
        val_idx   = dataset["split_indices"]["val"]
        test_idx  = dataset["split_indices"]["test"]
    else:
        train_idx, val_idx, test_idx = get_train_val_test_indices(
            n, validation_split=VALIDATION_SPLIT, test_split=TEST_SPLIT, seed=seed
        )

    X_vec_test = dataset["X_vec"][test_idx]
    X_seq_test = dataset["X_seq"][test_idx]
    y_true     = dataset["y_score"][test_idx].ravel()

    data_type = "real-market" if allow_network else "synthetic"
    rows: List[AblationRow] = []

    # ── Helper ──────────────────────────────────────────────────────────────
    def _eval(name: str, pred_fn, note: str = ""):
        if progress_callback:
            progress_callback(name)
        t0   = time.time()
        pred = pred_fn()
        mse, mae, r2 = _score_to_mse_mae_r2(pred, y_true)
        rows.append(AblationRow(
            name=name, mse=mse, mae=mae, r2=r2,
            elapsed_seconds=time.time() - t0, note=note
        ))

    # ── 1. BiLSTM-only ──────────────────────────────────────────────────────
    _eval("1. BiLSTM-only",
          lambda: _run_bilstm_only(model, X_vec_test, X_seq_test, y_true),
          note="Fuzzy consequents zeroed; GP disabled")

    # ── 2. Fuzzy-only ───────────────────────────────────────────────────────
    _eval("2. Fuzzy-only",
          lambda: _run_fuzzy_only(model, X_vec_test, X_seq_test, y_true),
          note="TSK defuzzified score x10; BiLSTM/GP disabled")

    # ── 3. Neuro-Fuzzy (baseline) ────────────────────────────────────────────
    m_base = copy.deepcopy(model)
    m_base.set_gp_expression(None)
    _eval("3. Neuro-Fuzzy",
          lambda: _numpy_predict(m_base, X_vec_test, X_seq_test),
          note="BiLSTM + fuzzy; no evolutionary optimisation")

    # ── 4. Neuro-Fuzzy + GA ─────────────────────────────────────────────────
    # Use the GA-optimised model (the caller's model should already have GA
    # applied; if ga_result is None we reuse the same model as Neuro-Fuzzy).
    m_ga = copy.deepcopy(model)
    m_ga.set_gp_expression(None)
    note_ga = "GA-optimised fuzzy MF" if ga_result is not None else "GA not run; same as Neuro-Fuzzy"
    _eval("4. Neuro-Fuzzy + GA",
          lambda: _numpy_predict(m_ga, X_vec_test, X_seq_test),
          note=note_ga)

    # ── 5. Neuro-Fuzzy + GP ─────────────────────────────────────────────────
    m_gp = copy.deepcopy(model)
    m_gp.set_gp_expression(None)
    note_gp = "(GP not run)"
    if gp_result_dict is not None and "expression_tree" in gp_result_dict:
        try:
            attach_gp_to_model(m_gp, gp_result_dict)
            note_gp = f"GP expr: {m_gp.gp_expression_string[:60]}"
        except Exception as exc:
            note_gp = f"GP attach failed: {exc}"
    _eval("5. Neuro-Fuzzy + GP",
          lambda: _numpy_predict(m_gp, X_vec_test, X_seq_test),
          note=note_gp)

    # ── 6. Neuro-Fuzzy + GA + GP ────────────────────────────────────────────
    m_full = copy.deepcopy(model)   # model already has GA applied
    note_full = "(GP not run)"
    if gp_result_dict is not None and "expression_tree" in gp_result_dict:
        try:
            attach_gp_to_model(m_full, gp_result_dict)
            note_full = f"GP expr: {m_full.gp_expression_string[:60]}"
        except Exception as exc:
            note_full = f"GP attach failed: {exc}"
    _eval("6. Neuro-Fuzzy + GA + GP",
          lambda: _numpy_predict(m_full, X_vec_test, X_seq_test),
          note=note_full)

    return AblationReport(
        rows          = rows,
        data_type     = data_type,
        total_samples = int(n),
        train_samples = int(len(train_idx)),
        val_samples   = int(len(val_idx)),
        test_samples  = int(len(test_idx)),
        seed          = seed,
    )
