"""
Research-Grade Evaluation and Baseline Engine.

Computes comprehensive performance metrics for continuous risk scores, discrete
risk categories, and future drawdown predictions. Implements competitive baseline
models (Linear Regression, Random Forest, XGBoost) and ensures fair, leakage-free
evaluation on identical chronological test splits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

try:
    import xgboost as xgb
    # Test initialization to ensure libomp runtime is present
    _test_xgb = xgb.XGBRegressor(n_estimators=1)
    XGB_AVAILABLE = True
except Exception:
    xgb = None
    XGB_AVAILABLE = False


@dataclass
class EvaluationMetrics:
    # Continuous Risk Score
    score_mae: float
    score_mse: float
    score_rmse: float
    score_r2: float

    # Categorical Classification
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion_mat: List[List[int]]

    # Future Maximum Drawdown
    mdd_mae: float
    mdd_rmse: float
    mdd_r2: float

    # Computational / Complexity
    latency_ms_per_sample: float
    param_count: int = 0
    model_size_kb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_mae": round(self.score_mae, 4),
            "score_mse": round(self.score_mse, 4),
            "score_rmse": round(self.score_rmse, 4),
            "score_r2": round(self.score_r2, 4),
            "accuracy": round(self.accuracy, 4),
            "precision_macro": round(self.precision_macro, 4),
            "recall_macro": round(self.recall_macro, 4),
            "f1_macro": round(self.f1_macro, 4),
            "precision_weighted": round(self.precision_weighted, 4),
            "recall_weighted": round(self.recall_weighted, 4),
            "f1_weighted": round(self.f1_weighted, 4),
            "confusion_matrix": self.confusion_mat,
            "mdd_mae": round(self.mdd_mae, 4),
            "mdd_rmse": round(self.mdd_rmse, 4),
            "mdd_r2": round(self.mdd_r2, 4),
            "latency_ms": round(self.latency_ms_per_sample, 4),
            "param_count": self.param_count,
            "model_size_kb": round(self.model_size_kb, 2),
        }


def compute_metrics(
    y_score_true: np.ndarray,
    y_score_pred: np.ndarray,
    y_cat_true: np.ndarray,
    y_cat_pred: np.ndarray,
    y_mdd_true: np.ndarray,
    y_mdd_pred: np.ndarray,
    latency_ms: float = 0.0,
    param_count: int = 0,
    model_size_kb: float = 0.0,
) -> EvaluationMetrics:
    """Computes all required financial risk prediction metrics."""
    s_true = np.asarray(y_score_true, dtype=float).ravel()
    s_pred = np.asarray(y_score_pred, dtype=float).ravel()

    c_true = np.asarray(y_cat_true, dtype=int).ravel()
    c_pred = np.asarray(y_cat_pred, dtype=int).ravel()

    m_true = np.asarray(y_mdd_true, dtype=float).ravel()
    m_pred = np.asarray(y_mdd_pred, dtype=float).ravel()

    # Continuous score metrics
    score_mae = float(mean_absolute_error(s_true, s_pred))
    score_mse = float(mean_squared_error(s_true, s_pred))
    score_rmse = float(np.sqrt(score_mse))
    score_r2 = float(r2_score(s_true, s_pred)) if np.var(s_true) > 1e-8 else 0.0

    # Categorical classification metrics (classes 0, 1, 2, 3)
    acc = float(accuracy_score(c_true, c_pred))
    prec_mac = float(precision_score(c_true, c_pred, average="macro", zero_division=0))
    rec_mac = float(recall_score(c_true, c_pred, average="macro", zero_division=0))
    f1_mac = float(f1_score(c_true, c_pred, average="macro", zero_division=0))

    prec_wt = float(precision_score(c_true, c_pred, average="weighted", zero_division=0))
    rec_wt = float(recall_score(c_true, c_pred, average="weighted", zero_division=0))
    f1_wt = float(f1_score(c_true, c_pred, average="weighted", zero_division=0))

    labels = [0, 1, 2, 3]
    cm = confusion_matrix(c_true, c_pred, labels=labels).tolist()

    # Drawdown metrics
    mdd_mae = float(mean_absolute_error(m_true, m_pred))
    mdd_mse = float(mean_squared_error(m_true, m_pred))
    mdd_rmse = float(np.sqrt(mdd_mse))
    mdd_r2 = float(r2_score(m_true, m_pred)) if np.var(m_true) > 1e-8 else 0.0

    return EvaluationMetrics(
        score_mae=score_mae,
        score_mse=score_mse,
        score_rmse=score_rmse,
        score_r2=score_r2,
        accuracy=acc,
        precision_macro=prec_mac,
        recall_macro=rec_mac,
        f1_macro=f1_mac,
        precision_weighted=prec_wt,
        recall_weighted=rec_wt,
        f1_weighted=f1_wt,
        confusion_mat=cm,
        mdd_mae=mdd_mae,
        mdd_rmse=mdd_rmse,
        mdd_r2=mdd_r2,
        latency_ms_per_sample=latency_ms,
        param_count=param_count,
        model_size_kb=model_size_kb,
    )


class BaselineEvaluator:
    """
    Fits and evaluates standard competitive baselines on flattened financial features.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _prepare_tabular_features(self, X_vec: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
        """
        Combines point-in-time vector with sequence statistics (mean, std, min, max)
        to give tabular baselines access to both current state and temporal trend.
        """
        seq_mean = np.mean(X_seq, axis=1)
        seq_std = np.std(X_seq, axis=1)
        seq_min = np.min(X_seq, axis=1)
        seq_max = np.max(X_seq, axis=1)
        # Combined tabular features (4 + 4 + 4 + 4 + 4 = 20 features)
        return np.column_stack([X_vec, seq_mean, seq_std, seq_min, seq_max])

    def evaluate_all_baselines(
        self,
        X_vec_tr: np.ndarray,
        X_seq_tr: np.ndarray,
        y_score_tr: np.ndarray,
        y_cat_tr: np.ndarray,
        y_mdd_tr: np.ndarray,
        X_vec_te: np.ndarray,
        X_seq_te: np.ndarray,
        y_score_te: np.ndarray,
        y_cat_te: np.ndarray,
        y_mdd_te: np.ndarray,
    ) -> Dict[str, EvaluationMetrics]:
        """Runs Linear Regression, Random Forest, and XGBoost baselines."""
        X_tab_tr = self._prepare_tabular_features(X_vec_tr, X_seq_tr)
        X_tab_te = self._prepare_tabular_features(X_vec_te, X_seq_te)

        ys_tr = y_score_tr.ravel()
        yc_tr = y_cat_tr.ravel()
        ym_tr = y_mdd_tr.ravel()

        results: Dict[str, EvaluationMetrics] = {}

        # 1. Linear Regression (Ridge)
        t0 = time.time()
        ridge_score = Ridge(alpha=1.0, random_state=self.seed).fit(X_tab_tr, ys_tr)
        ridge_mdd = Ridge(alpha=1.0, random_state=self.seed).fit(X_tab_tr, ym_tr)
        # Category from score bucket
        t_fit = time.time() - t0

        t0_inf = time.time()
        pred_score_ridge = np.clip(ridge_score.predict(X_tab_te), 0.0, 100.0)
        pred_mdd_ridge = np.clip(ridge_mdd.predict(X_tab_te), 0.0, 1.0)
        lat_ridge = (time.time() - t0_inf) / max(1, len(X_tab_te)) * 1000.0

        pred_cat_ridge = np.array([
            0 if s < 25.0 else 1 if s < 50.0 else 2 if s < 75.0 else 3
            for s in pred_score_ridge
        ])
        results["Linear Regression"] = compute_metrics(
            y_score_true=y_score_te,
            y_score_pred=pred_score_ridge,
            y_cat_true=y_cat_te,
            y_cat_pred=pred_cat_ridge,
            y_mdd_true=y_mdd_te,
            y_mdd_pred=pred_mdd_ridge,
            latency_ms=lat_ridge,
            param_count=X_tab_tr.shape[1] * 2,
        )

        # 2. Random Forest
        rf_score = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=self.seed, n_jobs=-1)
        rf_cat = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=self.seed, n_jobs=-1)
        rf_mdd = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=self.seed, n_jobs=-1)

        rf_score.fit(X_tab_tr, ys_tr)
        rf_cat.fit(X_tab_tr, yc_tr)
        rf_mdd.fit(X_tab_tr, ym_tr)

        t0_inf = time.time()
        pred_score_rf = np.clip(rf_score.predict(X_tab_te), 0.0, 100.0)
        pred_cat_rf = rf_cat.predict(X_tab_te)
        pred_mdd_rf = np.clip(rf_mdd.predict(X_tab_te), 0.0, 1.0)
        lat_rf = (time.time() - t0_inf) / max(1, len(X_tab_te)) * 1000.0

        results["Random Forest"] = compute_metrics(
            y_score_true=y_score_te,
            y_score_pred=pred_score_rf,
            y_cat_true=y_cat_te,
            y_cat_pred=pred_cat_rf,
            y_mdd_true=y_mdd_te,
            y_mdd_pred=pred_mdd_rf,
            latency_ms=lat_rf,
            param_count=3000,
        )

        # 3. XGBoost
        if XGB_AVAILABLE:
            xgb_score = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=self.seed)
            xgb_cat = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=self.seed)
            xgb_mdd = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=self.seed)

            xgb_score.fit(X_tab_tr, ys_tr)
            xgb_cat.fit(X_tab_tr, yc_tr)
            xgb_mdd.fit(X_tab_tr, ym_tr)

            t0_inf = time.time()
            pred_score_xgb = np.clip(xgb_score.predict(X_tab_te), 0.0, 100.0)
            pred_cat_xgb = xgb_cat.predict(X_tab_te)
            pred_mdd_xgb = np.clip(xgb_mdd.predict(X_tab_te), 0.0, 1.0)
            lat_xgb = (time.time() - t0_inf) / max(1, len(X_tab_te)) * 1000.0

            results["XGBoost"] = compute_metrics(
                y_score_true=y_score_te,
                y_score_pred=pred_score_xgb,
                y_cat_true=y_cat_te,
                y_cat_pred=pred_cat_xgb,
                y_mdd_true=y_mdd_te,
                y_mdd_pred=pred_mdd_xgb,
                latency_ms=lat_xgb,
                param_count=2500,
            )

        return results
