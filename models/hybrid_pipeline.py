"""
Hybrid Pipeline: Unified GA + GP Evolutionary Optimization Workflow.

Orchestrates the full optimization sequence:
    1.  Build (or accept) a labelled dataset with three-way chronological splits.
    2.  Run GA  -- optimize 24 Gaussian MF parameters (fuzzy membership).
    3.  Run GP  -- discover a symbolic risk expression on training data,
                  select by validation fitness, NEVER touch the test split.
    4.  Attach the evolved GP expression to the model (model.set_gp_expression).
    5.  Save model weights  + GA chromosome  + GP expression tree to disk.
    6.  Return a unified result dict for CLI display, Streamlit UI, and tests.

Public API
----------
    run_hybrid_pipeline(model, dataset, ...)  ->  HybridResult
    load_full_model(model)                    ->  bool
    evaluate_hybrid(model, dataset, test_idx) ->  dict
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from config.settings import DATA_DIR, GA_CONFIG
from models.genetic_programming import (
    GP_CONFIG,
    GP_SAVE_PATH,
    attach_gp_to_model,
    run_gp,
    save_gp_result,
)
from models.train import (
    TEST_SPLIT,
    TRAIN_SEED,
    VALIDATION_SPLIT,
    get_train_val_test_indices,
    load_gp_expression,
    load_weights,
    save_gp_expression,
    save_weights,
)

HYBRID_RESULT_PATH = Path(DATA_DIR) / "hybrid_pipeline_result.json"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    """All diagnostics from one complete GA + GP run."""

    # GA fields
    ga_baseline_fitness:  float
    ga_best_fitness:      float
    ga_fitness_gain:      float
    ga_improved:          bool
    ga_chromosome_length: int
    ga_history:           List[float]

    # GP fields
    gp_best_fitness:      float
    gp_val_mse:           float
    gp_val_r2:            float
    gp_expression:        str
    gp_node_count:        int
    gp_fitness_history:   List[float]
    gp_complexity_history: List[int]

    # Split info
    train_samples: int
    val_samples:   int
    test_samples:  int
    total_samples: int

    # Timing
    elapsed_seconds: float
    seed: int

    def summary(self) -> str:
        lines = [
            "=== Hybrid GA + GP Pipeline Result ===",
            f"  GA  baseline  : {self.ga_baseline_fitness:.6f}",
            f"  GA  best      : {self.ga_best_fitness:.6f}  ({'improved' if self.ga_improved else 'no improvement'})",
            f"  GP  fitness   : {self.gp_best_fitness:.6f}",
            f"  GP  val MSE   : {self.gp_val_mse:.3f}",
            f"  GP  val R2    : {self.gp_val_r2:.4f}",
            f"  GP  nodes     : {self.gp_node_count}",
            f"  GP  expression: {self.gp_expression}",
            f"  Split         : {self.train_samples}/{self.val_samples}/{self.test_samples} "
            f"(train/val/test)",
            f"  Elapsed       : {self.elapsed_seconds:.1f}s",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ga": {
                "baseline_fitness":  self.ga_baseline_fitness,
                "best_fitness":      self.ga_best_fitness,
                "fitness_gain":      self.ga_fitness_gain,
                "improved":          self.ga_improved,
                "chromosome_length": self.ga_chromosome_length,
                "history":           self.ga_history,
            },
            "gp": {
                "best_fitness":       self.gp_best_fitness,
                "val_mse":            self.gp_val_mse,
                "val_r2":             self.gp_val_r2,
                "expression":         self.gp_expression,
                "node_count":         self.gp_node_count,
                "fitness_history":    self.gp_fitness_history,
                "complexity_history": self.gp_complexity_history,
            },
            "splits": {
                "train_samples": self.train_samples,
                "val_samples":   self.val_samples,
                "test_samples":  self.test_samples,
                "total_samples": self.total_samples,
            },
            "elapsed_seconds": self.elapsed_seconds,
            "seed":            self.seed,
        }


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_hybrid_pipeline(
    model,
    dataset: Dict[str, np.ndarray],
    ga_config: Dict[str, Any] = GA_CONFIG,
    gp_config: Dict[str, Any] = GP_CONFIG,
    seed: int = TRAIN_SEED,
    ga_progress: Optional[Callable[[int, int, float], None]] = None,
    gp_progress: Optional[Callable[[int, int, float], None]] = None,
    skip_ga: bool = False,
    skip_gp: bool = False,
    save_to_disk: bool = True,
) -> HybridResult:
    """
    Run the complete GA -> GP optimization pipeline on *dataset*.

    The function uses the same three-way chronological split
    (get_train_val_test_indices) for both evolutionary algorithms so that
    neither GA nor GP ever observes the held-out test rows.

    Workflow
    --------
    1. Compute train/val/test indices from *dataset*.
    2. [Optional] Run GA to optimize 24 fuzzy MF parameters.
    3. [Optional] Run GP to discover a symbolic risk expression.
    4. Attach the evolved GP expression to *model* via set_gp_expression().
    5. Save model weights, GA chromosome, and GP expression tree.

    Args:
        model      : DeepNeuroFuzzyRiskModel instance (must already be trained).
        dataset    : Dict returned by build_dataset() -- must contain X_vec, X_seq,
                     y_score, y_cat, y_mdd.
        ga_config  : GA hyperparameter dict.
        gp_config  : GP hyperparameter dict.
        seed       : Master random seed (used for both GA and GP).
        ga_progress: Optional callback(gen, total, best_fitness) for GA.
        gp_progress: Optional callback(gen, total, best_fitness) for GP.
        skip_ga    : If True, GA step is skipped (uses model as-is).
        skip_gp    : If True, GP step is skipped (no expression attached).

    Returns:
        HybridResult dataclass.
    """
    from models.genetic_optimizer import optimize_model as ga_optimize

    t0 = time.time()

    n = len(dataset["X_vec"])
    if "split_indices" in dataset:
        train_idx = dataset["split_indices"]["train"]
        val_idx   = dataset["split_indices"]["val"]
        test_idx  = dataset["split_indices"]["test"]
    else:
        train_idx, val_idx, test_idx = get_train_val_test_indices(
            n,
            validation_split=VALIDATION_SPLIT,
            test_split=TEST_SPLIT,
            seed=seed,
        )

    # -- 1. GA ---------------------------------------------------------------
    if not skip_ga:
        ga_result = ga_optimize(
            model=model,
            dataset=dataset,
            population_size=int(ga_config.get("population_size", 20)),
            generations=int(ga_config.get("generations", 15)),
            seed=seed,
            progress_callback=ga_progress,
        )
        meta          = ga_result.get("metadata", {})
        ga_baseline   = float(meta.get("baseline_fitness", 0.0))
        ga_best       = float(meta.get("best_fitness", 0.0))
        ga_improved   = bool(meta.get("improved", False))
        ga_history    = list(ga_result.get("history", []))
        chromosome    = ga_result.get("chromosome", [])
        ga_chr_len    = int(len(chromosome)) if hasattr(chromosome, "__len__") else 24
    else:
        ga_baseline = ga_best = 0.0
        ga_improved = False
        ga_history  = []
        ga_chr_len  = 24

    # -- 2. GP ---------------------------------------------------------------
    if not skip_gp:
        gp_result = run_gp(
            dataset=dataset,
            config=gp_config,
            seed=seed,
            progress_callback=gp_progress,
        )
        gp_fitness   = float(gp_result["fitness"])
        gp_val_mse   = float(gp_result["val_mse"])
        gp_val_r2    = float(gp_result["val_r2"])
        gp_expr      = str(gp_result["expression"])
        gp_nodes     = int(gp_result["node_count"])
        gp_fit_hist  = list(gp_result["fitness_history"])
        gp_cplx_hist = list(gp_result["complexity_history"])

        # Attach to model so the GP signal flows through every forward pass
        attach_gp_to_model(model, gp_result)

        # Save GP result (full JSON) and the raw expression tree (for reload)
        if save_to_disk:
            save_gp_result(gp_result)
            save_gp_expression(model)
    else:
        gp_fitness = gp_val_mse = gp_val_r2 = 0.0
        gp_expr    = "(none)"
        gp_nodes   = 0
        gp_fit_hist = gp_cplx_hist = []

    # -- 3. Persist model weights (PyTorch only; no-op in NumPy fallback) ----
    if save_to_disk:
        try:
            save_weights(model)
        except Exception:
            pass   # NumPy fallback: no weights to save

    elapsed = time.time() - t0

    result = HybridResult(
        ga_baseline_fitness   = ga_baseline,
        ga_best_fitness       = ga_best,
        ga_fitness_gain       = ga_best - ga_baseline,
        ga_improved           = ga_improved,
        ga_chromosome_length  = ga_chr_len,
        ga_history            = ga_history,
        gp_best_fitness       = gp_fitness,
        gp_val_mse            = gp_val_mse,
        gp_val_r2             = gp_val_r2,
        gp_expression         = gp_expr,
        gp_node_count         = gp_nodes,
        gp_fitness_history    = gp_fit_hist,
        gp_complexity_history = gp_cplx_hist,
        train_samples         = int(len(train_idx)),
        val_samples           = int(len(val_idx)),
        test_samples          = int(len(test_idx)),
        total_samples         = int(n),
        elapsed_seconds       = elapsed,
        seed                  = seed,
    )

    # Save summary JSON
    HYBRID_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HYBRID_RESULT_PATH, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)

    return result


# ---------------------------------------------------------------------------
# Full-model loader (weights + GP expression)
# ---------------------------------------------------------------------------

def load_full_model(model) -> bool:
    """
    Load both neural-network weights and the persisted GP expression.

    Returns True when at least the NN weights were loaded successfully.
    The GP expression is loaded silently; its absence is not an error.
    """
    weights_ok = load_weights(model)
    load_gp_expression(model)   # no-op if file absent
    return weights_ok


# ---------------------------------------------------------------------------
# Held-out test evaluation
# ---------------------------------------------------------------------------

def evaluate_hybrid(
    model,
    dataset: Dict[str, np.ndarray],
    test_idx: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate the full hybrid model on the held-out test split.

    If *test_idx* is None the function recomputes it via
    get_train_val_test_indices so the caller does not need to track it.

    Returns a dict with keys: mse, mae, n, gp_expression.

    IMPORTANT: this function must only be called for final reporting.
    Neither GA nor GP should ever see these rows.
    """
    from models.train import evaluate_split

    if test_idx is None:
        n = len(dataset["X_vec"])
        _, _, test_idx = get_train_val_test_indices(
            n,
            validation_split=VALIDATION_SPLIT,
            test_split=TEST_SPLIT,
            seed=TRAIN_SEED,
        )

    metrics = evaluate_split(model, dataset, test_idx)
    metrics["gp_expression"] = getattr(model, "gp_expression_string", "(none)")
    return metrics
