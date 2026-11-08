#!/usr/bin/env python3
"""
Final Corrected Evaluation Script
==================================
Produces honest baseline-vs-optimised MSE and MAE on validation and test sets
using the three-way chronological split defined in models/train.py.

Usage:
    .venv/bin/python run_evaluation.py [--tickers A B C] [--epochs N]

No arguments required for offline/CI mode.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.deep_neuro_fuzzy_net import DeepNeuroFuzzyRiskModel
from models.genetic_optimizer import optimize_model, load_ga_params
from models.train import (
    TEST_SPLIT,
    TRAIN_SEED,
    VALIDATION_SPLIT,
    build_dataset,
    evaluate_split,
    get_train_val_test_indices,
    seed_everything,
    train_model,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

OFFLINE_TICKERS = ["OFFLINE_A", "OFFLINE_B", "OFFLINE_C", "OFFLINE_D"]


def _sep(label=""):
    w = 70
    if label:
        pad = (w - len(label) - 2) // 2
        print("=" * pad + f" {label} " + "=" * (w - pad - len(label) - 2))
    else:
        print("=" * w)


def main(tickers=None, epochs=3, ga_pop=6, ga_gens=4):
    if tickers is None:
        tickers = OFFLINE_TICKERS

    _sep("CORRECTED EVALUATION AUDIT")
    print(f"PyTorch available : {TORCH_AVAILABLE}")
    print(f"Tickers           : {tickers}")
    print(f"Training epochs   : {epochs}")
    print(f"GA pop/gens       : {ga_pop} x {ga_gens}")
    print()

    _sep("1 - DATASET")
    seed_everything(TRAIN_SEED)
    dataset = build_dataset(tickers=tickers, sequence_length=15, horizon=5, allow_network=False)
    n = len(dataset["X_vec"])

    train_idx, val_idx, test_idx = get_train_val_test_indices(
        n, validation_split=VALIDATION_SPLIT, test_split=TEST_SPLIT, seed=TRAIN_SEED
    )

    print(f"Total samples     : {n}")
    print(f"Train samples     : {len(train_idx)} ({100*len(train_idx)/n:.1f}%)")
    print(f"Val samples       : {len(val_idx)}   ({100*len(val_idx)/n:.1f}%)")
    print(f"Test samples      : {len(test_idx)}  ({100*len(test_idx)/n:.1f}%)")
    print()

    _sep("2 - INDEX DISJOINTNESS ASSERTION")
    tr, va, te = set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())
    assert len(tr & va) == 0, f"FAIL: train+val has {len(tr & va)} shared indices"
    assert len(tr & te) == 0, f"FAIL: train+test has {len(tr & te)} shared indices"
    assert len(va & te) == 0, f"FAIL: val+test has {len(va & te)} shared indices"
    assert len(tr) + len(va) + len(te) == n, "FAIL: union != {0,...,N-1}"
    print("PASS  train & val  == {} (0 shared samples)")
    print("PASS  train & test == {} (0 shared samples)")
    print("PASS  val   & test == {} (0 shared samples)")
    print(f"PASS  union covers all {n} rows, no gaps")
    print()

    _sep("3 - TEMPORAL ORDERING ASSERTION")
    max_trainval = max(int(train_idx.max()), int(val_idx.max()))
    min_test     = int(test_idx.min())
    assert min_test > max_trainval, \
        f"FAIL: min(test)={min_test} not > max(train+val)={max_trainval}"
    print(f"PASS  max(train+val) = {max_trainval}")
    print(f"PASS  min(test)      = {min_test}")
    print(f"      Every test index is strictly later than every train/val index.")
    print()

    _sep("4 - NEURAL NETWORK TRAINING")
    seed_everything(TRAIN_SEED)
    model = DeepNeuroFuzzyRiskModel()
    history = train_model(model, dataset, epochs=epochs, batch_size=64)
    print(f"Trained {epochs} epoch(s).")
    if history.get("val_mae"):
        print(f"Final train MAE : {history['train_mae'][-1]:.4f}")
        print(f"Final val MAE   : {history['val_mae'][-1]:.4f}")
    print()

    _sep("5 - BASELINE EVALUATION (pre-GA)")
    baseline_val  = evaluate_split(model, dataset, val_idx)
    baseline_test = evaluate_split(model, dataset, test_idx)
    print(f"Baseline | VAL   MSE={baseline_val['mse']:8.4f}  MAE={baseline_val['mae']:8.4f}  n={baseline_val['n']}")
    print(f"Baseline | TEST  MSE={baseline_test['mse']:8.4f}  MAE={baseline_test['mae']:8.4f}  n={baseline_test['n']}")
    print()

    _sep("6 - GA OPTIMISATION (val set only; test never exposed)")
    result = optimize_model(model, dataset, population_size=ga_pop, generations=ga_gens, seed=42)
    meta = result["metadata"]
    print(f"GA baseline fitness : {meta['baseline_fitness']:.6f}")
    print(f"GA best fitness     : {meta['best_fitness']:.6f}")
    print(f"GA improved         : {meta['improved']}")
    print(f"GA val_samples      : {meta['val_samples']}")
    print(f"GA test_samples     : {meta['test_samples']} (untouched)")
    print()

    _sep("7 - OPTIMISED EVALUATION (post-GA, same sets)")
    optimised_val  = evaluate_split(model, dataset, val_idx)
    optimised_test = evaluate_split(model, dataset, test_idx)
    print(f"Optimised| VAL   MSE={optimised_val['mse']:8.4f}  MAE={optimised_val['mae']:8.4f}  n={optimised_val['n']}")
    print(f"Optimised| TEST  MSE={optimised_test['mse']:8.4f}  MAE={optimised_test['mae']:8.4f}  n={optimised_test['n']}")
    print()

    _sep("8 - SUMMARY TABLE")
    hdr = f"{'Set':<8} {'Stage':<12} {'MSE':>10} {'MAE':>10} {'N':>8}"
    print(hdr)
    print("-" * len(hdr))
    rows = [
        ("Val",  "Baseline",  baseline_val),
        ("Val",  "Optimised", optimised_val),
        ("Test", "Baseline",  baseline_test),
        ("Test", "Optimised", optimised_test),
    ]
    for s, st, m in rows:
        print(f"{s:<8} {st:<12} {m['mse']:>10.4f} {m['mae']:>10.4f} {m['n']:>8}")
    print()

    _sep("9 - ROUND-TRIP WEIGHT VERIFICATION")
    import tempfile
    tmp_w = Path(tempfile.mktemp(suffix=".pt"))
    try:
        torch.save(model.state_dict(), tmp_w)
        fresh = DeepNeuroFuzzyRiskModel()
        state = torch.load(tmp_w, map_location="cpu", weights_only=True)
        fresh.load_state_dict(state)
        from models.genetic_optimizer import GA_PARAMS_PATH
        if GA_PARAMS_PATH.exists():
            load_ga_params(fresh)
        X_vec = torch.tensor(dataset["X_vec"][test_idx], dtype=torch.float32)
        X_seq = torch.tensor(dataset["X_seq"][test_idx], dtype=torch.float32)
        model.eval(); fresh.eval()
        with torch.no_grad():
            p_a, _, _, _ = model(X_vec, X_seq)
            p_b, _, _, _ = fresh(X_vec, X_seq)
        max_diff = float((p_a - p_b).abs().max().item())
        print(f"Max abs diff (original vs reloaded): {max_diff:.2e}")
        print("PASS  Round-trip reproduces identical predictions." if max_diff < 1e-3
              else f"WARN  max_diff={max_diff:.4f} exceeds 1e-3")
    finally:
        tmp_w.unlink(missing_ok=True)

    _sep("AUDIT COMPLETE")
    return dict(baseline_val=baseline_val, baseline_test=baseline_test,
                optimised_val=optimised_val, optimised_test=optimised_test)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--epochs",  type=int, default=3)
    p.add_argument("--ga-pop",  type=int, default=6)
    p.add_argument("--ga-gens", type=int, default=4)
    a = p.parse_args()
    main(tickers=a.tickers, epochs=a.epochs, ga_pop=a.ga_pop, ga_gens=a.ga_gens)
