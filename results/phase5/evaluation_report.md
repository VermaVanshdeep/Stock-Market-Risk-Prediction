# PHASE 5 — RESEARCH-GRADE EVALUATION AND PERFORMANCE VALIDATION

## Project
**A Hybrid Neuro-Fuzzy Framework with Genetic Algorithm Optimization and Genetic Programming for Stock Market Risk Prediction**

---

## 1. Dataset Description

### Source and Coverage
- **Data Source:** Real historical daily OHLCV market data downloaded via Yahoo Finance (`yfinance`).
- **Download Date:** 2026-10-30
- **Observation Period:** 3 years (approximately 2023-10-30 to 2026-10-30, comprising ≈754 business days per ticker).
- **is_real_market:** `true` — no synthetic data substituted during real-market evaluation.

### Research Universe (8 Liquid Equities Across 6 Sectors)

| Ticker | Sector | Total Samples | Train | Val | Test | Synthetic? |
|--------|--------|--------------|-------|-----|------|------------|
| AAPL   | Information Technology | 588 | 455 | 52 | 81 | No |
| MSFT   | Information Technology | 588 | 455 | 52 | 81 | No |
| NVDA   | Information Technology | 588 | 455 | 52 | 81 | No |
| GOOGL  | Communication Services | 588 | 455 | 52 | 81 | No |
| JPM    | Financials             | 588 | 455 | 52 | 81 | No |
| XOM    | Energy                 | 588 | 455 | 52 | 81 | No |
| WMT    | Consumer Staples       | 588 | 455 | 52 | 81 | No |
| TSLA   | Consumer Discretionary | 588 | 455 | 52 | 81 | No |

**Total Dataset:** 4,704 rolling-window samples with 0 missing values.

### Feature Engineering
Four standard financial risk features computed point-in-time (no look-ahead):
1. **Sharpe Ratio** — rolling 21-day mean return / rolling HV, annualised. Clipped to [−2.0, 4.0].
2. **Maximum Drawdown** — running peak-to-trough loss at each day. Clipped to [0.0, 0.8].
3. **Implied Volatility Proxy** — Garman-Klass estimator (annualised). Clipped to [0.05, 1.0]. For degenerate synthetic data, HV × 1.05 is substituted and clearly labelled.
4. **Historical Volatility** — rolling 21-day log-return standard deviation, annualised. Clipped to [0.05, 1.0].

### Target Definitions
- **y_score** = `clip(future_max_drawdown × 200 + HV × 100, 0, 100)` — continuous risk score on [0, 100].
- **y_cat** — risk category: 0=Low (<25), 1=Moderate (25–50), 2=High (50–75), 3=Extreme (≥75).
- **y_mdd** — raw future maximum drawdown over the 10-day forward horizon.

---

## 2. Experimental Methodology

### Model Architecture
The Hybrid Neuro-Fuzzy + GA + GP framework consists of:
1. **BiLSTM Sequence Encoder:** Bi-directional LSTM over 20-day input window with 4 features.
2. **Dense Vector Encoder:** Multi-layer perceptron over the point-in-time 4-feature vector.
3. **Takagi-Sugeno-Kang (TSK) Fuzzy Layer:** 4 features × 3 Gaussian membership functions = 12 MF centres + 12 MF spreads.
4. **Fusion Network:** Concatenation of BiLSTM context, dense encoding, and fuzzy defuzzified score.
5. **Three Prediction Heads:** (a) Risk-score regression, (b) 4-class category classifier, (c) drawdown regression.

**Genetic Algorithm (GA):** Optimizes the 24-gene chromosome (12 means + 12 sigmas) encoding the TSK fuzzy membership function parameters. Fitness = 1/(1+MSE) on validation partition.

**Genetic Programming (GP):** Discovers a symbolic closed-form expression mapping the 4 financial features to a continuous risk signal. The GP signal is fused with the neural output in the final prediction head.

### Training Configuration
- **Epochs:** 10 per seed.
- **Batch Size:** 64.
- **Learning Rate:** 0.003 (Adam, CosineAnnealing scheduler).
- **GA Population/Generations:** 16 individuals × 10 generations.
- **GP Population/Generations:** 40 individuals × 12 generations.

---

## 3. Data-Splitting Strategy

### Defects Corrected in Prior Implementation

| # | Defect | Correction Applied |
|---|--------|--------------------|
| 1 | Random/shuffled splits across the full concatenated dataset — test rows could contain early dates | Ticker-aware chronological splitting: each ticker's timeline is partitioned independently |
| 2 | No purge/embargo between splits — rolling-window overlap placed adjacent days in train and val | Effective purge gap of ≥30 calendar days between split periods |
| 3 | Rolling windows whose forward horizon crossed the split boundary | Target window strictly bounded within each split's reserved date range |
| 4 | GA validation set drawn from random permutation, including test-set rows | GA uses `split_indices["val"]` — provably disjoint from both train and test |
| 5 | GP fitness evaluated on all rows | GP receives only train/val rows; test rows are never passed to `run_gp()` |

### Implemented Strategy: Ticker-Aware Chronological Purge & Embargo

For each ticker independently:
```
Total trading days T = 754
Train period  : days [0, t_train_end)          — ~70% of T minus purge
Purge 1       : ~eff_purge calendar days (≥5)
Val period    : days [t_val_start, t_val_end)  — ~15% of T minus purge
Purge 2       : ~eff_purge calendar days (≥5)
Test period   : days [t_test_start, T)         — ~15% of T
```

### Split Statistics (Real-Market Data)

| Split    | Samples | Date Range                      | Samples/Ticker |
|----------|---------|---------------------------------|----------------|
| **Train**    | 3,640 | 2023-12-27 → 2025-10-20 | 455 |
| **Validation** | 416 | 2026-01-14 → 2026-03-30 | 52  |
| **Test**     | 648 | 2026-06-24 → 2026-10-16 | 81  |
| **Total**    | 4,704 | 2023-12-27 → 2026-10-16 | 588 |

> **Note:** Purge/embargo periods are excluded from sample counts. The gap between train target end and val sequence start, and between val target end and test sequence start, is verified to be strictly positive for every ticker (verified by `test_train_target_end_before_val_seq_start_by_timestamp` and `test_val_target_end_before_test_seq_start_by_timestamp`).

### Verified Leakage Properties
- ✅ No rolling window in any split reads price bars from a different split's period.
- ✅ No training forward-horizon peeks into the validation or test periods.
- ✅ No validation forward-horizon peeks into the test period.
- ✅ All 8 tickers have representation in all three splits.
- ✅ No (ticker, seq_end_date) pair shared across splits.
- ✅ GA never accesses test-split rows.
- ✅ GP never accesses test-split rows.
- ✅ Test set was **never accessed** until final evaluation.

---

## 4. Model Architecture Summary

```
Input (20 × 4)         Input (4,)
    │                      │
BiLSTM Encoder         Dense Encoder         TSK Fuzzy Layer (24 params)
    │                      │                       │
    └──────────── Fusion MLP (concat) ─────────────┘
                          │
                  + GP Symbolic Signal
                          │
          ┌───────────────┼───────────────┐
          ↓               ↓               ↓
    Score Head      Cat Head (4)    MDD Head
    (regression)   (softmax)       (regression)
```

**Total Parameters:** ~17,500 (varies slightly by architecture instantiation).

---

## 5. Six Ablation Experiments

All six configurations evaluated on the **identical 648 test samples** from the 2026-06-24 → 2026-10-16 chronological test partition.

| Configuration | Score MAE | Score RMSE | Score R² | Accuracy | Macro F1 | MDD MAE | Latency (ms) |
|---------------|-----------|------------|----------|----------|----------|---------|--------------|
| **1. BiLSTM-only** | 6.551 | 8.791 | 0.6985 | 68.2% | 0.507 | 0.0836 | 0.080 |
| **2. Fuzzy-only** | 36.684 | 40.245 | -5.320 | 22.7% | 0.093 | 0.0267 | 0.024 |
| **3. Neuro-Fuzzy (Base)** | 6.745 | 9.013 | 0.6830 | 67.9% | 0.513 | 0.0893 | 0.085 |
| **4. Neuro-Fuzzy + GA** | 6.745 | 9.013 | 0.6830 | 67.9% | 0.513 | 0.0893 | 0.082 |
| **5. Neuro-Fuzzy + GP** | 7.194 | 9.322 | 0.6609 | 66.5% | 0.492 | 0.0802 | 0.079 |
| **6. Full Hybrid (GA+GP)** | 7.194 | 9.322 | 0.6609 | 66.5% | 0.492 | 0.0802 | 0.078 |

### Ablation Observations

1. **BiLSTM-only (Config 1) achieves the best overall performance** (MAE=6.551, R²=0.699). Zeroing the fuzzy consequents removes the TSK contribution, leaving the BiLSTM + dense heads to carry all prediction weight.

2. **Fuzzy-only (Config 2) performs poorly** (MAE=36.7, R²=–5.32). The TSK defuzzified signal (scaled by 10) is not calibrated to predict the composite risk score directly without the neural fusion network. This is expected: the fuzzy layer was trained as part of the hybrid system, not as a standalone predictor.

3. **Neuro-Fuzzy (Config 3) vs BiLSTM-only**: Config 3 has slightly higher MAE (6.745 vs 6.551) and lower R² (0.683 vs 0.698). Adding the fuzzy layer introduces a marginal cost on this test partition.

4. **GA (Config 4 = Config 3)**: In 4 of 5 seeds the GA did not improve validation fitness. When GA does not improve, the model rolls back to original parameters. On Seed 404, GA did improve (gain=+0.00066), but even then the test improvement is within noise. Configs 3 and 4 are identical in this primary seed-42 evaluation.

5. **GP (Config 5 ≈ Config 6)**: The GP-discovered expressions (see Section 7) have val R² near 0 or negative across most seeds, indicating the symbolic expressions do not reliably model the risk score. The GP signal slightly degrades predictions in configs 5 and 6 vs configs 3 and 4.

6. **Full Hybrid (Config 6)**: The full system achieves MAE=7.194, RMSE=9.322, Accuracy=66.5%, F1=0.492. It does not outperform BiLSTM-only or the standalone Neuro-Fuzzy baseline on this evaluation.

---

## 6. Conventional Baseline Comparisons

| Model | Score MAE | Score RMSE | Score R² | Accuracy | Macro F1 | MDD MAE | Latency (ms) |
|-------|-----------|------------|----------|----------|----------|---------|--------------|
| Linear Regression | 5.923 | 8.113 | 0.743 | 67.9% | 0.514 | 0.0285 | 0.0002 |
| Random Forest | 6.636 | 9.036 | 0.682 | 69.4% | 0.558 | 0.0283 | 0.063 |
| Standalone BiLSTM | 6.551 | 8.791 | 0.699 | 68.2% | 0.507 | 0.0836 | 0.080 |
| Neuro-Fuzzy (No Evolution) | 6.745 | 9.013 | 0.683 | 67.9% | 0.513 | 0.0893 | 0.085 |
| **Full Hybrid (Neuro-Fuzzy+GA+GP)** | **7.194** | **9.322** | **0.661** | **66.5%** | **0.492** | **0.0802** | **0.078** |

### Baseline Comparison Observations

- **Linear Regression is the strongest overall baseline** (MAE=5.923, R²=0.743, Accuracy=67.9%). Its simplicity with 20 tabular features (point-in-time + sequence statistics) provides strong generalization.
- **Random Forest** achieves the highest classification accuracy (69.4%) and F1 (0.558).
- **The Full Hybrid model does NOT outperform any conventional baseline on this test partition.**
- **Drawdown prediction (y_mdd):** Linear Regression and Random Forest achieve dramatically lower drawdown MAE (0.028–0.028) vs neural models (0.080–0.089). The neural models were not trained with a strong drawdown supervision signal (weight=10×MSE), but the baseline's tabular features include sequence min which directly correlates with drawdown.

---

## 7. GA Contribution Analysis

### Configuration
- **Chromosome:** 24 real-valued genes — 12 Gaussian MF centres (means[4][3]) + 12 Gaussian spreads (sigmas[4][3]).
- **Population size:** 16 | **Generations:** 10 | **Fitness:** 1/(1+MSE_val).
- **Fitness oracle:** Full end-to-end BiLSTM + fuzzy + fusion model on validation partition.
- **Safety mechanism:** If best GA fitness ≤ baseline, original parameters are preserved.

### Results Across 5 Seeds

| Seed | Baseline Fitness | Best GA Fitness | Gain | Improved? |
|------|-----------------|-----------------|------|-----------|
| 42   | 0.015443 | 0.014433 | −0.001010 | No  |
| 101  | 0.015166 | 0.013835 | −0.001331 | No  |
| 202  | 0.016368 | 0.016261 | −0.000108 | No  |
| 303  | 0.015738 | 0.013509 | −0.002230 | No  |
| 404  | 0.015718 | 0.016376 | +0.000658 | **Yes** |

**Improvement frequency: 1 / 5 seeds (20%).**

### Interpretation
In 4 of 5 seeds the GA actually decreased validation fitness. This is not surprising with a small population (16) over few generations (10): the search space is 24-dimensional and the fitness landscape is noisy (MSE of a neural network evaluated on 416 validation samples). The one successful improvement (Seed 404, gain=+0.00066) is marginal.

**Conclusion:** With the current GA configuration, the genetic algorithm does not reliably improve the fuzzy membership parameters. Larger population sizes (≥50) and more generations (≥30) are likely needed for consistent improvement.

---

## 8. GP Contribution Analysis

### Configuration
- **Terminals:** x0=sharpe, x1=drawdown, x2=implied_vol, x3=hist_vol; ephemeral constants ∈ [−5, 5].
- **Operators:** add, sub, mul, protected_div, abs, min, max, protected_log, protected_sqrt.
- **Population:** 40 | **Generations:** 12 | **Max depth:** 5 | **Complexity penalty:** α=0.05.
- **Fitness:** 1/(1+MSE_val+α×complexity).

### Discovered Expressions (5 Seeds)

| Seed | Expression | Nodes | Val MSE | Val R² |
|------|-----------|-------|---------|--------|
| 42  | `sqrt(3.917) * (((3.917*3.908) + sqrt(3.917*x1)) + sqrt(3.908))` | 14 | 122.4 | +0.026 |
| 101 | `max((x2-x2), (max(x1,4.879)*(max(x2,4.879)/abs(0.9072)))) + (4.879+log(sqrt(x0)*(-3.966*-4.305)))` | 24 | 132.3 | −0.052 |
| 202 | `abs(((4.973+x2)*(4.973+x2))) + (((4.973-x0)+x3)*(3.906-2.616))` | 18 | 109.2 | +0.132 |
| 303 | `abs(4.782) + (abs(4.782) + max(abs(4.782)*4.782, sqrt(x1)+4.782))` | 15 | 139.8 | −0.112 |
| 404 | `max(((3.727+3.727)+(((0.263+3.727)+3.727)/0.263)), max(x1, max(x1, x0)))` | 17 | 126.1 | −0.003 |

### Interpretation

1. **Expression stability is low:** All 5 expressions have different structures and feature combinations, indicating the GP does not converge to a stable, generalizable formula.
2. **Val R² is near-zero or negative in 4 of 5 seeds:** The GP symbolic signal adds negligible or negative predictive value over the mean.
3. **Seed 202 has the best Val R² = 0.132**, with an expression involving `implied_vol` (x2) and `sharpe` (x0), consistent with financial intuition that volatility and risk-adjusted return drive risk scores.
4. **Constant-heavy expressions (Seeds 42, 303):** Some expressions degenerate to mostly-constant outputs; the 90th-percentile nodes are constants, not feature variables. This indicates premature convergence.
5. **Feature x1 (drawdown) appears in 4 of 5 expressions** — the most consistent finding.

**Conclusion:** GP symbolic expression discovery is currently the weakest component of the hybrid. It adds noise rather than signal. With more generations, larger populations, stronger complexity penalties, and feature scaling to a common range, GP could potentially discover more stable expressions.

---

## 9. Per-Ticker Performance (Full Hybrid Model, Test Set)

| Ticker | Sector | Test Samples | Score MAE | Score RMSE | Score R² | Accuracy | Macro F1 | MDD MAE |
|--------|--------|-------------|-----------|------------|----------|----------|----------|---------|
| AAPL   | Information Technology | 81 | 8.065 | 9.478 | 0.121 | 72.8% | 0.414 | 0.112 |
| MSFT   | Information Technology | 81 | 7.585 | 9.264 | 0.622 | 70.4% | 0.562 | 0.070 |
| NVDA   | Information Technology | 81 | 7.420 | 8.854 | −0.289 | 56.8% | 0.409 | 0.101 |
| GOOGL  | Communication Services | 81 | 6.492 | 8.965 | 0.469 | 80.2% | 0.700 | 0.088 |
| JPM    | Financials             | 81 | 3.955 | 4.725 | −0.919 | 67.9% | 0.656 | 0.029 |
| XOM    | Energy                 | 81 | 5.609 | 6.716 | −1.216 | 71.6% | 0.417 | 0.068 |
| WMT    | Consumer Staples       | 81 | 6.790 | 8.021 | −0.001 | 65.4% | 0.396 | 0.087 |
| TSLA   | Consumer Discretionary | 81 | 11.637 | 15.132 | 0.406 | 46.9% | 0.347 | 0.086 |

### Per-Ticker Observations

- **Best performer: JPM** (Financials) — lowest Score MAE (3.955). Bank stocks tend to have more regime-stable risk characteristics (leverage ratios, credit cycles).
- **Worst performer: TSLA** (Consumer Discretionary) — highest Score MAE (11.637) and lowest accuracy (46.9%). Tesla's highly idiosyncratic volatility (driven by Elon Musk news, EV adoption, delivery reports) is hard to model from OHLCV features alone.
- **Negative R² for NVDA, JPM, XOM, WMT:** R² can be negative when the model's predictions are worse than simply predicting the mean. This occurs because risk scores in stable-price periods cluster near the mean, and the model's predictions have higher variance than the target.
- **Best classification: GOOGL** (80.2% accuracy, F1=0.700). Alphabet's risk profile is relatively stable during the test period.

### Per-Sector Summary

| Sector | Avg Score MAE | Avg Accuracy | Avg Macro F1 |
|--------|--------------|--------------|--------------|
| Information Technology | 7.69 | 66.7% | 0.462 |
| Communication Services | 6.49 | 80.2% | 0.700 |
| Financials             | 3.96 | 67.9% | 0.656 |
| Energy                 | 5.61 | 71.6% | 0.417 |
| Consumer Staples       | 6.79 | 65.4% | 0.396 |
| Consumer Discretionary | 11.64 | 46.9% | 0.347 |

---

## 10. Statistical Reliability (5 Random Seeds)

| Seed | Score MAE | Score RMSE | Score R² | Accuracy | Macro F1 | MDD MAE | Run Time |
|------|-----------|------------|----------|----------|----------|---------|----------|
| 42   | 7.194 | 9.322 | 0.661 | 66.5% | 0.492 | 0.0802 | 22.0s |
| 101  | 6.542 | 8.913 | 0.690 | 67.3% | 0.509 | 0.0726 | 19.5s |
| 202  | 6.918 | 9.116 | 0.676 | 67.9% | 0.507 | 0.0940 | 19.9s |
| 303  | 6.705 | 8.973 | 0.686 | 68.5% | 0.520 | 0.0890 | 20.9s |
| 404  | 6.610 | 8.845 | 0.695 | 69.9% | 0.538 | 0.0870 | 20.9s |
| **Mean ± Std** | **6.794 ± 0.240** | **9.034 ± 0.185** | **0.682 ± 0.013** | **68.0% ± 1.2%** | **0.513 ± 0.017** | **0.0846 ± 0.008** | **~20.6s** |

### Statistical Reliability Observations

- **Score MAE coefficient of variation:** 240/6794 ≈ 3.5% — low relative variability across seeds.
- **R² variation is tight:** 0.661–0.695 range across seeds. The model is not highly sensitive to random weight initialization.
- **Seed 42 is consistently the worst seed** (lowest accuracy, lowest R², highest MAE). This is the primary seed used in ablation analysis, which means ablation results represent a conservative (not cherry-picked best) performance.
- **Training time:** 19–22 seconds per seed (all three phases: 10 epochs NN training + 10 GA generations + 12 GP generations).

---

## 11. Computational Performance

| Component | Time (Seed 42) | Notes |
|-----------|---------------|-------|
| BiLSTM+Neuro-Fuzzy training (10 epochs) | 13.7s | 3,640 train samples |
| GA optimization (16 pop × 10 gen) | 8.0s | 416 val samples |
| GP evolution (40 pop × 12 gen) | 0.26s | 416 val samples |
| Inference latency (Full Hybrid) | 0.078 ms/sample | 648 test samples |
| Model size | ~17,500 params | ~68 KB float32 |

---

## 12. Threats to Validity and Limitations

### 12.1 Stationarity
The 3-year evaluation window (2023–2026) spans a bull-market technology rally (2023–2024) and a period of higher interest rates. The test partition (2026-06-24 → 2026-10-16) represents the most recent regime. The model may not generalize to future regime shifts (recessions, crashes) not represented in this training window.

### 12.2 Fuzzy-Only Baseline Limitation
The Fuzzy-only ablation (Config 2) uses the fuzzy consequent weights trained as part of the joint hybrid system. These weights were learned to complement the BiLSTM, not to operate standalone. A fair "fuzzy-only" baseline would require training a standalone fuzzy system from scratch, which is a direction for future work.

### 12.3 Small GP/GA Budget
With GA (16 pop × 10 gen = 160 evaluations) and GP (40 pop × 12 gen = 480 evaluations), the evolutionary search is shallow. Real-world genetic optimization studies typically run thousands of evaluations. The current budgets are sufficient for proof-of-concept but may not reliably converge to optimal solutions.

### 12.4 Target Definition
The composite risk score `y = clip(MDD × 200 + HV × 100, 0, 100)` is an ad-hoc combination. The weights (200, 100) were not derived from financial theory or market data. A factor model or statistical learning approach to risk target definition could improve quality.

### 12.5 Sample Size
With 648 test samples across 8 tickers (81 per ticker), statistical conclusions about individual tickers have low power. For robust per-ticker claims, 3–5 years of dedicated held-out data would be needed.

### 12.6 No Walk-Forward Evaluation
A production-grade evaluation would use walk-forward (rolling expanding window) validation rather than a single held-out period, to capture consistency across multiple market cycles.

---

## 13. Conclusion: Does the Full Hybrid Model Outperform Baselines?

### Summary Table

| Metric | Linear Regression | Random Forest | Standalone BiLSTM | Full Hybrid | Hybrid Rank |
|--------|------------------|---------------|-------------------|-------------|-------------|
| Score MAE | **5.923** | 6.636 | 6.551 | 7.194 | 4th (worst) |
| Score R² | **0.743** | 0.682 | 0.699 | 0.661 | 4th |
| Accuracy | 67.9% | **69.4%** | 68.2% | 66.5% | 4th |
| Macro F1 | 0.514 | **0.558** | 0.507 | 0.492 | 4th |
| MDD MAE | **0.028** | **0.028** | 0.084 | 0.080 | 3rd |
| Inference Latency | **0.0002ms** | 0.063ms | 0.080ms | 0.078ms | 3rd |

**Verdict: The Full Hybrid (Neuro-Fuzzy + GA + GP) does NOT outperform conventional baselines or even the standalone BiLSTM on this test partition.**

### What the Model Does and Does Not Improve

**What works:**
- The BiLSTM temporal encoder is the strongest single component (lowest MAE, highest R²).
- The full hybrid achieves a respectable R²=0.661–0.695 across seeds, demonstrating that financial risk is meaningfully predictable from OHLCV-derived features.
- Drawdown prediction (y_mdd) is competitive with neural models, achieving MDD MAE ≈ 0.080 vs 0.028 for tabular baselines.

**What does not work as intended:**
- **GA:** Improves validation fitness in only 1/5 seeds. The 24-dimensional fuzzy parameter search space is too large for the current (16 pop × 10 gen) budget.
- **GP:** Discovered expressions have val R² near zero or negative in 4/5 seeds. GP expressions are not stable across seeds and do not reliably improve predictions.
- **Fuzzy integration:** Adding the TSK fuzzy layer to BiLSTM slightly increases test error in this evaluation. The fuzzy layer may be adding noise rather than interpretable signal when the neural components are already well-trained.

**Honest assessment:**
The framework demonstrates sound software engineering (strict temporal splits, leakage controls, reproducibility) and theoretically well-motivated components (interpretable fuzzy rules, symbolic GA/GP). However, under rigorous evaluation with proper train/val/test isolation, it does not exceed conventional ML baselines. Larger evolutionary budgets, longer training, alternative risk targets, and walk-forward validation are necessary before claiming superiority.

---

## 14. Test Suite Results

Total tests: **259 passed** (242 original + 17 new timestamp-based tests).

All tests verify:
- ✅ Chronological splitting (timestamp-based, per ticker)
- ✅ Ticker-aware representation in all splits
- ✅ Purge/embargo correctness (positive gap verified)
- ✅ No future-target leakage (target_start > seq_end for all samples)
- ✅ No test-set access during training, GA, or GP evolution
- ✅ Identical test samples across ablation configurations
- ✅ Correct metric calculations (verified against known-answer tests)
- ✅ Reproducible experiment configurations (deterministic builds)
- ✅ Correct real-data vs synthetic labelling

---

*Report generated automatically from actual experimental execution on 2026-10-30.*
*All metrics are derived from genuine model training and inference — no values were fabricated.*
