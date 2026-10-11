# About the Project & Technical Implementation Manual

A quantitative **Stock Market Risk Prediction System** that bridges **Soft
Computing** methodology with **Deep Learning** neural architectures. Traditional
quantitative finance relies on crisp mathematical assumptions that fail during
market panics, regime shifts and high-volatility events. By combining
**Takagi-Sugeno Adaptive Neuro-Fuzzy Inference Systems (ANFIS)**, **Genetic
Algorithm (GA) Evolutionary Optimisation** and **Deep Bidirectional LSTMs**, the
system captures linguistic uncertainty, non-linear market dynamics and tail-risk
drawdowns — and can explain every prediction it makes.

---

## Live Demo

A public deployment is hosted on Streamlit Community Cloud:

### **[stock-market-risk-prediction.streamlit.app →](https://stock-market-risk-prediction-yu5vwfnvxrrcuv9qumrgnj.streamlit.app/)**

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [The Flat-Score Problem and How It Was Fixed](#2-the-flat-score-problem-and-how-it-was-fixed)
3. [Soft Computing: Fuzzy Logic & ANFIS](#3-soft-computing-fuzzy-logic--anfis)
4. [Soft Computing: Genetic Algorithm](#4-soft-computing-genetic-algorithm)
5. [Deep Learning: The Fusion Network](#5-deep-learning-the-fusion-network)
6. [Supervised Training Engine](#6-supervised-training-engine)
7. [Financial Risk Metrics](#7-financial-risk-metrics)
8. [Data Pipeline and Sources](#8-data-pipeline-and-sources)
9. [User Interface Design](#9-user-interface-design)
10. [System Architecture](#10-system-architecture)
11. [Repository Sitemap](#11-repository-sitemap)
12. [How to Run](#12-how-to-run)
13. [Limitations and Future Work](#13-limitations-and-future-work)

---

## 1. Executive Overview

The system answers one question — *how risky is this instrument right now?* — and
answers it in two ways at once:

- **A neural network** produces a continuous 0–100 score, a risk band and a
  forward drawdown forecast.
- **A fuzzy rule base** shows *why*, by exposing the membership functions and the
  specific rules that fired.

That duality is the point of the hybrid architecture. A pure deep model is
accurate but opaque; a pure expert system is interpretable but brittle. Fusing
them inherits both properties.

---

## 2. The Flat-Score Problem and How It Was Fixed

This is worth documenting because it shaped the current design.

The original code shipped a `DeepNeuroFuzzyRiskModel` with **randomly
initialised weights and no training step anywhere in the repository**. Every
input produced essentially the same output:

| Profile | Score (untrained) |
|---|---|
| Defensive (Sharpe 2.0, DD 5%, IV 12%, HV 10%) | 46.80 |
| Balanced (1.2, 15%, 25%, 20%) | 46.76 |
| Stressed (0.1, 40%, 55%, 50%) | 46.73 |
| Crisis (−1.2, 65%, 85%, 75%) | 46.83 |
| **Spread** | **0.10** |

The fuzzy layer responded correctly to these inputs — only the deep head was
dead, because the sigmoid output of an untrained network sits at its random
fixed point.

`models/train.py` fixes this by fitting all three heads on real market history.
The same probe now spans **64.87 points** (Defensive 15.95 → Crisis 80.82), with
correct band assignment at every step. A regression test asserts the spread
exceeds 5 points so this bug cannot silently return.

---

## 3. Soft Computing: Fuzzy Logic & ANFIS

### 3.1 Fuzzification

Each crisp metric is mapped into a degree of membership between 0 and 1 using
differentiable Gaussian membership functions:

```
μ(x) = exp( −0.5 · ((x − c) / σ)² )
```

where `c` is the centre (mean) and `σ` the spread of each fuzzy set (Low, Medium,
High). Centres and spreads are stored as trainable parameters.

Configured bounds (`config/settings.py`):

| Input | Low (c, σ) | Medium (c, σ) | High (c, σ) |
|---|---|---|---|
| Sharpe Ratio | (0.0, 0.5) | (1.0, 0.5) | (2.0, 0.5) |
| Max Drawdown | (0.05, 0.05) | (0.20, 0.08) | (0.40, 0.10) |
| Implied Volatility | (0.12, 0.04) | (0.25, 0.06) | (0.45, 0.10) |
| Historical Volatility | (0.12, 0.04) | (0.25, 0.06) | (0.45, 0.10) |

### 3.2 Rule Firing (Product T-norm)

Each rule multiplies the four antecedent memberships:

```
wᵢ = μ_sharpe(x₁) · μ_drawdown(x₂) · μ_iv(x₃) · μ_hv(x₄)
```

The rule base is enumerated in a fixed, shared order (`build_rule_list()`), so a
rule index always maps to the same antecedent tuple — which is what lets the UI
print *"IF Sharpe is Low AND Drawdown is High … THEN risk is elevated"*.

### 3.3 TSK Defuzzification

Fired weights are normalised and combined with linear consequents:

```
Fuzzy Risk Score = Σ(wᵢ · fᵢ(x)) / Σ(wᵢ)
```

where each consequent is `fᵢ(x) = pᵢx₁ + qᵢx₂ + rᵢx₃ + sᵢx₄ + cᵢ`.

### 3.4 Why 16 rules

The full 3⁴ = 81 rule base is pruned to the first 16 in a deterministic
enumeration order. This keeps inference fast and the rule base readable in the
UI without materially changing behaviour, since the unused rules have low-
membership antecedents.

---

## 4. Soft Computing: Genetic Algorithm

The GA optimises the fuzzy membership parameters and feature importance weights.

| Aspect | Detail |
|---|---|
| Chromosome | 28 floats — 12 centres (4 × 3 terms), 12 spreads, 4 feature weights |
| Selection | Tournament selection over random pairs |
| Crossover | Simulated Binary Crossover (p = 0.8) |
| Mutation | Gaussian jitter (p = 0.15) |
| Fitness | `1 / (1 + MSE)` against a seeded benchmark |
| Defaults | Population 20, 15 generations, elite 2 |

`optimize()` accepts an optional `progress_callback(generation, total, fitness)`
so the web UI can drive a live progress bar. The callback is wrapped in a
`try/except` so a UI hiccup can never abort an optimisation run.

---

## 5. Deep Learning: The Fusion Network

`DeepNeuroFuzzyRiskModel` combines three parallel representations of the same
instrument:

| Branch | Input | Architecture | Output |
|---|---|---|---|
| **Fuzzy (soft computing)** | 4 metrics | Gaussian memberships → 16 TSK rules | 1 fuzzy score |
| **Deep sequence** | 20-day metric window | 2-layer bidirectional LSTM, hidden 64, dropout 0.2 | 128 |
| **Dense encoder** | 4 metrics | `4 → 32 → 32` (ReLU) | 32 |

These are concatenated into a fusion network (`161 → 64 → 32`) feeding three
multi-task heads:

- **Risk score** — sigmoid scaled to 0–100
- **Risk classification** — 4 logits over Low / Moderate / High / Extreme
- **Drawdown forecast** — sigmoid, expected forward maximum drawdown

`predict_risk()` accepts an optional `sequence_inputs` window. When omitted the
static vector is broadcast across the 20-step window, preserving the original
vector-only calling convention for the CLI.

---

## 6. Supervised Training Engine

`models/train.py` is what makes the score meaningful.

### 6.1 Rolling feature frame

Rather than recomputing a full metric set per sample, `rolling_metric_frame()`
builds all four metrics vectorised across the whole series, then clips each into
the exact feature domain the fuzzy layer was configured with. This keeps the
sequence channel and the dense vector channel in the **same feature space**,
which is what allows vector-only inference to stay consistent with the trained
model.

### 6.2 Target construction

For every window ending at day `t`, looking `HORIZON = 10` days ahead:

```
future_max_dd = (close[t] − min(close[t+1 … t+10])) / close[t]

y_score = clip(future_max_dd · 200 + HV[t] · 100, 0, 100)
y_cat   = Low (<25) | Moderate (<50) | High (<75) | Extreme (≥75)
y_mdd   = future_max_dd
```

### 6.3 Loss

```
L = MSE(risk_score, y_score) + 2.0 · CE(category, y_cat) + 10.0 · MSE(mdd, y_mdd)
```

with gradient-norm clipping at 5.0 and a cosine-annealing learning rate.

### 6.4 Dataset

| Property | Value |
|---|---|
| Universe | 20 tickers (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, XOM, WMT, UNH, JNJ, PG, HD, BAC, CVX, NFLX, AMD, COST, PEP) |
| Samples | ~24,100 |
| Sequence length | 20 |
| Forward horizon | 10 |
| Train / validation split | 85 / 15 |

Each ticker is fetched live via yfinance; any that fail fall back to a
deterministic synthetic series with a calibrated volatility level, so training
still runs offline and stays reproducible.

### 6.5 Persistence and reproducibility

Weights are saved to `data_cache/risk_model_weights.pt` and committed, so the
CLI, every browser session and the hosted demo all share one fitted model.
`seed_everything()` makes weight initialisation deterministic — without it the
"untrained" score changed between process starts.

### 6.6 Measured performance

| Metric | Value |
|---|---|
| Validation MAE | ~6.4 score points |
| Validation category accuracy | ~73% |
| Score standard deviation across the dataset | ~10.5 |
| Regime probe spread (Defensive → Crisis) | 64.87 |

---

## 7. Financial Risk Metrics

| Metric | Formula | Notes |
|---|---|---|
| **Sharpe Ratio** | `(Annualised Return − Rf) / Annualised Volatility` | Rf = 4% |
| **Maximum Drawdown** | `(Peak − Trough) / Peak` | Running `cummax` over the window |
| **Implied Volatility** | `√(0.5·ln²(H/L) − (2ln2 − 1)·ln²(C/O))`, mean over 21 days × √252 | Garman-Klass estimator |
| **Historical Volatility** | `std(ln(Pₜ/Pₜ₋₁)) × √252` | 21-day window |
| **Sortino Ratio** | `(Annualised Return − Rf) / Annualised Downside Volatility` | Negative returns only; capped at 5.0 |
| **Value at Risk 95%** | `|5th percentile of daily returns|` | Historical simulation |

Global constants (`config/settings.py`): risk-free rate `0.04`, annualisation
factor `252`, HV window `21`, drawdown window `252`, VaR confidence `0.95`.

---

## 8. Data Pipeline and Sources

### 8.1 Loading

`StockDataLoader.fetch_stock_data()` downloads via yfinance and handles
multi-index columns. On any failure it falls back to
`SyntheticStockGenerator`, which produces OHLCV paths from **Merton Jump
Diffusion** combined with **Heston stochastic volatility** — calibrated per
ticker so each synthetic instrument keeps a distinct volatility character.

### 8.2 Source matrix

| Source | Used for | Failure behaviour |
|---|---|---|
| Yahoo Finance | OHLCV history | Falls back to the synthetic simulator; UI labels data as synthetic |
| Google Finance | Live price quote | Validated (title must contain the ticker, price must be plausible) then shown; silently omitted on failure |
| Google Sheets | Custom price series | Explicit error messages for private sheets, bad dates or too few rows |
| Google Suggest | Autocomplete | Falls back to the built-in company map |

### 8.3 Company resolution

`resolve_free_text()` and `company_suggestions()` resolve free text to a ticker
in ranked tiers, longest-match-first:

1. Exact ticker symbol
2. Company-name prefix (`nvi` → NVIDIA)
3. Symbol prefix (`brk` → Berkshire Hathaway)
4. Word-boundary match (`eli` → Eli Lilly)
5. Substring match, 3+ characters only
6. Live Google suggestions, kept only when they resolve to a known listing

Backed by a map of **122 listings** and **79 exchange assignments**. Handle
aliases (`BRK.B` → `BRK-B`) are applied before fetching.

### 8.4 Google Sheets ingestion

Only **Date** and one of *Close / Adj Close / Price / Value* are required.
*Open / High / Low / Volume* are derived from Close when missing — and the UI
says so explicitly, because derived OHLC collapses the Garman-Klass implied
volatility to zero. At least 30 usable rows are required for the 21-day windows.

---

## 9. User Interface Design

### 9.1 Architecture

The web UI is a **presentation layer only**. `ui/services.py` exposes cached,
UI-ready wrappers (`predict_risk()`, `fetch_google_sheet()`, `run_genetic_algorithm()`,
`run_tests()`, `build_pdf_bytes()`) over the same engine modules the CLI uses.
No business logic is duplicated under `ui/`.

### 9.2 Rendering

- **Streamlit** (v1.65) with **Plotly** figures, themed strictly in blue on white
  via `.streamlit/config.toml` and a custom stylesheet
- **Caching** — `st.cache_resource` for the model singleton, `st.cache_data` for
  price history, Google quotes and suggestions
- **Blue-only risk ramp** — Low `#93c5fd` → Extreme `#172554`, giving intensity
  without introducing a second hue

### 9.3 Responsiveness and accessibility

- Stat cards render through a **CSS grid with `data-cols` media queries** rather
  than nested `st.columns`, so cards keep usable width at any container size
  (2 columns ≤1500px, 1 column ≤860px)
- Chart margins are computed from content: extra left padding whenever a y-axis
  title is present, extra headroom whenever a title and legend coexist
- Suggestion labels are staggered per term so two terms with identical membership
  never print on top of each other

### 9.4 Autocomplete

The company search dropdown is styled via `div[class*="st-key-sugg_"]`, which
Streamlit generates for keyed widgets. This scopes the styling to the panel
alone. The panel hangs beneath the input at matching width, stays open while
typing (the analysis updates live), and closes only on an explicit choice.

### 9.5 Honesty in the interface

The UI never implies more precision than the model has. A "Model status" note
reports whether the head is trained, and the dashboard states the price history —
not the single Google quote — is what drives the risk metrics.

---

## 10. System Architecture

| Layer | Files | Responsibility |
|---|---|---|
| **Presentation** | `app.py`, `ui/**` | Streamlit shell, screens, charts, theme |
| **CLI** | `main.py`, `test_runner.py`, `generate_pdf_report.py` | Rich terminal menus and reporting |
| **Soft computing core** | `models/fuzzy_layer.py`, `models/genetic_optimizer.py` | TSK ANFIS layer, evolutionary optimiser |
| **Deep learning core** | `models/deep_neuro_fuzzy_net.py`, `models/train.py` | BiLSTM fusion network, three heads, training loop |
| **Data engine** | `data/data_loader.py`, `data/feature_extractor.py` | Market data + synthetic fallback, feature vectors and sequences |
| **Math engine** | `utils/financial_metrics.py` | Sharpe, MDD, IV, HV, VaR, Sortino |
| **Configuration** | `config/settings.py` | Risk parameters, fuzzy bounds, model and GA hyperparameters |
| **Verification** | `tests/**` | 32 unit and integration tests |

---

## 11. Repository Sitemap

- **`config/settings.py`** — Global risk thresholds, fuzzy bounds, model and GA hyperparameters, 122-entry company map
- **`data/data_loader.py`** — yfinance downloader with Merton Jump Diffusion + Heston synthetic fallback
- **`data/feature_extractor.py`** — Feature vectors, normalisation, sequence datasets
- **`models/fuzzy_layer.py`** — PyTorch and NumPy TSK ANFIS layers, shared rule enumeration
- **`models/deep_neuro_fuzzy_net.py`** — BiLSTM + fuzzy fusion network with three prediction heads
- **`models/genetic_optimizer.py`** — Evolutionary optimiser with progress callback
- **`models/train.py`** — Supervised training engine, weight persistence, responsiveness probe
- **`utils/financial_metrics.py`** — Sharpe, MDD, IV, HV, VaR, Sortino maths
- **`utils/visualizer.py`** — Rich terminal reporting and matplotlib plots
- **`app.py`** — Streamlit entry point and sidebar navigation
- **`ui/services.py`** — Cached engine wrappers and data sources
- **`ui/components.py`** — Stat cards, risk gauge, pills, verdict block
- **`ui/charts.py`** — Plotly figure builders
- **`ui/styles.py`** — Blue-only theme and stylesheet
- **`ui/pages/`** — `dashboard`, `manual_input`, `training`, `optimizer`, `test_suite`, `reports`, `docs`
- **`.streamlit/config.toml`** — Theme and server defaults
- **`data_cache/risk_model_weights.pt`** — Committed trained weights
- **`tests/`** — 32 tests across `test_metrics`, `test_data`, `test_fuzzy`, `test_models`, `test_ga`, `test_pipeline`, `test_train`
- **`test_runner.py`** — Test discovery with Rich report and structured results
- **`main.py`** — Interactive CLI menu
- **`generate_pdf_report.py`** — PDF research report compiler

---

## 12. How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Web Interface (recommended)

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**. On first run the model trains automatically
(~2 minutes) and writes `data_cache/risk_model_weights.pt`; later runs load the
committed weights instantly. Use `--server.port 8502` or `--server.headless true`
as needed.

### 3. Train or Refit

Use the **Model Training** screen (choose epochs, watch live progress, export the
chromosome). To force a clean rebuild:

```bash
python -c "import pathlib; pathlib.Path('data_cache/risk_model_weights.pt').unlink(missing_ok=True)"
```

### 4. Interactive CLI Menu

```bash
python main.py
```

Eight options: run tests, analyse a company, score custom risk values, run the
genetic algorithm, view fuzzy curves, read documentation, generate the PDF
report, exit. The CLI shares engines and weights with the web app.

### 5. Automated Test Suite

```bash
python test_runner.py
```

All **32** tests; exits non-zero on failure for CI use.

### 6. PDF Research Report

```bash
python generate_pdf_report.py
```

---

## 13. Limitations and Future Work

### Current limitations

- **Absolute precision.** Validation MAE is ~6.4 score points because forward
  drawdown is inherently stochastic. Rankings are reliable; the exact number is
  not a forecast.
- **Small labelled universe.** Training uses 20 tickers with a synthetic
  fallback for any that are unreachable, which is sufficient to learn risk
  ranking but not a production trading signal.
- **No backtesting harness.** There is no walk-forward evaluation or
  transaction-cost model; the system scores instruments, it does not trade them.
- **Google Finance quote is scraped** from a public page. It is validated and
  degrades silently, but a markup change would drop the quote — never the
  analysis.
- **Research tool, not financial advice.**

### Future scope

- Walk-forward backtesting with transaction costs and position sizing
- News sentiment encoder (e.g. FinBERT) folded into the fuzzy antecedents
- Live options-chain implied volatility surfaces
- Multi-asset portfolio risk via NSGA-II
- Refitting on rolling windows with drift detection