# Stock Market Risk Prediction System

A quantitative research project that combines **Soft Computing** (Takagi-Sugeno
fuzzy inference + evolutionary optimisation) with **Deep Learning** (bidirectional
LSTMs) to score the market risk of any listed company, delivered through a
Streamlit web app and a terminal CLI.

---

## Live Demo

A public deployment is hosted on Streamlit Community Cloud:

### **[stock-market-risk-prediction.streamlit.app →](https://stock-market-risk-prediction-yu5vwfnvxrrcuv9qumrgnj.streamlit.app/)**

Nothing to install — the trained model weights ship with the repository, so the
risk score is fully responsive on first load.

---

## What It Does

- Scores any company on a **0–100 risk scale** with a Low / Moderate / High / Extreme band
- Explains itself: shows the **fuzzy membership curves**, the linguistic term assigned to every metric, and **which TSK rules fired** and by how much
- Accepts any ticker, a full company name, or **your own price series** from Google Sheets
- Shows a **live Google Finance quote** alongside the risk analysis
- Includes a **genetic algorithm** that evolves the fuzzy membership parameters
- Ships **32 automated tests** that run in the browser or on the command line

---

## Screens

| Screen | What it does |
|---|---|
| **Risk Dashboard** | Search a company, view its live Google quote, metrics, risk gauge, interactive price / drawdown / volatility charts and fuzzy-rule explanations. |
| **Manual Input** | Score your own Sharpe / drawdown / volatility values with sliders that re-predict live, plus Defensive → Crisis presets. |
| **Model Training** | Fit the risk head on real market history; watch training curves and the model's response across regimes. |
| **GA Optimizer** | Run the evolutionary search with a live progress bar; export the optimized chromosome. |
| **Test Suite** | Execute all 32 tests in-browser with structured pass/fail output and tracebacks. |
| **Reports** | Compile and download the PDF research report. |
| **Documentation** | This manual, rendered in-app. |

---

## Architecture

```
                    ┌──────────────────────────────┐
   OHLCV price  ───▶│  Feature Extraction           │──▶  [Sharpe, MaxDD, IV, HV]
   series            │  utils/financial_metrics.py  │              │
                    └──────────────────────────────┘              │
                                                                  ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                    Deep Neuro-Fuzzy Risk Model                     │
   │                                                                   │
   │   static vector ──▶ dense encoder (4→32→32) ─────────┐            │
   │                                                          │            │
   │   price history ──▶ BiLSTM (2×64, bidirectional) ─────┤            │
   │                                                          ▼            │
   │   4 metrics ──▶ Gaussian membership ──▶ 16 TSK rules ──▶ fusion net │
   │                  (Low / Medium / High)     (fuzzy score)              │
   │                                                      │                 │
   │                     ┌────────────────────────────────┼────┐            │
   │                     ▼                ▼               ▼    │            │
   │              Risk score 0–100   Category ×4     Max DD     │            │
   └───────────────────────────────────────────────────────────────────┘
                                     ▲
                                     │  trained on
                    ┌────────────────┴─────────────────┐
                    │  Supervised training engine      │
                    │  models/train.py                 │
                    └──────────────────────────────────┘
```

---

## The Model

### Supervised training

The risk head is **fitted on real market history**, not left at random
initialisation. `models/train.py` builds the dataset and trains all three heads.

| Aspect | Detail |
|---|---|
| Universe | 20 liquid tickers spanning defensive to speculative regimes |
| Dataset | ~24,000 supervised windows |
| Sequence | 20-day metric window per sample |
| Target horizon | 10 trading days forward |
| Score target | `clip(future_max_drawdown × 200 + HV × 100, 0, 100)` |
| Category target | Low (<25) / Moderate (<50) / High (<75) / Extreme (≥75) |
| Drawdown target | Forward-looking maximum peak-to-trough decline |
| Optimiser | Adam, lr 3e-3, cosine annealing, batch 64 |
| Persistence | `data_cache/risk_model_weights.pt` |

### Measured results

A fixed regime probe (identical inputs, before vs after training):

| Profile | Before | After | Band |
|---|---|---|---|
| Defensive | 46.80 | **15.95** | Low |
| Balanced | 46.76 | **23.37** | Low |
| Stressed | 46.73 | **36.86** | Moderate |
| Crisis | 46.83 | **80.82** | Extreme |
| **Score spread** | **0.10** | **64.87** | — |

Validation reaches roughly **73% category accuracy** with a **mean absolute error
of ~6.4 score points** — close to the noise floor, since forward drawdown is
genuinely unpredictable. The **ranking** of instruments is reliable; the absolute
number should be read as a score, not a precise forecast.

### Fuzzy layer

Four inputs, each fuzzified into three linguistic terms by a Gaussian
membership function `μ(x) = exp(-½((x − c)/σ)²)`, combined with the product
T-norm into **16 Takagi-Sugeno-Kang rules**, then defuzzified:

```
Fuzzy Risk Score = Σ(wᵢ · fᵢ(x)) / Σ(wᵢ)
```

This is what makes the output **interpretable** — every prediction can be traced
back to the rules that produced it.

### Genetic algorithm

A 28-gene chromosome (12 membership centres, 12 spreads, 4 feature weights)
evolved by tournament selection, simulated binary crossover (p = 0.8) and
Gaussian mutation (p = 0.15). Fitness is `1 / (1 + MSE)`.

---

## Risk Metrics

| Metric | Formula |
|---|---|
| Sharpe Ratio | `(Annualised Return − Rf) / Annualised Volatility`, Rf = 4% |
| Maximum Drawdown | `(Peak − Trough) / Peak` over the window |
| Implied Volatility | Garman-Klass `√(0.5·ln²(H/L) − (2ln2−1)·ln²(C/O))`, averaged over 21 days and annualised by √252 |
| Historical Volatility | `std(log returns) × √252` over 21 days |
| Sortino Ratio | `(Annualised Return − Rf) / Annualised Downside Volatility` |
| Value at Risk 95% | 5th percentile of daily returns, reported as a positive loss |

---

## Data Sources

| Source | Used for |
|---|---|
| **Yahoo Finance** | Full OHLCV price history. Falls back to a Merton Jump Diffusion simulator when offline, and the UI labels the data as synthetic. |
| **Google Finance** | Live price quotes, read from the public quote page and validated before display. |
| **Google Sheets** | Any custom price series — paste a link, the `export?format=csv` URL, or the sheet ID. |
| **Google Suggest** | Company-name autocomplete behind the search dropdown. |

Company names resolve automatically against a built-in map of **122 listings**
plus live Google suggestions, so typing `JPMorgan`, `Eli Lilly` or `brk.b` just
works.

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run the web app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**. On the very first run the model trains
automatically (~2 minutes) and saves its weights; later runs load them instantly.
Use `--server.port 8502` to change the port, or `--server.headless true` on a
remote machine.

### 3. Run the tests

```bash
python test_runner.py
```

### 4. Run the CLI

```bash
python main.py
```

Eight menu options: run tests, analyse a company, score custom risk values, run
the genetic algorithm, view fuzzy curves, read the docs, generate the PDF report,
or exit. The CLI shares the same engines and the same trained weights as the web
app.

### 5. Generate the PDF report

```bash
python generate_pdf_report.py
```

---

## Repository Layout

| Path | Responsibility |
|---|---|
| `app.py` | Streamlit entry point, sidebar navigation |
| `ui/services.py` | Cached wrappers over the data, model and optimizer engines |
| `ui/components.py` | Stat cards, risk gauge, pills, verdict block |
| `ui/charts.py` | Plotly figure builders |
| `ui/styles.py` | Blue-only theme and stylesheet |
| `ui/pages/` | One module per screen |
| `config/settings.py` | Risk parameters, fuzzy bounds, model and GA hyperparameters |
| `data/data_loader.py` | yfinance downloader + synthetic fallback generator |
| `data/feature_extractor.py` | Feature vectors and sequence datasets |
| `models/fuzzy_layer.py` | TSK ANFIS layer (PyTorch + NumPy variants) |
| `models/deep_neuro_fuzzy_net.py` | BiLSTM + fuzzy fusion network, three prediction heads |
| `models/genetic_optimizer.py` | Evolutionary optimiser |
| `models/train.py` | Supervised training engine and weight persistence |
| `utils/financial_metrics.py` | Sharpe, MDD, IV, HV, VaR, Sortino |
| `utils/visualizer.py` | Rich terminal output and matplotlib plots |
| `main.py` | CLI menu |
| `test_runner.py` | Test discovery and reporting |
| `generate_pdf_report.py` | PDF research report compiler |
| `tests/` | 32 unit and integration tests |

---

## Testing

```
Total Tests Run   32      Passed 32      Failures 0      Errors 0
```

Covers metric maths, synthetic data generation, fuzzy forward passes, the
fusion network, GA evolution, the end-to-end pipeline, and regression guards on
the training engine — including an explicit test that the trained score
responds to input, so the original flat-score bug cannot silently return.

`test_runner.py` exits non-zero on failure, so it drops straight into CI.

---

## Notes and Limitations

- **Research tool, not financial advice.**
- Absolute scores carry roughly ±6 points of error; treat the **band** (Low →
  Extreme) as the decision-relevant output.
- The Google Finance quote is scraped from a public page. It is validated before
  display and degrades silently if Google's markup ever changes — the risk
  analysis never depends on it.
- Without PyTorch the pure-NumPy engine still runs, but training is disabled.
- Requires Python 3.10+ (developed on 3.13).