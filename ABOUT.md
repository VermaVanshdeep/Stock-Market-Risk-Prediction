# About the Project & Technical Implementation Manual

## Executive Overview

This project is a quantitative **Stock Market Risk Prediction System** that bridges **Soft Computing** methodology with **Deep Learning** neural architectures. Traditional quantitative finance relies on crisp mathematical assumptions that often fail during market panics, regime shifts, or high-volatility events. By combining **Takagi-Sugeno Adaptive Neuro-Fuzzy Inference Systems (ANFIS)**, **Genetic Algorithm (GA) Evolutionary Optimization**, and **Deep Bidirectional LSTMs**, this system captures linguistic uncertainty, non-linear market dynamics, and tail-risk drawdowns.

---

## Soft Computing Concepts Used

### 1. Fuzzy Logic & ANFIS (Takagi-Sugeno-Kang Model)
- **Fuzzification**: Transforms crisp quantitative stock metrics into fuzzy linguistic degrees of membership between 0.0 and 1.0.
- **Membership Functions**: Implements differentiable Gaussian Membership Functions in PyTorch and NumPy:
  
  mu(x) = exp( -0.5 * ((x - Center) / Spread)^2 )
  
  where Center is the mean and Spread is the standard deviation of the fuzzy set (Low, Medium, High).

- **Fuzzy Rule Evaluation**: Evaluates non-linear rule bases using the product T-norm operator:
  
  Rule Firing Weight = Sharpe_Fuzzy * Drawdown_Fuzzy * IV_Fuzzy * HV_Fuzzy

- **TSK Defuzzification**: Computes a crisp output score via weighted average over linear rule consequents:
  
  Fuzzy Risk Score = Sum(Rule Weight * Rule Output) / Sum(Rule Weight)
  
  where Rule Output is a linear equation: p * Sharpe + q * Drawdown + r * IV + s * HV + Constant

### 2. Genetic Algorithm (GA) Evolutionary Optimization
- **Chromosome Representation**: Encodes fuzzy membership parameters (centers, spreads) and feature importance weights as a float vector chromosome of length 28.
- **Fitness Function**: Evaluates risk separation and prediction accuracy:
  
  Fitness = 1.0 / (1.0 + Mean Squared Error)

- **Evolutionary Operators**:
  - *Selection*: Tournament Selection operator comparing random chromosome pairs.
  - *Crossover*: Simulated Binary Crossover (SBX) with probability 0.8.
  - *Mutation*: Gaussian jitter mutation with probability 0.15.

---

## Deep Learning Architecture

### Hybrid Deep Neuro-Fuzzy Risk Network (DeepNeuroFuzzyRiskModel)
1. **Soft Computing Branch**: Evaluates ANFIS TSK fuzzy rule firings to generate fuzzy risk embeddings.
2. **Deep Sequence Representation Branch**: A 2-layer Bidirectional LSTM (BiLSTM) processes historical return sequences and rolling volatility windows to capture temporal dependencies and regime shifts.
3. **Dense Vector Encoder**: Encodes static metric vectors [Sharpe, MaxDrawdown, ImpliedVol, HistVol].
4. **Information Fusion Network**: Concatenates deep sequence representations, fuzzy risk scores, and dense embeddings.
5. **Multi-Task Prediction Heads**:
   - **Continuous Risk Score (0 to 100)**: Sigmoid output scaled to 0-100 range.
   - **Risk Classification**: Multi-class logits for 4 risk levels (Low, Moderate, High, Extreme).
   - **Drawdown Forecast**: Expected next-horizon maximum drawdown percentage.

---

## Financial Risk Metrics & Formulas

### 1. Sharpe Ratio
Measures risk-adjusted excess return per unit of total risk:

Sharpe Ratio = (Annual Return - Risk Free Rate) / Annual Volatility

where Risk Free Rate is 4% (0.04) and Annual Volatility is standard deviation of daily returns multiplied by sqrt(252).

### 2. Maximum Drawdown (MDD)
Measures peak-to-trough decline over a specified trading window:

Max Drawdown (MDD) = (Peak Price - Current Price) / Peak Price

### 3. Implied Volatility (IV) Proxy (Garman-Klass Estimator)
Uses High, Low, Open, Close price dynamics to estimate option-implied volatility:

Implied Volatility Proxy = Average( 0.5 * (ln(High/Low))^2 - 0.386 * (ln(Close/Open))^2 ) * sqrt(252)

### 4. Historical Volatility (HV)
Annualized standard deviation of daily logarithmic returns:

Historical Volatility (HV) = Standard Deviation of Daily Returns * sqrt(252)

### 5. Value at Risk (VaR 95%) & Sortino Ratio
- **VaR 95%**: 95th percentile expected 1-day maximum loss threshold.
- **Sortino Ratio**: Adjusted return using downside deviation (calculated only on negative returns):

  Sortino Ratio = (Annual Return - Risk Free Rate) / Downside Volatility

---

## Repository Sitemap & File Structure

- **`config/settings.py`**: Global risk thresholds, fuzzy bounds, model hyperparameters.
- **`data/data_loader.py`**: Downloads live stock data via `yfinance` with fallback to a high-fidelity synthetic Merton Jump Diffusion generator (`SyntheticStockGenerator`).
- **`data/feature_extractor.py`**: Converts stock series into normalized feature tensors.
- **`models/fuzzy_layer.py`**: PyTorch and NumPy TSK Adaptive Neuro-Fuzzy Layer.
- **`models/deep_neuro_fuzzy_net.py`**: Hybrid Deep Neuro-Fuzzy Neural Network.
- **`models/genetic_optimizer.py`**: Genetic Algorithm evolutionary optimizer.
- **`utils/financial_metrics.py`**: Math engine for Sharpe, MDD, IV, HV, VaR, Sortino.
- **`utils/visualizer.py`**: Rich terminal UI reporting & Matplotlib graphical plots.
- **`tests/`**: Unit & integration test suite (`test_metrics.py`, `test_data.py`, `test_fuzzy.py`, `test_models.py`, `test_ga.py`, `test_pipeline.py`).
- **`test_runner.py`**: Automated test launcher with visual pass/fail diagnostics.
- **`main.py`**: Interactive CLI Menu System.
- **`generate_pdf_report.py`**: Compiles PDF report (`Stock_Market_Risk_Prediction_Report.pdf`).

---

## How to Run

1. **Automated Test Suite Execution**:
   ```bash
   python test_runner.py
   ```
2. **Interactive CLI Menu**:
   ```bash
   python main.py
   ```
