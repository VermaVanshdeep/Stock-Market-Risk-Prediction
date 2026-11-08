# Public Cloud Deployment Guide

**Application:** Stock Market Risk Prediction System  
**Framework:** Streamlit (Python)  
**Primary Target:** Streamlit Community Cloud  
**Repository:** [VermaVanshdeep/Stock-Market-Risk-Prediction](https://github.com/VermaVanshdeep/Stock-Market-Risk-Prediction)  
**Branch:** `main`  
**Entry Point:** `app.py`  

---

## 1. Hosting Architecture Overview

The Stock Market Risk Prediction System is engineered as a zero-configuration, container-ready Streamlit web application. All computational modules—the Deep Neuro-Fuzzy network, BiLSTM sequential pipeline, Genetic Algorithm (GA) parameter tuner, and Genetic Programming (GP) symbolic risk generator—run in-process without requiring external database instances or secondary microservices.

```
+-------------------------------------------------------------------------+
|                       Streamlit Community Cloud                         |
|                                                                         |
|  +--------------------+    +--------------------+    +---------------+  |
|  |   Streamlit Web    |    |   Cached Model     |    | In-Memory     |  |
|  |     Frontend       |<-->|    Singletons      |<-->| Data Cache    |  |
|  |     (app.py)       |    | (@st.cache_resource|    | (@st.cache    |  |
|  +--------------------+    +--------------------+    |     _data)    |  |
|            |                                         +---------------+  |
|            v                                                 |          |
|  +--------------------+                                      v          |
|  | Live Yahoo Finance |                            +-----------------+  |
|  | Market Data Feed   |                            | Fallback Engine |  |
|  | (NSE & US Tickers) |                            |  (Merton Jump   |  |
|  +--------------------+                            |   Diffusion)    |  |
|                                                    +-----------------+  |
+-------------------------------------------------------------------------+
```

---

## 2. Step-by-Step Deployment Instructions

Follow these exact steps to deploy the application publicly from your GitHub account:

### Step 1: Access Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io).
2. Sign in using your GitHub account (**VermaVanshdeep**).

### Step 2: Create New App
1. Click the **"New app"** (or **"Create app"**) button located in the top-right corner.
2. Under **"Repository"**, select or enter:
   ```text
   VermaVanshdeep/Stock-Market-Risk-Prediction
   ```
3. Under **"Branch"**, select:
   ```text
   main
   ```
4. Under **"Main file path"**, enter:
   ```text
   app.py
   ```
5. *(Optional Custom Subdomain)* Under **"App URL"**, enter your desired subdomain:
   ```text
   stock-market-risk-prediction.streamlit.app
   ```
   *(If already claimed, Streamlit Cloud will append a short unique hash).*

### Step 3: Advanced Settings Verification
1. Click **"Advanced settings"**.
2. Set the Python version to **Python 3.11** (matching the verified CI test environment).
3. Secrets are **not required**: all market data is fetched via public Yahoo Finance APIs, and offline fallbacks are fully self-contained.

### Step 4: Deploy
1. Click **"Deploy!"**.
2. Streamlit Cloud will provision a Debian-based container, install dependencies from `requirements.txt`, load pre-trained weights from `data_cache/`, and launch the web server.
3. The build log typically finishes within 90–120 seconds.

---

## 3. Platform Lifecycle: Sleep & Wake-Up Behaviour

Streamlit Community Cloud provides free compute containers with strict operational lifecycles. Understanding these guarantees reliable presentation to faculty and public visitors:

### Inactivity Sleep Policy
* **Inactivity Window:** If an application receives zero HTTP visits for **7 consecutive days**, Streamlit Community Cloud suspends the container to preserve compute capacity.
* **Sleeping State:** The app is not deleted; its snapshot and repository link remain permanently stored.
* **Wake-Up Experience:** When a user visits the URL of a sleeping app, Streamlit displays a clean wake-up screen:
  > *"This app has gone to sleep due to inactivity. Wake it up to continue."*
* **Wake-Up Action:** The user or presenter clicks the blue **"Wake up app"** button. The container cold-boots and becomes fully interactive in approximately **30 to 60 seconds**.

### Startup & Caching Optimizations Implemented
To ensure the cold-boot and wake-up phase is as fast and seamless as possible, the codebase incorporates the following architectural optimizations:

1. **Pre-Trained Weights Checked Into Repository (`data_cache/`):**
   * Pre-trained PyTorch weights (`risk_model_weights.pt` — 600 KB), optimized GA hyperparameters (`ga_optimized_params.json` — 2.2 KB), and GP symbolic expression trees (`gp_expression_tree.json` — 209 B) are bundled directly in Git.
   * `services.ensure_trained()` checks `trainer.weights_exist()` and loads weights into RAM in **< 100 milliseconds**, completely bypassing the need for multi-epoch training on container startup.

2. **Persistent Singleton Caching (`@st.cache_resource`):**
   * The `DeepNeuroFuzzyRiskModel`, `StockDataLoader`, and `RiskFeatureExtractor` are wrapped in `@st.cache_resource(show_spinner=False)`.
   * Models are instantiated once per container lifecycle and shared across all incoming user sessions with zero re-instantiation overhead.

3. **Market Data TTL Caching (`@st.cache_data(ttl=1800)`):**
   * Historical OHLCV market data downloaded from Yahoo Finance is cached in memory for **30 minutes** (1800 seconds).
   * Rapid toggling between stocks or re-analyzing tickers avoids hitting Yahoo Finance rate limits and delivers instant response times (< 50ms) for previously queried symbols.

4. **Resilient Offline Synthetic Fallback (Merton Jump Diffusion):**
   * In free cloud environments, Yahoo Finance can occasionally rate-limit shared egress IPs.
   * `StockDataLoader(use_offline_fallback=True)` automatically catches network timeouts or HTTP 429 errors and generates high-fidelity synthetic market dynamics using Merton Jump Diffusion, displaying an amber **"Synthetic Fallback"** status pill instead of crashing.

5. **Resource Guardrails:**
   * **RAM Allocation:** Streamlit Cloud provides up to 1 GB RAM (burstable to ~2.7 GB). The complete runtime consumes ~180 MB of RAM, leaving plenty of headroom.
   * **CPU Allocation:** Single shared vCPU performs CPU PyTorch inference in ~30–45ms per stock.

### Policy on Keep-Alive Bots and Artificial Traffic
* **No Artificial Traffic Generation:** Streamlit Community Cloud terms of service prohibit automated pinging or synthetic keep-alive bots intended to bypass sleeping policies.
* **Best Practice:** Allow the native sleep behavior to operate naturally. For live presentations or faculty reviews, simply visit the public URL 2–3 minutes beforehand to ensure the container is warm and primed.

---

## 4. Post-Deployment Verification Checklist

Once the public application is running at `https://<your-app-subdomain>.streamlit.app`, verify the core functionalities using this checklist:

| Feature | Test Action | Expected Result |
| :--- | :--- | :--- |
| **Container Health** | Open public URL in browser | Dashboard loads with blue theme; sidebar displays "Engine status: PyTorch, yfinance live, Streamlit". |
| **Model Pre-Load** | Check "Model status" expander in sidebar | Status shows "Trained on 10 universe stocks" immediately without running manual training. |
| **Indian NSE Stocks** | Click quick-select pill `RELIANCE.NS` or `TCS.NS` | Price history loads, quote reflects INR (`₹`), currency pill indicates `INR (₹)`, and Risk Gauge renders. |
| **US Market Stocks** | Click quick-select pill `AAPL` or `NVDA` | Quote reflects USD (`$`), metrics calculate correctly. |
| **Interactive Charts** | Switch between "Price & drawdown", "Volatility", and "Risk breakdown" tabs | Plotly interactive candlestick and drawdown curves zoom and pan smoothly. |
| **Fuzzy Rule Activation** | Scroll to "Fuzzy Rule Explanations" | Fired TSK rules display with antecedent membership pills (e.g. `Low`, `Medium`, `High`) and weights. |
| **GA Optimizer Page** | Navigate to "GA Optimizer" in sidebar | Pre-computed optimal parameters load; runs optimization without container memory limits. |
| **Executive Reports** | Navigate to "Reports" and click "Generate PDF Report" | PDF report downloads cleanly via browser with complete statistical tables. |

---

## 5. Local vs Cloud Execution Parity

The application maintains 100% functional parity between local workstations and public cloud deployment:

* **Local Command:**
  ```bash
  streamlit run app.py
  ```
* **Cloud Command (Automated by Streamlit Cloud):**
  ```bash
  streamlit run app.py --server.port 8501 --server.address 0.0.0.0
  ```
* Both environments run identical neural-fuzzy weights, indicator calculations, currency handlers, and automated test suites.

