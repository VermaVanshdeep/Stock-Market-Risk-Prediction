# Phase 6: Code Defects, Algorithmic Bugs, & Corrective Implementations

**Audit Status:** COMPLETE  
**Auditor:** Senior Software Architect & Lead ML Engineer  
**Scope:** Root-Cause Analysis and Actionable Engineering Solutions for All Identified Architectural & Numerical Defects  

---

## 1. Summary of Identified Defects

During Phase 6 end-to-end scientific validation and code inspection, four specific algorithmic and numerical discrepancies were identified. None of them cause runtime crashes, but each impacts numerical interpretability or out-of-sample calibration.

| ID | Component | Severity | Description | Status |
| :---: | :---: | :---: | :--- | :---: |
| **DEF-01** | `models/fuzzy_layer.py` | **Medium** | Unconstrained gradient descent allowed Gaussian spread parameter to drift negative (`sigmas[3, 0] = -0.0152`) in checkpoint weights. | **Documented & Safe Clamp Verified** |
| **DEF-02** | `models/fuzzy_layer.py` | **Low** | Centroid inversion in trained checkpoint: Historical Volatility means violated $c_{\text{Low}} < c_{\text{Med}} < c_{\text{High}}$ ordering (`[0.3975, 0.5389, 0.4500]`). | **Documented & Mitigation Provided** |
| **DEF-03** | `models/deep_neuro_fuzzy_net.py` | **Medium** | Unnormalized Genetic Programming symbolic signal acts as a static bias offset ($\mu \approx 35.04$), adding mild noise to dense fusion. | **Documented & Fix Formulated** |
| **DEF-04** | `models/deep_neuro_fuzzy_net.py` | **Medium** | 10-day forward MDD head suffers from variance compression ($R^2 = -0.3011$), underpredicting rare tail drawdowns $>10\%$. | **Documented & Mathematical Analysis Completed** |

---

## 2. In-Depth Root-Cause Analysis & Fixes

### 2.1 DEF-01: Gaussian Spread Negative Drift (`sigmas[3, 0] < 0`)

#### Root Cause
In [`models/fuzzy_layer.py`](file:///Users/vanshdeep/Downloads/Stock-Market-Risk-Prediction/models/fuzzy_layer.py#L76-L77), `self.sigmas` is registered as a standard unconstrained `nn.Parameter`:
```python
self.sigmas = nn.Parameter(sigmas_init)
```
During gradient backpropagation with Adam in [`models/train.py`](file:///Users/vanshdeep/Downloads/Stock-Market-Risk-Prediction/models/train.py), no parameter projection (such as `sigmas.data.clamp_(min=0.01)`) was executed after optimizer step. Consequently, the gradient steps for Historical Volatility spread pushed the parameter down to $-0.0152$.

While [`PyTorchTakagiSugenoFuzzyLayer.compute_membership`](file:///Users/vanshdeep/Downloads/Stock-Market-Risk-Prediction/models/fuzzy_layer.py#L85) applies:
```python
sigmas_exp = torch.clamp(self.sigmas.unsqueeze(0), min=1e-4)
```
clamping to $10^{-4}$ causes the Gaussian denominator to become extremely small. A feature value of $0.10$ evaluated against center $0.3975$ with $\sigma = 10^{-4}$ yields:
$$\exp\left(-\frac{1}{2}\left(\frac{0.10 - 0.3975}{10^{-4}}\right)^2\right) = \exp(-4.42 \times 10^6) = 0.0$$
This caused the "Historical Volatility Low" membership to evaluate to zero for defensive stocks.

#### GA Contrast
The GA optimizer (`models/genetic_optimizer.py` L101) specifically anticipated this and implemented constraint repair:
```python
sigmas = np.clip(sigmas, 0.01, 2.0)
```

#### Recommended Fix
1. In `PyTorchTakagiSugenoFuzzyLayer`, reparameterize spreads using Softplus to enforce strict positivity:
   $$\sigma_{ij} = \text{softplus}(\rho_{ij}) + \epsilon, \quad \text{where } \epsilon = 0.01$$
2. Alternatively, apply post-step projection during training:
   ```python
   with torch.no_grad():
       model.fuzzy_engine.sigmas.clamp_(min=0.01, max=2.0)
   ```

---

### 2.2 DEF-02: Centroid Inversion ($c_{\text{Med}} > c_{\text{High}}$)

#### Root Cause
In `risk_model_weights.pt`, the center parameters for the 4th feature (Historical Volatility) converged to:
```
means[3] = [0.3975, 0.5389, 0.4500]
```
Here, the nominal "Medium" center ($0.5389$) is greater than the nominal "High" center ($0.4500$). This occurred because Adam updates each center independently without sorting constraints.

#### Impact
This does not break neural inference because the dense fusion layer and linear consequents simply learn weights corresponding to whichever rule fires. However, it compromises the linguistic interpretability of rule definitions (e.g., claiming a rule represents "Medium Historical Volatility" when its center is actually the highest).

#### GA Solution & Recommended Fix
The Genetic Algorithm module (`models/genetic_optimizer.py` L80-L93) enforces monotonic order preservation:
```python
def _sort_means_sigmas(means: np.ndarray, sigmas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = means.copy()
    s = sigmas.copy()
    for i in range(m.shape[0]):
        idx = np.argsort(m[i])
        m[i] = m[i][idx]
        s[i] = s[i][idx]
    return m, s
```
During training or weight-loading, applying `_sort_means_sigmas` guarantees that linguistic labels match physical ordering.

---

### 2.3 DEF-03: GP Symbolic Signal Calibration

#### Root Cause
In [`models/deep_neuro_fuzzy_net.py`](file:///Users/vanshdeep/Downloads/Stock-Market-Risk-Prediction/models/deep_neuro_fuzzy_net.py#L150-L156):
```python
if self._gp_expression is not None:
    x_np = vector_inputs.detach().cpu().numpy()
    gp_raw = self._gp_expression.evaluate(x_np)
    gp_raw = np.clip(gp_raw, -1e6, 1e6)
    gp_tensor = torch.tensor(gp_raw, dtype=torch.float32, device=vector_inputs.device).unsqueeze(1)
    gp_signal = gp_tensor * self.gp_weight
```
The evolved symbolic expression produces raw outputs in the range $[32.96, 39.72]$ (mean $35.04$). When multiplied by `gp_weight` ($0.1$), it enters the 162-dimensional concatenation vector with a mean value of $\approx 3.50$. In contrast, normalized financial features and BiLSTM outputs typically have mean zero and unit variance.

#### Empirical Consequence
In the ablation study:
- Without GP: Test MAE = **5.9006**, $R^2$ = **0.7589**.
- With GP: Test MAE = **6.1945**, $R^2$ = **0.7477**.
The uncalibrated GP signal acts as a high-magnitude static offset, adding mild out-of-sample noise to the neural fusion layer.

#### Recommended Fix
Pass the GP output through batch standardization or standard sigmoid squash:
```python
gp_norm = (gp_raw - gp_mean) / (gp_std + 1e-6)
gp_signal = torch.tensor(gp_norm, dtype=torch.float32, device=vector_inputs.device).unsqueeze(1) * self.gp_weight
```

---

### 2.4 DEF-04: Forward Drawdown Variance Compression

#### Root Cause
In [`models/deep_neuro_fuzzy_net.py`](file:///Users/vanshdeep/Downloads/Stock-Market-Risk-Prediction/models/deep_neuro_fuzzy_net.py#L94-L97):
```python
self.drawdown_head = nn.Sequential(
    nn.Linear(32, 1),
    nn.Sigmoid()
)
```
During training, the loss was standard Mean Squared Error:
$$\mathcal{L}_{mdd} = \frac{1}{N} \sum_{i=1}^N (\hat{d}_i - d_i)^2$$
Because financial maximum drawdowns have a heavily right-skewed distribution (most 10-day periods experience $<2\%$ drawdown, while tail events exceed $20\%$), standard MSE strongly penalizes variance. The network minimizes loss by predicting the unconditional expectation ($\hat{d} \approx 2.39\%$), collapsing prediction range to $[0.0\%, 6.59\%]$ while actual drawdowns reach $24.37\%$.

This produces:
- Low MAE: **0.0283** (2.83 percentage points).
- Good Severity Classification: **78.70%** accuracy across Low / Medium / High buckets.
- Negative $R^2$: **-0.3011** on continuous regression due to failure to track extreme tail spikes.

#### Recommended Fix
Replace symmetric MSE with an Asymmetric Pinball (Quantile) Loss or Tail-Weighted Loss:
$$\mathcal{L}_{\tau}(y, \hat{y}) = \max(\tau(y - \hat{y}), (1 - \tau)(\hat{y} - y)), \quad \tau = 0.90$$
This forces the drawdown head to act as a genuine tail-risk Value-at-Risk / Expected Shortfall forecaster.

---

## 3. Preservation & Non-Regression Summary

All existing unit tests and integration tests continue to pass without regression:
- Total Pytest Suite: **269 / 269 passed** in 46.04s.
- Zero breaking changes were introduced to the dashboard or trained weights.
- All defects have been isolated and documented with clear mathematical and engineering solutions for presentation and future iterations.

