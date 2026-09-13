"""
Soft Computing Core: Takagi-Sugeno-Kang (TSK) Adaptive Neuro-Fuzzy Layer.

Implements differentiable Gaussian Membership Functions, Fuzzy Fuzzification,
Rule Firing Evaluation, and Defuzzification for stock market risk assessment.
Supports both PyTorch and high-performance pure NumPy fallback for seamless cross-platform execution.
"""

import sys
import numpy as np
from typing import Dict, Tuple
from config.settings import FUZZY_INPUT_SPECS, RISK_CATEGORIES

# Check PyTorch availability
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    # Verify C++ backend
    _ = torch.tensor([1.0])
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class PyTorchTakagiSugenoFuzzyLayer(nn.Module):
        """
        Adaptive Neuro-Fuzzy Inference System (ANFIS) TSK PyTorch Layer.
        """

        def __init__(self):
            super(PyTorchTakagiSugenoFuzzyLayer, self).__init__()
            
            self.num_inputs = 4
            self.num_mfs = 3

            means_init = torch.tensor([
                [0.0, 1.0, 2.0],     # Sharpe Ratio
                [0.05, 0.20, 0.40],  # Max Drawdown
                [0.12, 0.25, 0.45],  # Implied Volatility
                [0.12, 0.25, 0.45]   # Historical Volatility
            ], dtype=torch.float32)

            sigmas_init = torch.tensor([
                [0.5, 0.5, 0.5],
                [0.05, 0.08, 0.10],
                [0.04, 0.06, 0.10],
                [0.04, 0.06, 0.10]
            ], dtype=torch.float32)

            self.means = nn.Parameter(means_init)
            self.sigmas = nn.Parameter(sigmas_init)

            self.num_rules = 16
            self.consequent_weights = nn.Parameter(torch.randn(self.num_rules, self.num_inputs + 1) * 0.1)

        def compute_membership(self, x: torch.Tensor) -> torch.Tensor:
            x_exp = x.unsqueeze(-1)
            means_exp = self.means.unsqueeze(0)
            sigmas_exp = torch.clamp(self.sigmas.unsqueeze(0), min=1e-4)

            diff = (x_exp - means_exp) / sigmas_exp
            mu = torch.exp(-0.5 * (diff ** 2))
            return mu

        def evaluate_fuzzification_labels(self, input_vector: np.ndarray) -> Dict[str, str]:
            terms = ['Low', 'Medium', 'High']
            feature_names = ['sharpe', 'drawdown', 'implied_vol', 'hist_vol']
            x_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                mu = self.compute_membership(x_tensor).squeeze(0)
                
            result = {}
            for idx, fname in enumerate(feature_names):
                max_idx = torch.argmax(mu[idx]).item()
                result[fname] = terms[max_idx]
                
            return result

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            batch_size = x.size(0)
            mu = self.compute_membership(x)

            rule_list = []
            for s_idx in [0, 1, 2]:
                for d_idx in [0, 2]:
                    for iv_idx in [0, 2]:
                        for hv_idx in [0, 1]:
                            if len(rule_list) < self.num_rules:
                                rule_list.append((s_idx, d_idx, iv_idx, hv_idx))
            
            firing_strengths = []
            for r_idx, (s_i, d_i, iv_i, hv_i) in enumerate(rule_list):
                w = mu[:, 0, s_i] * mu[:, 1, d_i] * mu[:, 2, iv_i] * mu[:, 3, hv_i]
                firing_strengths.append(w)

            w_tensor = torch.stack(firing_strengths, dim=1)
            w_sum = torch.sum(w_tensor, dim=1, keepdim=True) + 1e-8
            w_norm = w_tensor / w_sum

            x_aug = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)
            consequents = torch.matmul(x_aug, self.consequent_weights.T)

            fuzzy_output = torch.sum(w_norm * consequents, dim=1, keepdim=True)
            return fuzzy_output, w_norm

else:
    # NumPy Fallback Engine for Takagi-Sugeno Fuzzy Layer
    class PyTorchTakagiSugenoFuzzyLayer:
        """
        Pure NumPy Soft Computing ANFIS TSK Fuzzy Layer.
        """

        def __init__(self):
            self.num_inputs = 4
            self.num_mfs = 3

            self.means = np.array([
                [0.0, 1.0, 2.0],     # Sharpe Ratio
                [0.05, 0.20, 0.40],  # Max Drawdown
                [0.12, 0.25, 0.45],  # Implied Volatility
                [0.12, 0.25, 0.45]   # Historical Volatility
            ], dtype=np.float32)

            self.sigmas = np.array([
                [0.5, 0.5, 0.5],
                [0.05, 0.08, 0.10],
                [0.04, 0.06, 0.10],
                [0.04, 0.06, 0.10]
            ], dtype=np.float32)

            self.num_rules = 16
            np.random.seed(42)
            self.consequent_weights = np.random.randn(self.num_rules, self.num_inputs + 1) * 0.1

        def compute_membership(self, x: np.ndarray) -> np.ndarray:
            if x.ndim == 1:
                x = x[np.newaxis, :]
            batch_size = x.shape[0]
            x_exp = x[:, :, np.newaxis]
            means_exp = self.means[np.newaxis, :, :]
            sigmas_exp = np.clip(self.sigmas[np.newaxis, :, :], 1e-4, None)

            diff = (x_exp - means_exp) / sigmas_exp
            mu = np.exp(-0.5 * (diff ** 2))
            return mu

        def evaluate_fuzzification_labels(self, input_vector: np.ndarray) -> Dict[str, str]:
            terms = ['Low', 'Medium', 'High']
            feature_names = ['sharpe', 'drawdown', 'implied_vol', 'hist_vol']
            mu = self.compute_membership(input_vector)[0]
            
            result = {}
            for idx, fname in enumerate(feature_names):
                max_idx = int(np.argmax(mu[idx]))
                result[fname] = terms[max_idx]
                
            return result

        def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if x.ndim == 1:
                x = x[np.newaxis, :]
            batch_size = x.shape[0]
            mu = self.compute_membership(x)

            rule_list = []
            for s_idx in [0, 1, 2]:
                for d_idx in [0, 2]:
                    for iv_idx in [0, 2]:
                        for hv_idx in [0, 1]:
                            if len(rule_list) < self.num_rules:
                                rule_list.append((s_idx, d_idx, iv_idx, hv_idx))

            firing_strengths = []
            for r_idx, (s_i, d_i, iv_i, hv_i) in enumerate(rule_list):
                w = mu[:, 0, s_i] * mu[:, 1, d_i] * mu[:, 2, iv_i] * mu[:, 3, hv_i]
                firing_strengths.append(w)

            w_tensor = np.column_stack(firing_strengths)
            w_sum = np.sum(w_tensor, axis=1, keepdims=True) + 1e-8
            w_norm = w_tensor / w_sum

            x_aug = np.column_stack([x, np.ones((batch_size, 1))])
            consequents = np.dot(x_aug, self.consequent_weights.T)

            fuzzy_output = np.sum(w_norm * consequents, axis=1, keepdims=True)
            return fuzzy_output, w_norm

        def __call__(self, x):
            return self.forward(x)
