"""
Deep Learning Core: Deep Neuro-Fuzzy Risk Prediction Network.
Fuses Deep Bidirectional LSTM sequence modeling with Soft Computing Takagi-Sugeno Fuzzy Logic.
Supports both PyTorch and pure NumPy fallback for seamless cross-platform execution.

GP Integration (Phase 3)
─────────────────────────
An optional Genetic Programming expression (``Node`` from
``models.genetic_programming``) can be attached to the model via
``model.set_gp_expression(node)``.  Its output is a single scalar per sample
(evaluated in NumPy, then converted to a torch Tensor) that is concatenated into
the fusion-network input as a 162nd feature (after the existing 161 = 128 BiLSTM
+ 1 fuzzy + 32 dense).

When no expression is attached the GP channel is filled with zeros and the model
behaves identically to the original 161-dim version.  A learnable ``gp_weight``
scalar scales the raw GP signal before concatenation; it starts at 0.1 so the
GP channel is initially a small perturbation rather than a dominant signal.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from models.fuzzy_layer import PyTorchTakagiSugenoFuzzyLayer, TORCH_AVAILABLE
from config.settings import MODEL_CONFIG, RISK_CATEGORIES

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DeepNeuroFuzzyRiskModel(nn.Module):
        """
        Hybrid Deep Neuro-Fuzzy Stock Market Risk Network (PyTorch implementation).

        GP Integration
        ──────────────
        Attach an evolved GP expression with ``model.set_gp_expression(node)``.
        The expression is evaluated on the raw feature vector (NumPy), scaled by
        the learnable ``gp_weight`` parameter, and concatenated as a 162nd element
        into the fusion-network input.  When no expression is set the channel is
        zeros and the model is numerically identical to the original 161-dim version.
        """

        def __init__(
            self,
            input_dim: int = MODEL_CONFIG['input_dim'],
            hidden_dim: int = MODEL_CONFIG['hidden_dim'],
            num_categories: int = 4
        ):
            super(DeepNeuroFuzzyRiskModel, self).__init__()

            self.fuzzy_engine = PyTorchTakagiSugenoFuzzyLayer()

            self.bilstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=MODEL_CONFIG['bilstm_layers'],
                batch_first=True,
                bidirectional=True,
                dropout=MODEL_CONFIG['dropout']
            )

            self.dense_vector_net = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU()
            )

            # 128 (BiLSTM) + 1 (fuzzy) + 32 (dense) + 1 (GP signal) = 162
            fusion_dim = hidden_dim * 2 + 1 + 32 + 1

            self.fusion_net = nn.Sequential(
                nn.Linear(fusion_dim, 64),
                nn.ReLU(),
                nn.Dropout(MODEL_CONFIG['dropout']),
                nn.Linear(64, 32),
                nn.ReLU()
            )

            self.risk_score_head = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

            self.classification_head = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, num_categories)
            )

            self.drawdown_head = nn.Sequential(
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

            # Learnable scale for the GP signal (initialised small so the GP
            # channel starts as a gentle perturbation, not a dominant signal).
            self.gp_weight = nn.Parameter(torch.tensor(0.1))

            # GP expression (Node from models.genetic_programming), not a
            # nn.Module — stored as a plain Python attribute so it is not
            # included in state_dict() and does not interfere with save/load.
            self._gp_expression = None   # type: Optional[object]
            self._gp_mean = 0.0
            self._gp_std = 1.0

            if hasattr(self.fuzzy_engine, 'clamp_and_sort_parameters'):
                self.fuzzy_engine.clamp_and_sort_parameters()

        # ------------------------------------------------------------------ #
        # GP expression management
        # ------------------------------------------------------------------ #

        def set_gp_expression(self, node) -> None:
            """Attach an evolved GP expression (Node or None) to the model and calibrate reference scaling."""
            self._gp_expression = node
            if node is None:
                self._gp_mean = 0.0
                self._gp_std = 1.0
                return

            # Calibrate reference normalization statistics over representative feature domain
            # Feature order: [sharpe, drawdown, implied_vol, hist_vol]
            # Bounds: sharpe [-2, 4], drawdown [0, 0.8], iv [0.05, 1.2], hv [0.05, 1.2]
            rng = np.random.default_rng(42)
            probe_grid = np.column_stack([
                rng.uniform(-2.0, 4.0, 500),
                rng.uniform(0.0, 0.8, 500),
                rng.uniform(0.05, 1.20, 500),
                rng.uniform(0.05, 1.20, 500),
            ]).astype(np.float32)

            try:
                raw_vals = np.clip(node.evaluate(probe_grid), -1e6, 1e6)
                std_val = float(np.std(raw_vals))
                self._gp_mean = float(np.mean(raw_vals))
                self._gp_std = std_val if std_val > 1e-4 else 1.0
            except Exception:
                self._gp_mean = 0.0
                self._gp_std = 1.0

        @property
        def gp_expression_string(self) -> str:
            if self._gp_expression is None:
                return "(none)"
            return self._gp_expression.to_string()

        # ------------------------------------------------------------------ #
        # Forward pass
        # ------------------------------------------------------------------ #

        def forward(
            self,
            vector_inputs: torch.Tensor,
            sequence_inputs: torch.Tensor = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
            batch_size = vector_inputs.size(0)
            fuzzy_score, firing_strengths = self.fuzzy_engine(vector_inputs)
            vector_feats = self.dense_vector_net(vector_inputs)

            if sequence_inputs is not None:
                lstm_out, _ = self.bilstm(sequence_inputs)
                seq_feats = lstm_out[:, -1, :]
            else:
                # Fallback: broadcast the static metric vector across the window
                # so vector-only inference matches the trained sequence length.
                window = MODEL_CONFIG.get('sequence_length', 20)
                dummy_seq = vector_inputs.unsqueeze(1).repeat(1, window, 1)
                lstm_out, _ = self.bilstm(dummy_seq)
                seq_feats = lstm_out[:, -1, :]

            # ── GP signal (1-D per sample) ─────────────────────────────────
            # Evaluated in NumPy (GP expression is a pure-numpy Node), then
            # standardized with calibrated domain statistics to eliminate constant DC bias.
            if self._gp_expression is not None:
                x_np = vector_inputs.detach().cpu().numpy()          # (B, 4)
                gp_raw = self._gp_expression.evaluate(x_np)          # (B,)
                gp_raw = np.clip(gp_raw, -1e6, 1e6)                   # safety
                gp_norm = (gp_raw - self._gp_mean) / (self._gp_std + 1e-6)
                gp_norm = np.clip(gp_norm, -5.0, 5.0)                # bounded standardized signal
                gp_tensor = torch.tensor(
                    gp_norm, dtype=torch.float32, device=vector_inputs.device
                ).unsqueeze(1)                                         # (B, 1)
                gp_signal = gp_tensor * self.gp_weight
            else:
                gp_signal = torch.zeros(
                    batch_size, 1,
                    dtype=vector_inputs.dtype,
                    device=vector_inputs.device,
                )                                                       # (B, 1)

            combined = torch.cat(
                [seq_feats, fuzzy_score, vector_feats, gp_signal], dim=1
            )  # (B, 162)
            fused = self.fusion_net(combined)

            risk_score = self.risk_score_head(fused) * 100.0
            category_logits = self.classification_head(fused)
            predicted_mdd = self.drawdown_head(fused)

            fuzzy_info = {
                'fuzzy_score': fuzzy_score,
                'firing_strengths': firing_strengths,
                'gp_signal': gp_signal,
            }

            return risk_score, category_logits, predicted_mdd, fuzzy_info

        def predict_risk(
            self,
            vector_inputs: np.ndarray,
            sequence_inputs: Optional[np.ndarray] = None
        ) -> Tuple[float, str, float, Dict[str, str]]:
            """
            Scores one instrument.

            ``sequence_inputs`` (shape ``[time_steps, 4]``) is optional; when
            omitted the vector is broadcast across the window, which preserves
            the original vector-only behaviour.
            """
            self.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(vector_inputs, dtype=torch.float32).unsqueeze(0)
                seq_tensor = None
                if sequence_inputs is not None:
                    seq = np.asarray(sequence_inputs, dtype=np.float32)
                    if seq.ndim == 2:
                        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

                risk_score_t, logits_t, mdd_t, _ = self.forward(x_tensor, seq_tensor)

                risk_score = float(risk_score_t.item())
                predicted_mdd = float(mdd_t.item())

                probs = F.softmax(logits_t, dim=1).squeeze(0).numpy()
                category_idx = int(np.argmax(probs))
                category_name = RISK_CATEGORIES[category_idx]

                fuzzy_labels = self.fuzzy_engine.evaluate_fuzzification_labels(vector_inputs)

            return risk_score, category_name, predicted_mdd, fuzzy_labels

else:
    # Pure NumPy Fallback Engine for Deep Neuro-Fuzzy Risk Model
    class DeepNeuroFuzzyRiskModel:
        """
        Pure NumPy Deep Neuro-Fuzzy Risk Model (GP-aware).
        """

        def __init__(self, input_dim: int = 4, num_categories: int = 4):
            self.fuzzy_engine = PyTorchTakagiSugenoFuzzyLayer()
            np.random.seed(42)
            self.w1 = np.random.randn(input_dim, 32) * 0.1
            self.w2 = np.random.randn(32, 16) * 0.1
            self.w_out = np.random.randn(16, 1) * 0.1
            self.w_mdd = np.random.randn(16, 1) * 0.1
            self.w_cat = np.random.randn(16, num_categories) * 0.1
            # GP integration
            self._gp_expression = None
            self._gp_mean = 0.0
            self._gp_std = 1.0
            self.gp_weight = 0.1

            if hasattr(self.fuzzy_engine, 'clamp_and_sort_parameters'):
                self.fuzzy_engine.clamp_and_sort_parameters()

        def set_gp_expression(self, node) -> None:
            self._gp_expression = node
            if node is None:
                self._gp_mean = 0.0
                self._gp_std = 1.0
                return

            rng = np.random.default_rng(42)
            probe_grid = np.column_stack([
                rng.uniform(-2.0, 4.0, 500),
                rng.uniform(0.0, 0.8, 500),
                rng.uniform(0.05, 1.20, 500),
                rng.uniform(0.05, 1.20, 500),
            ]).astype(np.float32)

            try:
                raw_vals = np.clip(node.evaluate(probe_grid), -1e6, 1e6)
                std_val = float(np.std(raw_vals))
                self._gp_mean = float(np.mean(raw_vals))
                self._gp_std = std_val if std_val > 1e-4 else 1.0
            except Exception:
                self._gp_mean = 0.0
                self._gp_std = 1.0

        @property
        def gp_expression_string(self) -> str:
            if self._gp_expression is None:
                return "(none)"
            return self._gp_expression.to_string()

        def relu(self, x):
            return np.maximum(0, x)

        def sigmoid(self, x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

        def forward(self, vector_inputs: np.ndarray, sequence_inputs=None):
            if vector_inputs.ndim == 1:
                vector_inputs = vector_inputs[np.newaxis, :]

            fuzzy_score, firing_strengths = self.fuzzy_engine.forward(vector_inputs)

            h1 = self.relu(np.dot(vector_inputs, self.w1))
            h2 = self.relu(np.dot(h1, self.w2))

            # GP signal normalized
            if self._gp_expression is not None:
                gp_raw = self._gp_expression.evaluate(vector_inputs)  # (B,)
                gp_raw = np.clip(gp_raw, -1e6, 1e6)
                gp_norm = (gp_raw - self._gp_mean) / (self._gp_std + 1e-6)
                gp_norm = np.clip(gp_norm, -5.0, 5.0)
                gp_contribution = gp_norm * self.gp_weight
            else:
                gp_contribution = 0.0

            # Combine fuzzy risk score, GP signal and deep dense representations
            risk_score = self.sigmoid(
                np.dot(h2, self.w_out) + fuzzy_score * 0.05 + gp_contribution * 0.001
            ) * 100.0
            predicted_mdd = self.sigmoid(np.dot(h2, self.w_mdd))
            category_logits = np.dot(h2, self.w_cat)

            fuzzy_info = {
                'fuzzy_score': fuzzy_score,
                'firing_strengths': firing_strengths,
                'gp_signal': gp_contribution,
            }

            return risk_score, category_logits, predicted_mdd, fuzzy_info

        def predict_risk(self, vector_inputs: np.ndarray) -> Tuple[float, str, float, Dict[str, str]]:
            risk_score_arr, logits_arr, mdd_arr, _ = self.forward(vector_inputs)
            
            risk_score = float(risk_score_arr[0, 0])
            predicted_mdd = float(mdd_arr[0, 0])

            exp_logits = np.exp(logits_arr[0] - np.max(logits_arr[0]))
            probs = exp_logits / np.sum(exp_logits)
            
            # Map risk score to risk category
            if risk_score < 25.0:
                category_idx = 0
            elif risk_score < 50.0:
                category_idx = 1
            elif risk_score < 75.0:
                category_idx = 2
            else:
                category_idx = 3

            category_name = RISK_CATEGORIES[category_idx]
            fuzzy_labels = self.fuzzy_engine.evaluate_fuzzification_labels(vector_inputs)

            return risk_score, category_name, predicted_mdd, fuzzy_labels
