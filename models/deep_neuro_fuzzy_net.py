"""
Deep Learning Core: Deep Neuro-Fuzzy Risk Prediction Network.
Fuses Deep Bidirectional LSTM sequence modeling with Soft Computing Takagi-Sugeno Fuzzy Logic.
Supports both PyTorch and pure NumPy fallback for seamless cross-platform execution.
"""

import numpy as np
from typing import Tuple, Dict
from models.fuzzy_layer import PyTorchTakagiSugenoFuzzyLayer, TORCH_AVAILABLE
from config.settings import MODEL_CONFIG, RISK_CATEGORIES

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DeepNeuroFuzzyRiskModel(nn.Module):
        """
        Hybrid Deep Neuro-Fuzzy Stock Market Risk Network (PyTorch implementation).
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

            fusion_dim = hidden_dim * 2 + 1 + 32
            
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
                dummy_seq = vector_inputs.unsqueeze(1).repeat(1, 10, 1)
                lstm_out, _ = self.bilstm(dummy_seq)
                seq_feats = lstm_out[:, -1, :]

            combined = torch.cat([seq_feats, fuzzy_score, vector_feats], dim=1)
            fused = self.fusion_net(combined)

            risk_score = self.risk_score_head(fused) * 100.0
            category_logits = self.classification_head(fused)
            predicted_mdd = self.drawdown_head(fused)

            fuzzy_info = {
                'fuzzy_score': fuzzy_score,
                'firing_strengths': firing_strengths
            }

            return risk_score, category_logits, predicted_mdd, fuzzy_info

        def predict_risk(self, vector_inputs: np.ndarray) -> Tuple[float, str, float, Dict[str, str]]:
            self.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(vector_inputs, dtype=torch.float32).unsqueeze(0)
                risk_score_t, logits_t, mdd_t, _ = self.forward(x_tensor)

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
        Pure NumPy Deep Neuro-Fuzzy Risk Model.
        """

        def __init__(self, input_dim: int = 4, num_categories: int = 4):
            self.fuzzy_engine = PyTorchTakagiSugenoFuzzyLayer()
            np.random.seed(42)
            self.w1 = np.random.randn(input_dim, 32) * 0.1
            self.w2 = np.random.randn(32, 16) * 0.1
            self.w_out = np.random.randn(16, 1) * 0.1
            self.w_mdd = np.random.randn(16, 1) * 0.1
            self.w_cat = np.random.randn(16, num_categories) * 0.1

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

            # Combine fuzzy risk score with deep dense representations
            risk_score = self.sigmoid(np.dot(h2, self.w_out) + fuzzy_score * 0.05) * 100.0
            predicted_mdd = self.sigmoid(np.dot(h2, self.w_mdd))
            category_logits = np.dot(h2, self.w_cat)

            fuzzy_info = {
                'fuzzy_score': fuzzy_score,
                'firing_strengths': firing_strengths
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
