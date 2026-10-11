"""
Unit & integration tests for the supervised training engine.

These cover the pieces that make the risk score responsive: rolling feature
construction, dataset/target construction, the training loop and weight
persistence. Kept deliberately small so the suite stays fast.
"""

import unittest

import numpy as np
import pandas as pd

from models import train as T
from models.deep_neuro_fuzzy_net import DeepNeuroFuzzyRiskModel
from data.data_loader import SyntheticStockGenerator


class TestRollingMetricFrame(unittest.TestCase):
    """Vectorised rolling feature engineering."""

    def setUp(self):
        self.df = SyntheticStockGenerator.generate_stock_data(
            ticker="TEST", days=400, sigma=0.30, seed=7
        )

    def test_columns_and_shape(self):
        metrics = T.rolling_metric_frame(self.df)
        self.assertEqual(list(metrics.columns), T._FEATURES)
        self.assertEqual(len(metrics), len(self.df))

    def test_values_respect_feature_bounds(self):
        metrics = T.rolling_metric_frame(self.df).dropna()
        self.assertGreater(len(metrics), 100)
        for key in T._FEATURES:
            low, high = T._BOUNDS[key]
            column = metrics[key]
            self.assertGreaterEqual(float(column.min()), low - 1e-6)
            self.assertLessEqual(float(column.max()), high + 1e-6)

    def test_no_infinite_values(self):
        metrics = T.rolling_metric_frame(self.df).dropna()
        values = metrics.to_numpy(dtype=np.float64)
        self.assertTrue(np.isfinite(values).all())

    def test_higher_volatility_series_has_higher_hv(self):
        calm = SyntheticStockGenerator.generate_stock_data(
            ticker="CALM", days=400, sigma=0.10, seed=3
        )
        wild = SyntheticStockGenerator.generate_stock_data(
            ticker="WILD", days=400, sigma=0.60, seed=3
        )
        calm_hv = T.rolling_metric_frame(calm)["hist_vol"].dropna().mean()
        wild_hv = T.rolling_metric_frame(wild)["hist_vol"].dropna().mean()
        self.assertGreater(wild_hv, calm_hv)


TRAIN_TEST_TICKERS = ["OFFLINE_A", "OFFLINE_B", "OFFLINE_C", "OFFLINE_D"]


def _offline_dataset(sequence_length: int = 15, horizon: int = 5) -> "dict":
    """Fast, network-free, multi-regime dataset for the training tests."""
    return T.build_dataset(
        TRAIN_TEST_TICKERS,
        sequence_length=sequence_length,
        horizon=horizon,
        allow_network=False,
    )


class TestDatasetConstruction(unittest.TestCase):
    """Supervised dataset and forward-looking targets."""

    def test_shapes_and_ranges(self):
        dataset = _offline_dataset()
        n = len(dataset["X_vec"])

        self.assertGreater(n, 100)
        self.assertEqual(dataset["X_seq"].shape, (n, 15, 4))
        self.assertEqual(dataset["X_vec"].shape, (n, 4))
        self.assertEqual(dataset["y_score"].shape, (n, 1))
        self.assertEqual(dataset["y_cat"].shape, (n,))
        self.assertEqual(dataset["y_mdd"].shape, (n, 1))

        self.assertGreaterEqual(float(dataset["y_score"].min()), 0.0)
        self.assertLessEqual(float(dataset["y_score"].max()), 100.0)
        self.assertGreaterEqual(float(dataset["y_mdd"].min()), 0.0)
        self.assertLessEqual(float(dataset["y_mdd"].max()), 1.0)
        self.assertTrue(np.isfinite(dataset["X_seq"]).all())

    def test_dataset_is_regime_diverse(self):
        """Targets must span the bands, otherwise the head learns one constant."""
        dataset = _offline_dataset()
        counts = np.bincount(dataset["y_cat"], minlength=4)
        self.assertGreater(int(counts[0]), 0)
        self.assertGreater(int(counts[-1]), 0)
        self.assertGreater(float(dataset["y_score"].std()), 5.0)

    def test_category_buckets_match_score_bands(self):
        dataset = _offline_dataset()
        scores = dataset["y_score"].reshape(-1)
        for score, bucket in zip(scores[:200], dataset["y_cat"][:200]):
            expected = 0 if score < 25 else 1 if score < 50 else 2 if score < 75 else 3
            self.assertEqual(int(bucket), expected)

    def test_sequence_tail_equals_vector(self):
        dataset = _offline_dataset()
        np.testing.assert_allclose(
            dataset["X_seq"][0][-1], dataset["X_vec"][0], rtol=1e-5, atol=1e-6
        )


class TestTrainingLoop(unittest.TestCase):
    """The loop must actually change the weights and improve the score."""

    def test_training_updates_weights_and_records_history(self):
        T.seed_everything(T.TRAIN_SEED)
        model = DeepNeuroFuzzyRiskModel()
        dataset = _offline_dataset()

        before = [p.detach().clone() for p in model.parameters()]
        history = T.train_model(model, dataset, epochs=2, batch_size=64)

        after = [p.detach().clone() for p in model.parameters()]
        changed = any(not torch_equal(a, b) for a, b in zip(before, after))

        self.assertTrue(changed, "training did not update any parameters")
        self.assertEqual(len(history["epoch"]), 2)
        self.assertEqual(len(history["val_mae"]), 2)
        self.assertEqual(len(history["val_acc"]), 2)
        self.assertEqual(history["samples"], len(dataset["X_vec"]))
        self.assertGreater(history["train_samples"], 0)

    def test_trained_scores_vary_with_input(self):
        """Regression guard for the flat-score bug this module exists to fix."""
        T.seed_everything(T.TRAIN_SEED)

        untrained = DeepNeuroFuzzyRiskModel()
        profiles = [
            np.array([2.0, 0.05, 0.12, 0.10], dtype=np.float32),
            np.array([0.1, 0.40, 0.55, 0.50], dtype=np.float32),
            np.array([-1.2, 0.65, 0.85, 0.75], dtype=np.float32),
        ]
        flat = [untrained.predict_risk(v)[0] for v in profiles]
        flat_spread = max(flat) - min(flat)

        model = DeepNeuroFuzzyRiskModel()
        T.train_model(model, _offline_dataset(), epochs=4, batch_size=64)
        scores = [model.predict_risk(v)[0] for v in profiles]
        spread = max(scores) - min(scores)

        self.assertGreater(
            spread, max(5.0, flat_spread * 10),
            f"risk score barely responds to input (spread {spread:.3f}, "
            f"untrained spread {flat_spread:.3f})",
        )

    def test_score_spread_probe(self):
        T.seed_everything(T.TRAIN_SEED)
        model = DeepNeuroFuzzyRiskModel()
        dataset = _offline_dataset()
        T.train_model(model, dataset, epochs=3, batch_size=64)
        self.assertGreater(T.score_spread(model, dataset), 1.0)


class TestPersistence(unittest.TestCase):
    """Weights round-trip through disk unchanged."""

    def setUp(self):
        self.original = T.WEIGHTS_PATH
        self.backup = T.WEIGHTS_PATH.with_suffix(".bak")
        if T.WEIGHTS_PATH.exists():
            self.backup.write_bytes(T.WEIGHTS_PATH.read_bytes())

    def tearDown(self):
        if self.backup.exists():
            self.backup.replace(T.WEIGHTS_PATH)
        elif T.WEIGHTS_PATH.exists():
            T.WEIGHTS_PATH.unlink()

    def test_save_and_load_round_trip(self):
        T.seed_everything(T.TRAIN_SEED)
        model = DeepNeuroFuzzyRiskModel()
        model.train()
        T.save_weights(model)
        self.assertTrue(T.weights_exist())

        T.seed_everything(T.TRAIN_SEED)
        fresh = DeepNeuroFuzzyRiskModel()
        self.assertTrue(T.load_weights(fresh))

        probe = np.array([1.2, 0.15, 0.25, 0.20], dtype=np.float32)
        self.assertAlmostEqual(
            model.predict_risk(probe)[0], fresh.predict_risk(probe)[0], places=5
        )

    def test_load_returns_false_when_missing(self):
        T.WEIGHTS_PATH.unlink(missing_ok=True)
        self.assertFalse(T.load_weights(DeepNeuroFuzzyRiskModel()))


def torch_equal(a, b) -> bool:
    try:
        import torch

        return bool(torch.equal(a, b))
    except Exception:
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))


if __name__ == "__main__":
    unittest.main()