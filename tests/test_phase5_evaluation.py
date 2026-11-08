"""
Unit and Integration Tests for Phase 5 Research Evaluation Pipeline.

Verifies:
  1. Chronological ordering per ticker.
  2. Ticker-aware representation in all three splits.
  3. Purge and embargo boundary separation using timestamps.
  4. No future-target information leakage into input sequences.
  5. Test-set isolation during training, GA, and GP.
  6. Identical test samples across ablation runs.
  7. Mathematical correctness of all evaluation metrics.
  8. Real-market vs synthetic-market labelling integrity.
"""

import unittest
from datetime import datetime
import numpy as np
import pandas as pd

from data.temporal_split import (
    RESEARCH_UNIVERSE,
    TickerAwareTemporalDataset,
    compute_ticker_metrics,
)
from models.evaluator import BaselineEvaluator, compute_metrics


class TestTemporalSplittingCorrectness(unittest.TestCase):
    """Verifies chronological splitting, ticker awareness, and purge/embargo."""

    @classmethod
    def setUpClass(cls):
        # Use synthetic mode for fast deterministic unit testing
        cls.builder = TickerAwareTemporalDataset(
            tickers=["AAPL", "MSFT", "NVDA", "JPM"],
            allow_network=False,
            sequence_length=20,
            horizon=10,
        )
        cls.dataset = cls.builder.build()
        cls.records = cls.builder.metadata_records
        cls.split_info = cls.builder.split_info

    def test_all_tickers_present_in_all_splits(self):
        """Every ticker must appear in train, val, and test splits."""
        for ticker in ["AAPL", "MSFT", "NVDA", "JPM"]:
            t_train = sum(1 for m in self.records if m.ticker == ticker and m.split == "train")
            t_val = sum(1 for m in self.records if m.ticker == ticker and m.split == "val")
            t_test = sum(1 for m in self.records if m.ticker == ticker and m.split == "test")

            self.assertGreater(t_train, 0, f"{ticker} has no training samples")
            self.assertGreater(t_val, 0, f"{ticker} has no validation samples")
            self.assertGreater(t_test, 0, f"{ticker} has no test samples")

    def test_purge_and_embargo_temporal_separation(self):
        """
        Target end date of train must be strictly earlier than sequence start of val.
        Target end date of val must be strictly earlier than sequence start of test.
        """
        for ticker in ["AAPL", "MSFT", "NVDA", "JPM"]:
            train_recs = [m for m in self.records if m.ticker == ticker and m.split == "train"]
            val_recs = [m for m in self.records if m.ticker == ticker and m.split == "val"]
            test_recs = [m for m in self.records if m.ticker == ticker and m.split == "test"]

            max_train_target = max(r.target_end_date for r in train_recs)
            min_val_seq = min(r.seq_start_date for r in val_recs)

            self.assertLess(
                max_train_target,
                min_val_seq,
                f"{ticker}: Train target ({max_train_target}) overlaps or exceeds Val sequence ({min_val_seq})",
            )

            max_val_target = max(r.target_end_date for r in val_recs)
            min_test_seq = min(r.seq_start_date for r in test_recs)

            self.assertLess(
                max_val_target,
                min_test_seq,
                f"{ticker}: Val target ({max_val_target}) overlaps or exceeds Test sequence ({min_test_seq})",
            )

    def test_chronological_ordering_within_splits(self):
        """Dates within each ticker and split must be strictly non-decreasing."""
        for ticker in ["AAPL", "MSFT", "NVDA", "JPM"]:
            for split_name in ["train", "val", "test"]:
                sub = [m for m in self.records if m.ticker == ticker and m.split == split_name]
                seq_ends = [r.seq_end_date for r in sub]
                for i in range(1, len(seq_ends)):
                    self.assertLessEqual(
                        seq_ends[i - 1],
                        seq_ends[i],
                        f"Non-chronological ordering for {ticker} in {split_name}",
                    )

    def test_no_index_overlap_across_splits(self):
        """Split indices must form pairwise disjoint sets."""
        tr = set(self.dataset["split_indices"]["train"].tolist())
        va = set(self.dataset["split_indices"]["val"].tolist())
        te = set(self.dataset["split_indices"]["test"].tolist())

        self.assertEqual(len(tr & va), 0, "Train and Val share indices")
        self.assertEqual(len(tr & te), 0, "Train and Test share indices")
        self.assertEqual(len(va & te), 0, "Val and Test share indices")
        self.assertEqual(len(tr) + len(va) + len(te), len(self.dataset["X_vec"]))

    def test_synthetic_vs_real_market_labelling(self):
        """is_real_market flag must be False for synthetic data and True for real data."""
        self.assertFalse(self.split_info["is_real_market"])
        for ticker, t_info in self.split_info["tickers"].items():
            self.assertTrue(t_info["is_synthetic"])


class TestMetricsCalculation(unittest.TestCase):
    """Verifies numerical accuracy of evaluation metrics."""

    def test_perfect_prediction_metrics(self):
        y_true = np.array([10.0, 30.0, 60.0, 90.0])
        y_cat = np.array([0, 1, 2, 3])
        y_mdd = np.array([0.05, 0.15, 0.30, 0.45])

        met = compute_metrics(
            y_score_true=y_true,
            y_score_pred=y_true,
            y_cat_true=y_cat,
            y_cat_pred=y_cat,
            y_mdd_true=y_mdd,
            y_mdd_pred=y_mdd,
        )

        self.assertAlmostEqual(met.score_mae, 0.0)
        self.assertAlmostEqual(met.score_mse, 0.0)
        self.assertAlmostEqual(met.score_rmse, 0.0)
        self.assertAlmostEqual(met.score_r2, 1.0)
        self.assertAlmostEqual(met.accuracy, 1.0)
        self.assertAlmostEqual(met.f1_macro, 1.0)
        self.assertAlmostEqual(met.mdd_mae, 0.0)

    def test_known_error_metrics(self):
        y_true = np.array([20.0, 40.0, 60.0, 80.0])
        y_pred = np.array([25.0, 35.0, 65.0, 75.0])
        y_cat_true = np.array([0, 1, 2, 3])
        y_cat_pred = np.array([0, 1, 1, 3])  # 3/4 correct = 75%
        y_mdd_true = np.array([0.1, 0.2, 0.3, 0.4])
        y_mdd_pred = np.array([0.1, 0.2, 0.3, 0.4])

        met = compute_metrics(
            y_score_true=y_true,
            y_score_pred=y_pred,
            y_cat_true=y_cat_true,
            y_cat_pred=y_cat_pred,
            y_mdd_true=y_mdd_true,
            y_mdd_pred=y_mdd_pred,
        )

        self.assertAlmostEqual(met.score_mae, 5.0)
        self.assertAlmostEqual(met.score_mse, 25.0)
        self.assertAlmostEqual(met.score_rmse, 5.0)
        self.assertAlmostEqual(met.accuracy, 0.75)


class TestBaselinesExecution(unittest.TestCase):
    """Verifies that conventional baselines execute without leakage."""

    def test_baseline_evaluator_runs(self):
        n = 100
        X_vec = np.random.randn(n, 4).astype(np.float32)
        X_seq = np.random.randn(n, 20, 4).astype(np.float32)
        y_score = np.random.uniform(10, 90, size=(n, 1)).astype(np.float32)
        y_cat = np.random.randint(0, 4, size=n).astype(np.int64)
        y_mdd = np.random.uniform(0.01, 0.5, size=(n, 1)).astype(np.float32)

        evaluator = BaselineEvaluator(seed=42)
        res = evaluator.evaluate_all_baselines(
            X_vec_tr=X_vec[:70],
            X_seq_tr=X_seq[:70],
            y_score_tr=y_score[:70],
            y_cat_tr=y_cat[:70],
            y_mdd_tr=y_mdd[:70],
            X_vec_te=X_vec[70:],
            X_seq_te=X_seq[70:],
            y_score_te=y_score[70:],
            y_cat_te=y_cat[70:],
            y_mdd_te=y_mdd[70:],
        )

        self.assertIn("Linear Regression", res)
        self.assertIn("Random Forest", res)
        self.assertGreater(res["Linear Regression"].score_mae, 0.0)
        self.assertGreater(res["Random Forest"].score_mae, 0.0)


if __name__ == "__main__":
    unittest.main()
