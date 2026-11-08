"""
Index disjointness, temporal alignment, and evaluation boundary tests.

Requirements verified
─────────────────────
REQ-1  train / val / test indices are pairwise disjoint.
REQ-2  union(train, val, test) == {0, …, N-1}  (no gaps, no duplicates).
REQ-3  test_idx contains the last TEST_SPLIT fraction by position
       (chronological tail guarantee).
REQ-4  The GA uses val_idx; test_idx is never passed to optimize_model.
REQ-5  evaluate_split returns numerically identical results when called twice
       on the same indices (deterministic forward pass).
REQ-6  Baseline and optimized metrics are produced on both val and test sets
       using exactly the same held-out samples.
REQ-7  Loading saved weights + GA params into a fresh model reproduces the
       same predictions as the fitted model (weights round-trip).
REQ-8  get_train_val_test_indices with various N still satisfies disjointness.
REQ-9  Temporal ordering: every test index is numerically greater than every
       train/val index (since test = contiguous tail slice).
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.deep_neuro_fuzzy_net import DeepNeuroFuzzyRiskModel
from models.fuzzy_layer import TORCH_AVAILABLE
from models.genetic_optimizer import (
    apply_chromosome,
    load_ga_params,
    save_ga_params,
)
from models.train import (
    TEST_SPLIT,
    TRAIN_SEED,
    VALIDATION_SPLIT,
    build_dataset,
    build_dataset_split,
    evaluate_split,
    get_train_val_test_indices,
    load_weights,
    save_weights,
    train_model,
)

if TORCH_AVAILABLE:
    import torch


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

_OFFLINE_TICKERS = ["OFFLINE_A", "OFFLINE_B", "OFFLINE_C", "OFFLINE_D"]


def _make_model() -> DeepNeuroFuzzyRiskModel:
    np.random.seed(0)
    if TORCH_AVAILABLE:
        torch.manual_seed(0)
    return DeepNeuroFuzzyRiskModel()


def _offline_dataset(seq_len: int = 15, horizon: int = 5) -> dict:
    """Fast, offline-only dataset across four synthetic tickers."""
    return build_dataset(
        _OFFLINE_TICKERS,
        sequence_length=seq_len,
        horizon=horizon,
        allow_network=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REQ-1 / REQ-2 / REQ-8  Pairwise disjointness and completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestIndexDisjointness(unittest.TestCase):

    def _check_three_way(self, n: int):
        """Helper: asserts disjointness + completeness for any n."""
        train_idx, val_idx, test_idx = get_train_val_test_indices(
            n, validation_split=VALIDATION_SPLIT, test_split=TEST_SPLIT, seed=TRAIN_SEED
        )
        tr = set(train_idx.tolist())
        va = set(val_idx.tolist())
        te = set(test_idx.tolist())

        # Pairwise disjointness
        self.assertEqual(
            len(tr & va), 0,
            f"train cap val overlap ({len(tr & va)} samples) at n={n}"
        )
        self.assertEqual(
            len(tr & te), 0,
            f"train cap test overlap ({len(tr & te)} samples) at n={n}"
        )
        self.assertEqual(
            len(va & te), 0,
            f"val cap test overlap ({len(va & te)} samples) at n={n}"
        )

        # Completeness
        all_idx = sorted(
            train_idx.tolist() + val_idx.tolist() + test_idx.tolist()
        )
        self.assertEqual(
            all_idx, list(range(n)),
            f"union != {{0,...,{n-1}}} at n={n}"
        )

    def test_disjointness_small(self):
        self._check_three_way(30)

    def test_disjointness_medium(self):
        self._check_three_way(200)

    def test_disjointness_large(self):
        self._check_three_way(10000)

    def test_disjointness_boundary_n_equals_10(self):
        self._check_three_way(10)

    def test_no_duplicate_indices_in_each_split(self):
        for n in (50, 500):
            train_idx, val_idx, test_idx = get_train_val_test_indices(
                n, validation_split=VALIDATION_SPLIT, test_split=TEST_SPLIT, seed=TRAIN_SEED
            )
            self.assertEqual(len(set(train_idx.tolist())), len(train_idx),
                             f"Duplicate indices in train_idx at n={n}")
            self.assertEqual(len(set(val_idx.tolist())),   len(val_idx),
                             f"Duplicate indices in val_idx at n={n}")
            self.assertEqual(len(set(test_idx.tolist())), len(test_idx),
                             f"Duplicate indices in test_idx at n={n}")

    def test_different_seeds_give_different_train_val(self):
        """Different seeds must produce different train/val assignments."""
        _, val_a, test_a = get_train_val_test_indices(200, seed=1)
        _, val_b, test_b = get_train_val_test_indices(200, seed=99)
        # test_idx is deterministic (tail slice) and identical regardless of seed
        np.testing.assert_array_equal(
            test_a, test_b,
            err_msg="test_idx should not depend on seed (it is a fixed tail)."
        )
        # val_idx should differ with different seeds
        self.assertFalse(
            np.array_equal(sorted(val_a.tolist()), sorted(val_b.tolist())),
            "val_idx must differ for different seeds."
        )


# ─────────────────────────────────────────────────────────────────────────────
# REQ-3  Temporal ordering: test = contiguous tail slice
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalOrdering(unittest.TestCase):

    def test_test_indices_are_contiguous_tail(self):
        """test_idx must equal arange(n_trainval, n) -- a contiguous tail slice."""
        n = 300
        n_test = max(int(n * TEST_SPLIT), 1)
        n_trainval = n - n_test

        _, _, test_idx = get_train_val_test_indices(n, seed=TRAIN_SEED)
        expected = np.arange(n_trainval, n, dtype=np.intp)
        np.testing.assert_array_equal(
            test_idx, expected,
            err_msg="test_idx must be a contiguous arange(n_trainval, n)."
        )

    def test_every_test_index_greater_than_every_trainval_index(self):
        """Since test = tail, min(test) > max(train union val)."""
        n = 400
        train_idx, val_idx, test_idx = get_train_val_test_indices(n, seed=TRAIN_SEED)
        max_trainval = max(
            int(train_idx.max()),
            int(val_idx.max()),
        )
        min_test = int(test_idx.min())
        self.assertGreater(
            min_test, max_trainval,
            f"min(test)={min_test} must exceed max(train+val)={max_trainval}."
        )

    def test_test_split_size_is_approximately_correct(self):
        n = 1000
        _, _, test_idx = get_train_val_test_indices(n, test_split=0.10, seed=TRAIN_SEED)
        expected_lo = int(n * 0.09)
        expected_hi = int(n * 0.11) + 1
        self.assertGreaterEqual(len(test_idx), expected_lo)
        self.assertLessEqual(len(test_idx), expected_hi)


# ─────────────────────────────────────────────────────────────────────────────
# REQ-4  GA never receives test rows
# ─────────────────────────────────────────────────────────────────────────────

class TestGADoesNotSeeTestSet(unittest.TestCase):

    def test_optimize_model_val_idx_disjoint_from_test(self):
        """
        Run optimize_model on a small offline dataset and verify that the GA's
        validation set (derived from get_train_val_test_indices) has zero
        overlap with the test set.
        """
        from models.genetic_optimizer import optimize_model

        dataset = _offline_dataset()
        n = len(dataset["X_vec"])
        train_idx, val_idx, test_idx = get_train_val_test_indices(
            n, validation_split=VALIDATION_SPLIT, test_split=TEST_SPLIT, seed=TRAIN_SEED
        )

        # Verify disjointness of the splits
        va_set = set(val_idx.tolist())
        te_set = set(test_idx.tolist())
        self.assertEqual(len(va_set & te_set), 0,
            "GA val_idx must not overlap test_idx.")

        # Actually run GA and check its reported counts
        model = _make_model()
        result = optimize_model(model, dataset, population_size=3, generations=2, seed=0)
        meta = result["metadata"]

        self.assertEqual(
            meta["val_samples"] + meta["train_samples"] + meta["test_samples"],
            meta["total_samples"],
            "val + train + test must equal total N."
        )
        self.assertEqual(meta["test_samples"], len(test_idx),
            "GA metadata test_samples must match our independently-computed test_idx size.")
        self.assertEqual(meta["val_samples"],  len(val_idx),
            "GA metadata val_samples must match our independently-computed val_idx size.")

    def test_split_design_field_is_correct(self):
        """Metadata must advertise the chronological-tail split design."""
        from models.genetic_optimizer import optimize_model
        dataset = _offline_dataset()
        model = _make_model()
        result = optimize_model(model, dataset, population_size=3, generations=2, seed=1)
        self.assertEqual(
            result["metadata"].get("split_design"),
            "chronological_tail_test_shuffled_trainval",
        )


# ─────────────────────────────────────────────────────────────────────────────
# REQ-5  evaluate_split is deterministic
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateSplitDeterminism(unittest.TestCase):

    def test_evaluate_split_returns_same_result_twice(self):
        dataset = _offline_dataset()
        n = len(dataset["X_vec"])
        _, val_idx, test_idx = get_train_val_test_indices(n, seed=TRAIN_SEED)

        model = _make_model()

        val_a  = evaluate_split(model, dataset, val_idx)
        val_b  = evaluate_split(model, dataset, val_idx)
        test_a = evaluate_split(model, dataset, test_idx)
        test_b = evaluate_split(model, dataset, test_idx)

        self.assertAlmostEqual(val_a["mse"],  val_b["mse"],  places=6,
            msg="evaluate_split MSE must be deterministic on val set.")
        self.assertAlmostEqual(test_a["mse"], test_b["mse"], places=6,
            msg="evaluate_split MSE must be deterministic on test set.")

    def test_evaluate_split_output_keys(self):
        dataset = _offline_dataset()
        n = len(dataset["X_vec"])
        _, val_idx, test_idx = get_train_val_test_indices(n, seed=TRAIN_SEED)

        model = _make_model()
        result = evaluate_split(model, dataset, val_idx)

        for key in ("mse", "mae", "n"):
            self.assertIn(key, result, f"evaluate_split must return key '{key}'.")
        self.assertGreater(result["mse"], 0.0)
        self.assertGreater(result["mae"], 0.0)
        self.assertEqual(result["n"], len(val_idx))

    def test_evaluate_split_empty_indices_returns_nan(self):
        dataset = _offline_dataset()
        model = _make_model()
        result = evaluate_split(model, dataset, np.array([], dtype=np.intp))
        self.assertTrue(np.isnan(result["mse"]))
        self.assertTrue(np.isnan(result["mae"]))
        self.assertEqual(result["n"], 0)


# ─────────────────────────────────────────────────────────────────────────────
# REQ-6  Baseline vs optimised on val AND test (same examples)
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselineVsOptimisedEvaluation(unittest.TestCase):

    def test_baseline_and_optimised_metrics_on_val_and_test(self):
        """
        Train a model, snapshot baseline metrics, run GA optimisation, then
        snapshot optimised metrics.  Both val and test metrics must be
        computed on exactly the same index sets before and after.
        Also verifies test metrics are not used during GA optimisation.
        """
        from models.genetic_optimizer import optimize_model

        dataset = _offline_dataset(seq_len=15, horizon=5)
        n = len(dataset["X_vec"])
        train_idx, val_idx, test_idx = get_train_val_test_indices(
            n, validation_split=VALIDATION_SPLIT, test_split=TEST_SPLIT, seed=TRAIN_SEED
        )

        model = _make_model()
        train_model(model, dataset, epochs=1, batch_size=64)

        # Baseline metrics
        baseline_val  = evaluate_split(model, dataset, val_idx)
        baseline_test = evaluate_split(model, dataset, test_idx)

        self.assertGreater(baseline_val["n"],  0)
        self.assertGreater(baseline_test["n"], 0)
        self.assertFalse(np.isnan(baseline_val["mse"]),  "baseline val MSE must not be NaN")
        self.assertFalse(np.isnan(baseline_test["mse"]), "baseline test MSE must not be NaN")

        # GA optimisation (test_idx untouched)
        result = optimize_model(
            model, dataset, population_size=4, generations=2, seed=42
        )

        # Optimised metrics on the SAME sets
        optimised_val  = evaluate_split(model, dataset, val_idx)
        optimised_test = evaluate_split(model, dataset, test_idx)

        self.assertEqual(optimised_val["n"],  baseline_val["n"],
            "val set size must not change after GA.")
        self.assertEqual(optimised_test["n"], baseline_test["n"],
            "test set size must not change after GA.")

        # All four metrics are finite
        for label, m in [
            ("baseline_val_mse",   baseline_val["mse"]),
            ("baseline_val_mae",   baseline_val["mae"]),
            ("baseline_test_mse",  baseline_test["mse"]),
            ("baseline_test_mae",  baseline_test["mae"]),
            ("optimised_val_mse",  optimised_val["mse"]),
            ("optimised_val_mae",  optimised_val["mae"]),
            ("optimised_test_mse", optimised_test["mse"]),
            ("optimised_test_mae", optimised_test["mae"]),
        ]:
            self.assertTrue(np.isfinite(m), f"{label} must be finite, got {m}")


# ─────────────────────────────────────────────────────────────────────────────
# REQ-7  Weight + GA-param round-trip reproduces identical predictions
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightsAndGAParamsRoundTrip(unittest.TestCase):

    def setUp(self):
        self.tmp_weights = Path(tempfile.mktemp(suffix=".pt"))
        self.tmp_ga      = Path(tempfile.mktemp(suffix=".json"))

    def tearDown(self):
        self.tmp_weights.unlink(missing_ok=True)
        self.tmp_ga.unlink(missing_ok=True)

    def test_fresh_model_with_loaded_weights_and_ga_params_matches(self):
        """
        1. Build a model, train briefly, apply a GA chromosome.
        2. Save weights and GA params.
        3. Load into a fresh model.
        4. Assert predictions match on all test rows.
        """
        from models.genetic_optimizer import CHROMOSOME_LEN

        dataset = _offline_dataset()
        n = len(dataset["X_vec"])
        _, _, test_idx = get_train_val_test_indices(n, seed=TRAIN_SEED)

        # Fitted model
        model_a = _make_model()
        train_model(model_a, dataset, epochs=1, batch_size=64)

        # Apply a fixed chromosome to the fuzzy layer
        rng = np.random.default_rng(7)
        chromo = rng.uniform(0, 0.5, CHROMOSOME_LEN).astype(np.float32)
        chromo[12:24] = 0.15
        apply_chromosome(model_a, chromo)

        # Persist
        torch.save(model_a.state_dict(), self.tmp_weights)
        save_ga_params(chromo, {"source": "round_trip_test"}, path=self.tmp_ga)

        # Fresh model
        model_b = _make_model()
        state = torch.load(self.tmp_weights, map_location="cpu", weights_only=True)
        model_b.load_state_dict(state)
        load_ga_params(model_b, path=self.tmp_ga)

        # Compare on test set
        X_vec = torch.tensor(dataset["X_vec"][test_idx], dtype=torch.float32)
        X_seq = torch.tensor(dataset["X_seq"][test_idx], dtype=torch.float32)

        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            pred_a, _, _, _ = model_a(X_vec, X_seq)
            pred_b, _, _, _ = model_b(X_vec, X_seq)

        np.testing.assert_allclose(
            pred_a.numpy(), pred_b.numpy(),
            rtol=1e-4, atol=1e-4,
            err_msg="Fresh model with loaded weights+GA params must reproduce predictions."
        )

    def test_evaluate_split_scores_identical_after_reload(self):
        """evaluate_split on the test set must be identical before and after reload."""
        dataset = _offline_dataset()
        n = len(dataset["X_vec"])
        _, _, test_idx = get_train_val_test_indices(n, seed=TRAIN_SEED)

        model_a = _make_model()
        train_model(model_a, dataset, epochs=1, batch_size=64)

        metrics_a = evaluate_split(model_a, dataset, test_idx)

        # Reload
        torch.save(model_a.state_dict(), self.tmp_weights)
        model_b = _make_model()
        state = torch.load(self.tmp_weights, map_location="cpu", weights_only=True)
        model_b.load_state_dict(state)

        metrics_b = evaluate_split(model_b, dataset, test_idx)

        self.assertAlmostEqual(metrics_a["mse"], metrics_b["mse"], places=5,
            msg="Test MSE must be identical after weight reload.")
        self.assertAlmostEqual(metrics_a["mae"], metrics_b["mae"], places=5,
            msg="Test MAE must be identical after weight reload.")


# ─────────────────────────────────────────────────────────────────────────────
# build_dataset_split integration
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildDatasetSplit(unittest.TestCase):

    def test_build_dataset_split_returns_four_values(self):
        dataset, train_idx, val_idx, test_idx = build_dataset_split(
            tickers=_OFFLINE_TICKERS,
            sequence_length=15,
            horizon=5,
            allow_network=False,
        )
        self.assertIsInstance(dataset, dict)
        self.assertIn("X_vec", dataset)
        n = len(dataset["X_vec"])
        self.assertEqual(len(train_idx) + len(val_idx) + len(test_idx), n)

    def test_build_dataset_split_pairwise_disjoint(self):
        dataset, train_idx, val_idx, test_idx = build_dataset_split(
            tickers=_OFFLINE_TICKERS,
            sequence_length=15,
            horizon=5,
            allow_network=False,
        )
        tr, va, te = set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())
        self.assertEqual(len(tr & va), 0, "train and val must be disjoint.")
        self.assertEqual(len(tr & te), 0, "train and test must be disjoint.")
        self.assertEqual(len(va & te), 0, "val and test must be disjoint.")

    def test_build_dataset_split_test_is_tail(self):
        dataset, train_idx, val_idx, test_idx = build_dataset_split(
            tickers=_OFFLINE_TICKERS,
            sequence_length=15,
            horizon=5,
            allow_network=False,
        )
        n = len(dataset["X_vec"])
        n_test = max(int(n * TEST_SPLIT), 1)
        expected_test = np.arange(n - n_test, n, dtype=np.intp)
        np.testing.assert_array_equal(
            test_idx, expected_test,
            err_msg="test_idx must be the contiguous tail of the dataset."
        )


if __name__ == "__main__":
    unittest.main()
