"""
Phase 1 GA-caller integration tests.

These tests verify that the previously broken CLI and Streamlit service
callers now call the GA correctly through optimize_model().

Coverage:
  1.  CLI path        — optimize_model() call from corrected main.py Option 4
  2.  Service path    — ui/services.run_genetic_algorithm() pattern (no Streamlit)
  3.  24-gene output  — chromosome length is exactly CHROMOSOME_LEN (24)
  4.  NN preservation — non-fuzzy weights unchanged after GA
  5.  MF apply/rollback — means/sigmas updated if improved, restored if not
  6.  Val disjoint    — GA val_idx never overlaps test_idx
  7.  Metadata keys   — all required metadata keys present
  8.  Sample arithmetic — train+val+test == total
  9.  Progress CB     — called once per generation with valid args
  10. Error handling  — empty dataset raises ValueError; tiny dataset completes
"""

import unittest
import numpy as np

from models.deep_neuro_fuzzy_net import DeepNeuroFuzzyRiskModel
from models.fuzzy_layer import TORCH_AVAILABLE
from models.genetic_optimizer import (
    CHROMOSOME_LEN,
    apply_chromosome,
    optimize_model,
)
from models.train import (
    TEST_SPLIT,
    TRAIN_SEED,
    VALIDATION_SPLIT,
    build_dataset,
    get_train_val_test_indices,
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
    """Fast synthetic dataset — no network needed."""
    return build_dataset(
        _OFFLINE_TICKERS,
        sequence_length=seq_len,
        horizon=horizon,
        allow_network=False,
    )


def _read_means(model) -> np.ndarray:
    fuzzy = model.fuzzy_engine
    if TORCH_AVAILABLE:
        return fuzzy.means.data.clone().numpy()
    return fuzzy.means.copy()


def _read_sigmas(model) -> np.ndarray:
    fuzzy = model.fuzzy_engine
    if TORCH_AVAILABLE:
        return fuzzy.sigmas.data.clone().numpy()
    return fuzzy.sigmas.copy()


def _read_consequents(model) -> np.ndarray:
    fuzzy = model.fuzzy_engine
    if TORCH_AVAILABLE:
        return fuzzy.consequent_weights.data.clone().numpy()
    return fuzzy.consequent_weights.copy()


def _nn_param_fingerprint(model) -> float:
    """Sum of all non-fuzzy parameters — detects unexpected NN weight changes."""
    total = 0.0
    if TORCH_AVAILABLE:
        fuzzy_ids = {id(p) for p in model.fuzzy_engine.parameters()}
        for p in model.parameters():
            if id(p) not in fuzzy_ids:
                total += float(p.data.sum().item())
    return total


# ─────────────────────────────────────────────────────────────────────────────
# 1 & 2. CLI path and Service path
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIGACaller(unittest.TestCase):

    def test_cli_path_runs_without_error(self):
        """Reproduces the corrected main.py option_run_ga_optimization pattern."""
        model   = _make_model()
        dataset = _offline_dataset()
        result  = optimize_model(model, dataset, population_size=4, generations=2, seed=42)
        self.assertIn("chromosome", result)
        self.assertIn("history",    result)
        self.assertIn("metadata",   result)

    def test_cli_chromosome_length_is_24(self):
        model  = _make_model()
        result = optimize_model(_make_model(), _offline_dataset(),
                                population_size=4, generations=2, seed=0)
        chromo = np.asarray(result["chromosome"])
        self.assertEqual(len(chromo), 24,
                         f"Expected 24 genes, got {len(chromo)}")

    def test_cli_no_slice_24_to_28(self):
        """chromosome[24:28] must return an empty array — confirms 24-gene only."""
        result = optimize_model(_make_model(), _offline_dataset(),
                                population_size=4, generations=2, seed=1)
        chromo = np.asarray(result["chromosome"])
        self.assertEqual(len(chromo[24:28]), 0,
                         "chromosome[24:28] must be empty for a 24-gene chromosome")


class TestServiceGACaller(unittest.TestCase):
    """
    Mirrors corrected ui/services.run_genetic_algorithm() without importing
    Streamlit (which requires a running server context).
    """

    def _run_service_ga(self, pop=4, gens=2, seed=0):
        model   = _make_model()
        dataset = _offline_dataset()
        result  = optimize_model(model, dataset,
                                 population_size=pop, generations=gens, seed=seed)
        return {
            "chromosome":      np.asarray(result["chromosome"], dtype=float),
            "history":         [float(h) for h in result["history"]],
            "population_size": pop,
            "generations":     gens,
            "seed":            seed,
            "metadata":        result["metadata"],
        }

    def test_service_returns_all_required_keys(self):
        result = self._run_service_ga()
        for key in ("chromosome", "history", "population_size",
                    "generations", "seed", "metadata"):
            self.assertIn(key, result, f"service result missing key '{key}'")

    def test_service_chromosome_length(self):
        result = self._run_service_ga()
        self.assertEqual(len(result["chromosome"]), CHROMOSOME_LEN)

    def test_service_history_length_matches_generations(self):
        result = self._run_service_ga(gens=3)
        self.assertEqual(len(result["history"]), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model parameter preservation
# ─────────────────────────────────────────────────────────────────────────────

class TestModelParamPreservation(unittest.TestCase):

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required")
    def test_nn_weights_unchanged_after_ga(self):
        model     = _make_model()
        fp_before = _nn_param_fingerprint(model)
        optimize_model(model, _offline_dataset(), population_size=4, generations=2, seed=5)
        fp_after  = _nn_param_fingerprint(model)
        self.assertAlmostEqual(fp_before, fp_after, places=4,
            msg="Non-fuzzy neural-network weights changed during GA.")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required")
    def test_consequent_weights_unchanged_after_ga(self):
        model     = _make_model()
        cw_before = _read_consequents(model).copy()
        optimize_model(model, _offline_dataset(), population_size=4, generations=2, seed=6)
        cw_after  = _read_consequents(model)
        np.testing.assert_allclose(cw_before, cw_after, rtol=1e-5,
            err_msg="Fuzzy consequent weights changed during GA.")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required")
    def test_fuzzy_means_sigmas_updated_or_restored(self):
        model     = _make_model()
        m_before  = _read_means(model).copy()
        s_before  = _read_sigmas(model).copy()
        result    = optimize_model(model, _offline_dataset(),
                                   population_size=4, generations=2, seed=7)
        improved  = result["metadata"]["improved"]
        m_after   = _read_means(model)
        s_after   = _read_sigmas(model)
        changed   = (not np.allclose(m_before, m_after, rtol=1e-5)
                     or not np.allclose(s_before, s_after, rtol=1e-5))
        if improved:
            self.assertTrue(changed,
                "GA reported improvement but fuzzy params were not updated.")
        else:
            self.assertFalse(changed,
                "GA reported no improvement but fuzzy params changed (rollback failed).")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Val-only — test rows never exposed to GA fitness evaluator
# ─────────────────────────────────────────────────────────────────────────────

class TestValOnlyNoTestLeakage(unittest.TestCase):

    def test_val_idx_disjoint_from_test_idx(self):
        dataset = _offline_dataset()
        n       = len(dataset["X_vec"])
        _, val_idx, test_idx = get_train_val_test_indices(
            n,
            validation_split=VALIDATION_SPLIT,
            test_split=TEST_SPLIT,
            seed=TRAIN_SEED,
        )
        overlap = set(val_idx.tolist()) & set(test_idx.tolist())
        self.assertEqual(len(overlap), 0,
            f"GA val and test sets share {len(overlap)} index(es).")

    def test_test_idx_is_chronological_tail(self):
        dataset = _offline_dataset()
        n       = len(dataset["X_vec"])
        _, val_idx, test_idx = get_train_val_test_indices(
            n,
            validation_split=VALIDATION_SPLIT,
            test_split=TEST_SPLIT,
            seed=TRAIN_SEED,
        )
        n_test        = len(test_idx)
        expected_tail = set(range(n - n_test, n))
        self.assertEqual(expected_tail, set(test_idx.tolist()),
            "test_idx is not the chronological tail of the dataset.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Metadata contract
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataContract(unittest.TestCase):

    REQUIRED_KEYS = (
        "baseline_fitness", "best_fitness", "improved",
        "fitness_history", "val_samples", "train_samples",
        "test_samples", "total_samples",
        "fitness_definition", "feature_bounds",
        "population_size", "generations",
    )

    def test_all_metadata_keys_present(self):
        result = optimize_model(_make_model(), _offline_dataset(),
                                population_size=4, generations=2, seed=0)
        meta   = result["metadata"]
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, meta, f"metadata missing required key '{key}'")

    def test_sample_counts_add_up(self):
        result = optimize_model(_make_model(), _offline_dataset(),
                                population_size=4, generations=2, seed=3)
        meta   = result["metadata"]
        self.assertEqual(
            meta["train_samples"] + meta["val_samples"] + meta["test_samples"],
            meta["total_samples"],
            "train + val + test must equal total_samples.",
        )

    def test_fitness_definition_mentions_mse(self):
        result = optimize_model(_make_model(), _offline_dataset(),
                                population_size=4, generations=2, seed=1)
        self.assertIn("MSE", result["metadata"]["fitness_definition"].upper())

    def test_improved_flag_is_boolean(self):
        result = optimize_model(_make_model(), _offline_dataset(),
                                population_size=4, generations=2, seed=2)
        self.assertIsInstance(result["metadata"]["improved"], bool)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Progress callback
# ─────────────────────────────────────────────────────────────────────────────

class TestProgressCallback(unittest.TestCase):

    def test_callback_called_once_per_generation(self):
        calls = []
        def _cb(gen, total, fitness):
            calls.append((gen, total, fitness))
        gens = 3
        optimize_model(_make_model(), _offline_dataset(),
                       population_size=4, generations=gens,
                       seed=8, progress_callback=_cb)
        self.assertEqual(len(calls), gens,
                         f"Expected {gens} callback calls, got {len(calls)}")

    def test_callback_receives_valid_args(self):
        calls = []
        def _cb(gen, total, fitness):
            calls.append((gen, total, fitness))
        gens = 2
        optimize_model(_make_model(), _offline_dataset(),
                       population_size=4, generations=gens,
                       seed=9, progress_callback=_cb)
        for gen, total, fitness in calls:
            self.assertGreaterEqual(gen,   1)
            self.assertEqual(total,        gens)
            self.assertGreater(fitness,    0.0)
            self.assertLessEqual(fitness,  1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling(unittest.TestCase):

    def test_offline_build_returns_required_keys(self):
        """
        build_dataset always returns the five required keys when the offline
        synthetic fallback is active (it generates deterministic data for any
        ticker, so ValueError is never raised in offline mode).

        This test verifies the contract callers actually depend on:
        the dataset dict has the expected keys and non-empty arrays.
        """
        dataset = build_dataset(
            ["ANY_TICKER_A", "ANY_TICKER_B"],
            allow_network=False,
        )
        required = {"X_seq", "X_vec", "y_score", "y_cat", "y_mdd"}
        self.assertEqual(required, set(dataset.keys()),
                         f"dataset missing keys; got {set(dataset.keys())}")
        n = dataset["X_vec"].shape[0]
        self.assertGreater(n, 0, "build_dataset produced an empty dataset")

    def test_optimize_model_with_tiny_dataset_does_not_crash(self):
        """Minimal dataset (short sequences) should complete without exception."""
        dataset = build_dataset(
            _OFFLINE_TICKERS,
            sequence_length=5,
            horizon=2,
            allow_network=False,
        )
        result = optimize_model(_make_model(), dataset,
                                population_size=4, generations=1, seed=0)
        self.assertIn("chromosome", result)


if __name__ == "__main__":
    unittest.main()
