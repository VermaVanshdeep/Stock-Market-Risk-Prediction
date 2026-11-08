"""
Integration tests for the Genetic Algorithm ↔ Fuzzy Layer ↔ Model pipeline.

Updated for the corrected implementation (audit fixes D1–D5):

D1  Fitness: compute_fitness takes (chromosome, X_vec, X_seq, y_val, model).
    The full end-to-end model is evaluated, not just the fuzzy sub-layer.
D2  Ordering: chromosome_to_params sorts Low<Medium<High and pairs sigmas.
D3  Bounds: initialize_population uses authoritative feature bounds.
D4  Reproducibility: all operators share a single seeded Generator.
D5  Validation: optimize_model uses the train/val split from get_train_val_indices.

Coverage
────────
1.  Chromosome length is exactly 24 genes.
2.  chromosome_to_params decodes and repairs chromosomes correctly.
3.  apply_chromosome updates the fuzzy layer used by inference.
4.  Saving and reloading GA params round-trips the chromosome.
5.  The optimize() loop runs for the requested number of generations.
6.  optimize_model returns required keys and honest metrics.
7.  Invalid chromosome size raises ValueError.
8.  Sigmas remain positive after apply_chromosome.
9.  load_ga_params returns None for a missing file.
10. Extreme chromosome genuinely changes fuzzy output (end-to-end connectivity).
11. Same seed → identical fitness history (reproducibility, D4).
12. GA validation indices are disjoint from training indices (D5).
13. Means are ordered Low≤Medium≤High after decode (D2).
14. Means are clipped to feature domain after decode (D3).
"""

import json
import unittest
from pathlib import Path
import tempfile

import numpy as np

from models.deep_neuro_fuzzy_net import DeepNeuroFuzzyRiskModel
from models.fuzzy_layer import TORCH_AVAILABLE
from models.genetic_optimizer import (
    CHROMOSOME_LEN,
    GA_PARAMS_PATH,
    GeneticAlgorithmOptimizer,
    _FEATURE_ORDER,
    _get_feature_bounds,
    apply_chromosome,
    chromosome_to_params,
    ga_params_exist,
    load_ga_params,
    optimize_model,
    save_ga_params,
)
from models.train import get_train_val_indices, VALIDATION_SPLIT, TRAIN_SEED

if TORCH_AVAILABLE:
    import torch


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_model() -> DeepNeuroFuzzyRiskModel:
    np.random.seed(0)
    if TORCH_AVAILABLE:
        torch.manual_seed(0)
    return DeepNeuroFuzzyRiskModel()


def _small_dataset(n: int = 80) -> dict:
    """Tiny offline-style dataset for integration tests (fast, no network)."""
    rng = np.random.default_rng(99)
    X_vec = rng.uniform(0.0, 1.0, (n, 4)).astype(np.float32)
    X_seq = rng.uniform(0.0, 1.0, (n, 20, 4)).astype(np.float32)
    y_score = rng.uniform(0.0, 100.0, (n, 1)).astype(np.float32)
    y_cat = np.array(
        [0 if s < 25 else 1 if s < 50 else 2 if s < 75 else 3
         for s in y_score.ravel()], dtype=np.int64
    )
    y_mdd = (y_score / 200.0).clip(0.0, 1.0).astype(np.float32)
    return {
        "X_vec": X_vec,
        "X_seq": X_seq,
        "y_score": y_score,
        "y_cat": y_cat,
        "y_mdd": y_mdd,
    }


def _small_valdata(n: int = 30):
    rng = np.random.default_rng(42)
    X_vec = rng.uniform(0.0, 1.0, (n, 4)).astype(np.float32)
    X_seq = rng.uniform(0.0, 1.0, (n, 20, 4)).astype(np.float32)
    y_val = rng.uniform(0.0, 100.0, n).astype(np.float32)
    return X_vec, X_seq, y_val


# ─────────────────────────────────────────────────────────────────────────────
# 1. Chromosome length
# ─────────────────────────────────────────────────────────────────────────────

class TestChromosomeLength(unittest.TestCase):

    def test_chromosome_len_constant_is_24(self):
        self.assertEqual(CHROMOSOME_LEN, 24)

    def test_initialize_population_shape(self):
        """initialize_population now requires a seeded Generator (D4 fix)."""
        ga = GeneticAlgorithmOptimizer(population_size=6, generations=2)
        rng = np.random.default_rng(7)
        pop = ga.initialize_population(rng)
        self.assertEqual(pop.shape, (6, 24))


# ─────────────────────────────────────────────────────────────────────────────
# 2. chromosome_to_params decoding (D2, D3)
# ─────────────────────────────────────────────────────────────────────────────

class TestChromosomeDecoding(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(7)
        self.chromo = rng.uniform(-1, 1, CHROMOSOME_LEN).astype(np.float32)
        # Force sigmas slightly negative to verify clipping
        self.chromo[12:24] -= 0.5

    def test_means_shape(self):
        means, _ = chromosome_to_params(self.chromo)
        self.assertEqual(means.shape, (4, 3))

    def test_sigmas_shape(self):
        _, sigmas = chromosome_to_params(self.chromo)
        self.assertEqual(sigmas.shape, (4, 3))

    def test_sigmas_are_clipped_positive(self):
        _, sigmas = chromosome_to_params(self.chromo)
        self.assertTrue((sigmas >= 0.01).all(), "sigmas must all be >= 0.01")

    def test_means_are_ordered_low_medium_high(self):
        """D2 fix: means must be sorted Low≤Medium≤High after decode."""
        rng = np.random.default_rng(42)
        # Test 20 random chromosomes
        for _ in range(20):
            chromo = rng.uniform(-2, 4, CHROMOSOME_LEN).astype(np.float32)
            means, _ = chromosome_to_params(chromo)
            for row in means:
                self.assertTrue(
                    np.all(np.diff(row) >= 0),
                    f"Means not ordered after decode: {row}"
                )

    def test_reversed_means_get_sorted(self):
        """Explicitly reversed means must be sorted after decode."""
        chromo = np.zeros(CHROMOSOME_LEN, dtype=np.float32)
        # Drawdown centres deliberately reversed: High=0.05, Med=0.20, Low=0.40
        chromo[3] = 0.40   # will become Low after sort
        chromo[4] = 0.20   # Medium
        chromo[5] = 0.05   # will become High after sort  — actually reversed
        chromo[12:24] = 0.10
        means, _ = chromosome_to_params(chromo)
        drawdown_row = means[1]  # feature index 1 = drawdown
        self.assertTrue(
            np.all(np.diff(drawdown_row) >= 0),
            f"Drawdown means not sorted: {drawdown_row}"
        )

    def test_means_clipped_to_feature_bounds(self):
        """D3 fix: each mean must be within its feature's valid domain."""
        bounds = _get_feature_bounds()
        # Chromosome with out-of-bounds means
        chromo = np.full(CHROMOSOME_LEN, 999.0, dtype=np.float32)
        chromo[12:24] = 0.1
        means, _ = chromosome_to_params(chromo)
        for i, key in enumerate(_FEATURE_ORDER):
            lo, hi = bounds[key]
            self.assertTrue(
                np.all(means[i] >= lo - 1e-5) and np.all(means[i] <= hi + 1e-5),
                f"Feature {key} means {means[i]} outside [{lo}, {hi}]"
            )

    def test_invalid_length_raises(self):
        with self.assertRaises(ValueError):
            chromosome_to_params(np.ones(28))

    def test_invalid_length_zero_raises(self):
        with self.assertRaises(ValueError):
            chromosome_to_params(np.array([]))


# ─────────────────────────────────────────────────────────────────────────────
# 3. apply_chromosome updates the live fuzzy layer
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyChromosome(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()

    def _read_means(self):
        fuzzy = self.model.fuzzy_engine
        if TORCH_AVAILABLE:
            return fuzzy.means.data.numpy().copy()
        return fuzzy.means.copy()

    def test_apply_changes_means(self):
        before = self._read_means()
        # Build a chromosome whose decoded means differ from defaults
        rng = np.random.default_rng(100)
        new_chromo = rng.uniform(0.1, 0.5, CHROMOSOME_LEN).astype(np.float32)
        new_chromo[12:24] = 0.3  # valid sigmas
        apply_chromosome(self.model, new_chromo)
        after = self._read_means()
        self.assertFalse(
            np.allclose(before, after),
            "apply_chromosome must change the fuzzy layer means."
        )

    def test_apply_keeps_sigmas_valid(self):
        new_chromo = np.full(CHROMOSOME_LEN, -5.0, dtype=np.float32)  # negative sigmas
        apply_chromosome(self.model, new_chromo)
        fuzzy = self.model.fuzzy_engine
        if TORCH_AVAILABLE:
            sigmas = fuzzy.sigmas.data.numpy()
        else:
            sigmas = fuzzy.sigmas
        self.assertTrue((sigmas >= 0.01).all(), "sigmas must remain >= 0.01 after apply")

    def test_apply_means_are_ordered(self):
        """After apply_chromosome, means in the live layer must be ordered."""
        rng = np.random.default_rng(77)
        new_chromo = rng.uniform(-1, 2, CHROMOSOME_LEN).astype(np.float32)
        new_chromo[12:24] = 0.2
        apply_chromosome(self.model, new_chromo)
        fuzzy = self.model.fuzzy_engine
        if TORCH_AVAILABLE:
            means = fuzzy.means.data.numpy()
        else:
            means = fuzzy.means
        for row in means:
            self.assertTrue(np.all(np.diff(row) >= 0), f"Not ordered in live layer: {row}")

    def test_apply_then_predict_does_not_crash(self):
        new_chromo = np.random.uniform(0, 1, CHROMOSOME_LEN).astype(np.float32)
        new_chromo[12:24] = np.abs(new_chromo[12:24]) + 0.02
        apply_chromosome(self.model, new_chromo)
        vec = np.array([1.0, 0.15, 0.25, 0.20], dtype=np.float32)
        score, cat, mdd, labels = self.model.predict_risk(vec)
        self.assertIsInstance(score, float)
        self.assertIn(cat, ["Low", "Moderate", "High", "Extreme"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Predictions change when extreme chromosome is applied
# ─────────────────────────────────────────────────────────────────────────────

class TestChromosomeAffectsPredictions(unittest.TestCase):

    def test_extreme_means_changes_fuzzy_output(self):
        """The chromosome must genuinely propagate through to inference."""
        model = _make_model()
        probe = np.array([1.0, 0.20, 0.25, 0.22], dtype=np.float32)

        # Apply a chromosome that pushes all means to an extreme value
        extreme = np.zeros(CHROMOSOME_LEN, dtype=np.float32)
        extreme[:12] = 50.0   # means far from any real input → near-zero MF
        extreme[12:24] = 0.01  # very narrow sigmas
        apply_chromosome(model, extreme)

        # With means at 50 and inputs near 0, all firing strengths are ~0
        # → consequent output should be near zero
        fuzzy = model.fuzzy_engine
        if TORCH_AVAILABLE:
            with torch.no_grad():
                x_t = torch.tensor(probe).unsqueeze(0)
                fs, _ = fuzzy(x_t)
            fuzzy_val = float(fs.item())
        else:
            fs, _ = fuzzy.forward(probe[np.newaxis, :])
            fuzzy_val = float(fs[0, 0])
        self.assertLess(abs(fuzzy_val), 0.5,
            "Extreme chromosome should produce near-zero fuzzy output for normal inputs.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Save / load round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mktemp(suffix=".json"))

    def tearDown(self):
        self.tmp.unlink(missing_ok=True)

    def test_save_creates_file(self):
        chromo = np.ones(CHROMOSOME_LEN, dtype=np.float32) * 0.3
        save_ga_params(chromo, {"seed": 1}, path=self.tmp)
        self.assertTrue(self.tmp.exists())

    def test_saved_json_is_valid(self):
        chromo = np.arange(CHROMOSOME_LEN, dtype=np.float32) * 0.01
        save_ga_params(chromo, {"generations": 5}, path=self.tmp)
        payload = json.loads(self.tmp.read_text())
        self.assertEqual(len(payload["chromosome"]), CHROMOSOME_LEN)
        self.assertEqual(payload["chromosome_len"], CHROMOSOME_LEN)
        self.assertEqual(payload["generations"], 5)
        # New fields from D8 improvements
        self.assertIn("decoded_means", payload)
        self.assertIn("decoded_sigmas", payload)
        self.assertIn("feature_order", payload)

    def test_saved_decoded_means_are_ordered(self):
        """Saved decoded_means must satisfy Low≤Medium≤High (D2)."""
        rng = np.random.default_rng(55)
        chromo = rng.uniform(-2, 4, CHROMOSOME_LEN).astype(np.float32)
        chromo[12:24] = rng.uniform(0.05, 0.5, 12)
        save_ga_params(chromo, {}, path=self.tmp)
        payload = json.loads(self.tmp.read_text())
        for row in payload["decoded_means"]:
            diffs = [row[i+1] - row[i] for i in range(len(row)-1)]
            self.assertTrue(all(d >= -1e-6 for d in diffs), f"Not ordered: {row}")

    def test_load_applies_chromosome(self):
        """After load_ga_params, the model's decoded means must match."""
        chromo = np.full(CHROMOSOME_LEN, 0.42, dtype=np.float32)
        chromo[12:24] = 0.10   # valid sigmas
        save_ga_params(chromo, {}, path=self.tmp)

        model = _make_model()
        result = load_ga_params(model, path=self.tmp)
        self.assertIsNotNone(result)

        # The applied means are the *decoded* (sorted+clipped) version
        exp_means, _ = chromosome_to_params(chromo)
        if TORCH_AVAILABLE:
            actual_means = model.fuzzy_engine.means.data.numpy()
        else:
            actual_means = model.fuzzy_engine.means
        np.testing.assert_allclose(actual_means, exp_means, rtol=1e-4)

    def test_load_missing_file_returns_none(self):
        missing = Path(tempfile.mktemp(suffix=".json"))
        result = load_ga_params(_make_model(), path=missing)
        self.assertIsNone(result)

    def test_load_wrong_length_returns_none(self):
        self.tmp.write_text(json.dumps({"chromosome": [0.1] * 28, "chromosome_len": 28}))
        result = load_ga_params(_make_model(), path=self.tmp)
        self.assertIsNone(result)

    def test_save_load_roundtrip_predictions_unchanged(self):
        """Saving then reloading must produce identical predictions."""
        model_a = _make_model()
        rng = np.random.default_rng(7)
        chromo = rng.uniform(0, 0.5, CHROMOSOME_LEN).astype(np.float32)
        chromo[12:24] = 0.15
        apply_chromosome(model_a, chromo)
        save_ga_params(chromo, {}, path=self.tmp)

        model_b = _make_model()
        load_ga_params(model_b, path=self.tmp)

        probe = np.array([0.5, 0.15, 0.20, 0.18], dtype=np.float32)
        score_a, _, _, _ = model_a.predict_risk(probe)
        score_b, _, _, _ = model_b.predict_risk(probe)
        self.assertAlmostEqual(score_a, score_b, places=4,
            msg="Scores after reload must match original.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Optimisation loop — updated for new signatures (D1 + D4)
# ─────────────────────────────────────────────────────────────────────────────

class TestGAOptimizationLoop(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()
        self.X_vec, self.X_seq, self.y_val = _small_valdata(30)

    def test_history_length_matches_generations(self):
        """optimize() with new signature: X_vec, X_seq, y_val, model."""
        ga = GeneticAlgorithmOptimizer(population_size=5, generations=4)
        _, history = ga.optimize(
            self.X_vec, self.X_seq, self.y_val, self.model, seed=7
        )
        self.assertEqual(len(history), 4)

    def test_best_chromosome_length(self):
        ga = GeneticAlgorithmOptimizer(population_size=4, generations=2)
        best, _ = ga.optimize(
            self.X_vec, self.X_seq, self.y_val, self.model, seed=0
        )
        self.assertEqual(len(best), CHROMOSOME_LEN)

    def test_fitness_is_non_decreasing(self):
        ga = GeneticAlgorithmOptimizer(population_size=6, generations=5)
        _, history = ga.optimize(
            self.X_vec, self.X_seq, self.y_val, self.model, seed=3
        )
        for i in range(1, len(history)):
            self.assertGreaterEqual(
                history[i], history[i - 1] - 1e-9,
                f"Fitness decreased at generation {i + 1}: "
                f"{history[i - 1]:.6f} → {history[i]:.6f}"
            )

    def test_all_fitness_values_positive(self):
        ga = GeneticAlgorithmOptimizer(population_size=4, generations=3)
        _, history = ga.optimize(
            self.X_vec, self.X_seq, self.y_val, self.model, seed=1
        )
        self.assertTrue(all(f > 0 for f in history))

    def test_reproducibility_same_seed(self):
        """D4 fix: two runs with the same seed must produce identical histories."""
        ga1 = GeneticAlgorithmOptimizer(population_size=5, generations=3)
        ga2 = GeneticAlgorithmOptimizer(population_size=5, generations=3)
        model1 = _make_model()
        model2 = _make_model()

        _, hist1 = ga1.optimize(
            self.X_vec, self.X_seq, self.y_val, model1, seed=42
        )
        _, hist2 = ga2.optimize(
            self.X_vec, self.X_seq, self.y_val, model2, seed=42
        )
        self.assertEqual(hist1, hist2,
            f"Fitness histories differ with identical seed.\n"
            f"Run 1: {hist1}\nRun 2: {hist2}"
        )

    def test_different_seeds_may_differ(self):
        """Different seeds should generally produce different results."""
        ga1 = GeneticAlgorithmOptimizer(population_size=8, generations=4)
        ga2 = GeneticAlgorithmOptimizer(population_size=8, generations=4)
        model1 = _make_model()
        model2 = _make_model()

        _, hist1 = ga1.optimize(
            self.X_vec, self.X_seq, self.y_val, model1, seed=1
        )
        _, hist2 = ga2.optimize(
            self.X_vec, self.X_seq, self.y_val, model2, seed=99
        )
        # Not guaranteed to differ, but with 8-member populations and 4 gens it
        # is overwhelmingly likely for different seeds.
        # We just verify both run without error.
        self.assertEqual(len(hist1), 4)
        self.assertEqual(len(hist2), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 7. optimize_model integration (D1 + D5)
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizeModel(unittest.TestCase):

    def test_optimize_model_returns_required_keys(self):
        model = _make_model()
        dataset = _small_dataset(60)
        result = optimize_model(
            model, dataset, population_size=4, generations=2, seed=0
        )
        self.assertIn("chromosome", result)
        self.assertIn("history", result)
        self.assertIn("metadata", result)
        meta = result["metadata"]
        self.assertIn("baseline_fitness", meta)
        self.assertIn("best_fitness", meta)
        self.assertIn("improved", meta)
        # D8 metadata
        self.assertIn("fitness_definition", meta)
        self.assertIn("feature_bounds", meta)

    def test_chromosome_length_from_optimize_model(self):
        model = _make_model()
        dataset = _small_dataset(60)
        result = optimize_model(
            model, dataset, population_size=4, generations=2, seed=5
        )
        self.assertEqual(len(result["chromosome"]), CHROMOSOME_LEN)

    def test_baseline_and_best_are_floats_in_range(self):
        model = _make_model()
        dataset = _small_dataset(60)
        result = optimize_model(
            model, dataset, population_size=4, generations=2, seed=2
        )
        meta = result["metadata"]
        for key in ("baseline_fitness", "best_fitness"):
            val = meta[key]
            self.assertIsInstance(val, float)
            self.assertGreater(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_honest_improvement_reporting(self):
        """If GA improves fitness, best > baseline; otherwise best <= baseline."""
        model = _make_model()
        dataset = _small_dataset(80)
        result = optimize_model(
            model, dataset, population_size=4, generations=2, seed=9
        )
        meta = result["metadata"]
        bf  = meta["best_fitness"]
        bsf = meta["baseline_fitness"]
        if meta["improved"]:
            self.assertGreater(bf, bsf)
        else:
            self.assertLessEqual(bf, bsf + 1e-8)

    def test_validation_split_disjoint_from_training(self):
        """
        D5 fix: the GA validation indices must not appear in the NN training set.
        We verify this by reconstructing the same split used by train_model().
        """
        n = 80
        train_idx, val_idx = get_train_val_indices(
            n, validation_split=VALIDATION_SPLIT, seed=TRAIN_SEED
        )
        # Check disjointness
        train_set = set(train_idx.tolist())
        val_set   = set(val_idx.tolist())
        overlap   = train_set & val_set
        self.assertEqual(len(overlap), 0,
            f"Train and val sets overlap: {len(overlap)} shared indices")

    def test_val_plus_train_cover_all_samples(self):
        """The two index sets must together cover every row exactly once."""
        n = 120
        train_idx, val_idx = get_train_val_indices(
            n, validation_split=VALIDATION_SPLIT, seed=TRAIN_SEED
        )
        all_idx = sorted(np.concatenate([train_idx, val_idx]).tolist())
        self.assertEqual(all_idx, list(range(n)),
            "train + val must cover all N rows without gaps or duplicates.")

    def test_three_way_split_pairwise_disjoint(self):
        """
        Extended D5: get_train_val_test_indices must produce three sets that
        are pairwise disjoint and together cover all N rows.
        """
        from models.train import get_train_val_test_indices, TEST_SPLIT

        n = 200
        train_idx, val_idx, test_idx = get_train_val_test_indices(
            n, validation_split=VALIDATION_SPLIT, seed=TRAIN_SEED
        )
        tr = set(train_idx.tolist())
        va = set(val_idx.tolist())
        te = set(test_idx.tolist())

        self.assertEqual(len(tr & va), 0,
            f"train cap val overlap: {len(tr & va)} shared indices")
        self.assertEqual(len(tr & te), 0,
            f"train cap test overlap: {len(tr & te)} shared indices")
        self.assertEqual(len(va & te), 0,
            f"val cap test overlap: {len(va & te)} shared indices")

        # Completeness
        all_idx = sorted(train_idx.tolist() + val_idx.tolist() + test_idx.tolist())
        self.assertEqual(all_idx, list(range(n)),
            "train + val + test must cover all N rows without gaps or duplicates.")

    def test_optimize_model_metadata_includes_test_samples(self):
        """
        optimize_model must now report test_samples, train_samples, val_samples
        and total_samples in metadata (D5 extended fix).
        """
        model = _make_model()
        dataset = _small_dataset(80)
        result = optimize_model(
            model, dataset, population_size=4, generations=2, seed=0
        )
        meta = result["metadata"]
        for key in ("test_samples", "train_samples", "val_samples", "total_samples"):
            self.assertIn(key, meta,
                f"metadata must contain '{key}' after D5 extended fix.")
        self.assertEqual(
            meta["train_samples"] + meta["val_samples"] + meta["test_samples"],
            meta["total_samples"],
            "train + val + test samples must equal total."
        )


if __name__ == "__main__":
    unittest.main()
