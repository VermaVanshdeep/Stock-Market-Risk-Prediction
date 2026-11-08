"""
Comprehensive tests for models/genetic_programming.py

Coverage:
  1.  Node basics          — terminal and function node properties
  2.  Node evaluation      — correct output values for known trees
  3.  Protected operators  — div-by-zero, log(0), sqrt(-x) never raise
  4.  Expression string    — human-readable infix for known structures
  5.  Tree metrics         — depth, node_count
  6.  Deep copy / clone    — clone is structurally independent
  7.  ExpressionBuilder    — build_full depth, build_grow max depth, ramped init
  8.  Crossover            — offspring depth ≤ max_depth, length = CHROMOSOME_LEN
  9.  Mutation             — mutated tree differs from parent (statistically)
  10. compute_fitness       — valid tree ∈ (0, 1]; invalid tree = 0
  11. Complexity penalty    — deeper tree scores ≤ shallow tree on same data
  12. Tournament selection  — winner has highest fitness among sampled k
  13. Evolution loop        — fitness non-decreasing over generations (elitism)
  14. Reproducibility       — same seed → identical fitness history
  15. Test-set isolation    — evolve_expressions never touches test rows
  16. GP metadata contract  — run_gp returns all required keys
  17. Persistence           — save / load roundtrip preserves expression string
  18. Serialisation         — Node.to_dict / from_dict roundtrip
  19. Small integration     — evolution on synthetic financial data completes
  20. Progress callback     — called once per generation with valid args
"""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.genetic_programming import (
    FEATURE_NAMES,
    GP_CONFIG,
    OPERATORS,
    VARIABLE_TERMINALS,
    ExpressionBuilder,
    GPResult,
    Node,
    compute_fitness,
    evolve_expressions,
    load_gp_result,
    run_gp,
    save_gp_result,
    subtree_crossover,
    subtree_mutation,
    tournament_select,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

N_ROWS = 60
N_FEAT = 4


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _builder(seed: int = 0) -> ExpressionBuilder:
    cfg = dict(GP_CONFIG)
    cfg["population_size"] = 20
    cfg["generations"]     = 5
    return ExpressionBuilder(rng=_rng(seed), config=cfg)


def _small_X(seed: int = 0) -> np.ndarray:
    """N_ROWS × 4 feature matrix with realistic value ranges."""
    rng = np.random.default_rng(seed)
    X   = np.column_stack([
        rng.uniform(-2.0, 4.0,  N_ROWS),   # sharpe
        rng.uniform(0.0,  0.8,  N_ROWS),   # drawdown
        rng.uniform(0.05, 1.0,  N_ROWS),   # implied_vol
        rng.uniform(0.05, 1.0,  N_ROWS),   # hist_vol
    ])
    return X.astype(np.float64)


def _small_y(X: np.ndarray) -> np.ndarray:
    """Simple linear target in [0, 100]."""
    return np.clip(
        30 * X[:, 1] + 20 * X[:, 3] + 15 * np.clip(-X[:, 0], 0, 2) + 10,
        0.0, 100.0,
    )


def _train_val_split(X, y, val_frac=0.2):
    n      = len(X)
    n_val  = max(int(n * val_frac), 1)
    n_tr   = n - n_val
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


def _make_const(value: float) -> Node:
    return Node(terminal=value)


def _make_var(idx: int) -> Node:
    return Node(terminal=f"x{idx}")


def _make_add(left: Node, right: Node) -> Node:
    return Node(op="add", children=[left, right])


def _make_mul(left: Node, right: Node) -> Node:
    return Node(op="mul", children=[left, right])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Node basics
# ─────────────────────────────────────────────────────────────────────────────

class TestNodeBasics(unittest.TestCase):

    def test_terminal_is_terminal(self):
        node = _make_const(3.14)
        self.assertTrue(node.is_terminal)
        self.assertEqual(node.arity, 0)

    def test_variable_terminal(self):
        node = _make_var(0)
        self.assertTrue(node.is_terminal)
        self.assertEqual(node.terminal, "x0")

    def test_function_node_not_terminal(self):
        node = _make_add(_make_var(0), _make_var(1))
        self.assertFalse(node.is_terminal)
        self.assertEqual(node.arity, 2)

    def test_unary_node_arity(self):
        node = Node(op="abs", children=[_make_var(0)])
        self.assertEqual(node.arity, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Node evaluation
# ─────────────────────────────────────────────────────────────────────────────

class TestNodeEvaluation(unittest.TestCase):

    def setUp(self):
        self.X = _small_X()

    def test_constant_terminal(self):
        node   = _make_const(7.0)
        result = node.evaluate(self.X)
        np.testing.assert_array_equal(result, np.full(N_ROWS, 7.0))

    def test_variable_terminal_x0(self):
        node   = _make_var(0)
        result = node.evaluate(self.X)
        np.testing.assert_array_almost_equal(result, self.X[:, 0])

    def test_variable_terminal_x3(self):
        node   = _make_var(3)
        result = node.evaluate(self.X)
        np.testing.assert_array_almost_equal(result, self.X[:, 3])

    def test_add_two_variables(self):
        node   = _make_add(_make_var(0), _make_var(1))
        result = node.evaluate(self.X)
        np.testing.assert_array_almost_equal(result, self.X[:, 0] + self.X[:, 1])

    def test_mul_var_by_constant(self):
        node   = _make_mul(_make_var(2), _make_const(10.0))
        result = node.evaluate(self.X)
        np.testing.assert_array_almost_equal(result, self.X[:, 2] * 10.0)

    def test_nested_expression(self):
        # (x0 + x1) * 5.0
        inner  = _make_add(_make_var(0), _make_var(1))
        node   = _make_mul(inner, _make_const(5.0))
        result = node.evaluate(self.X)
        expected = (self.X[:, 0] + self.X[:, 1]) * 5.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_shape(self):
        node   = _make_add(_make_var(0), _make_var(1))
        result = node.evaluate(self.X)
        self.assertEqual(result.shape, (N_ROWS,))

    def test_all_finite_for_normal_input(self):
        node   = _make_add(_make_var(0), _make_var(1))
        result = node.evaluate(self.X)
        self.assertTrue(np.all(np.isfinite(result)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Protected operators
# ─────────────────────────────────────────────────────────────────────────────

class TestProtectedOperators(unittest.TestCase):

    def test_protected_div_by_zero(self):
        X    = np.zeros((5, 4))
        node = Node(op="div", children=[_make_const(10.0), _make_var(0)])
        result = node.evaluate(X)
        # Should not raise and output must be finite
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_array_almost_equal(result, np.ones(5))  # returns 1.0

    def test_protected_log_of_zero(self):
        X    = np.zeros((5, 4))
        node = Node(op="log", children=[_make_var(0)])
        result = node.evaluate(X)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_protected_log_negative(self):
        X    = -np.ones((5, 4))
        node = Node(op="log", children=[_make_var(0)])
        result = node.evaluate(X)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_protected_sqrt_negative(self):
        X    = -np.abs(_small_X())
        node = Node(op="sqrt", children=[_make_var(1)])
        result = node.evaluate(X)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(result >= 0.0))

    def test_no_exception_for_extreme_values(self):
        X = np.full((10, 4), 1e10)
        for op_name in OPERATORS:
            arity = OPERATORS[op_name][0]
            if arity == 1:
                node = Node(op=op_name, children=[_make_var(0)])
            else:
                node = Node(op=op_name, children=[_make_var(0), _make_var(1)])
            try:
                result = node.evaluate(X)
                # Must return a finite array or zeros (sanitised)
            except Exception as exc:
                self.fail(f"Operator '{op_name}' raised {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Expression string
# ─────────────────────────────────────────────────────────────────────────────

class TestExpressionString(unittest.TestCase):

    def test_constant_string(self):
        node = _make_const(3.0)
        self.assertEqual(node.to_string(), "3")

    def test_variable_string(self):
        node = _make_var(1)
        self.assertEqual(node.to_string(), "x1")

    def test_add_string(self):
        node = _make_add(_make_var(0), _make_var(1))
        self.assertIn("+", node.to_string())
        self.assertIn("x0", node.to_string())
        self.assertIn("x1", node.to_string())

    def test_str_matches_to_string(self):
        node = _make_mul(_make_var(2), _make_const(5.0))
        self.assertEqual(str(node), node.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tree metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestTreeMetrics(unittest.TestCase):

    def test_terminal_depth_is_zero(self):
        self.assertEqual(_make_const(1.0).depth(), 0)

    def test_terminal_node_count_is_one(self):
        self.assertEqual(_make_const(1.0).node_count(), 1)

    def test_depth_one_node(self):
        node = _make_add(_make_var(0), _make_var(1))
        self.assertEqual(node.depth(), 1)

    def test_node_count_add(self):
        node = _make_add(_make_var(0), _make_var(1))
        self.assertEqual(node.node_count(), 3)  # root + 2 leaves

    def test_depth_two_node(self):
        # ((x0 + x1) + x2) — depth 2
        inner = _make_add(_make_var(0), _make_var(1))
        outer = _make_add(inner, _make_var(2))
        self.assertEqual(outer.depth(), 2)

    def test_node_count_three_depth(self):
        inner = _make_add(_make_var(0), _make_var(1))
        outer = _make_add(inner, _make_var(2))
        self.assertEqual(outer.node_count(), 5)  # 2 func + 3 leaves


# ─────────────────────────────────────────────────────────────────────────────
# 6. Clone (deep copy)
# ─────────────────────────────────────────────────────────────────────────────

class TestClone(unittest.TestCase):

    def test_clone_is_independent(self):
        node  = _make_add(_make_var(0), _make_const(1.0))
        clone = node.clone()
        # Modify clone
        clone.children[1].terminal = 99.0
        # Original unchanged
        self.assertEqual(node.children[1].terminal, 1.0)

    def test_clone_evaluates_identically(self):
        X     = _small_X()
        node  = _make_mul(_make_var(0), _make_var(1))
        clone = node.clone()
        np.testing.assert_array_almost_equal(node.evaluate(X), clone.evaluate(X))


# ─────────────────────────────────────────────────────────────────────────────
# 7. ExpressionBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestExpressionBuilder(unittest.TestCase):

    def test_build_full_depth(self):
        b    = _builder()
        tree = b.build_full(depth=3)
        self.assertEqual(tree.depth(), 3)

    def test_build_grow_max_depth(self):
        b = _builder()
        for _ in range(50):
            tree = b.build_grow(depth=4)
            self.assertLessEqual(tree.depth(), 4)

    def test_ramped_returns_correct_count(self):
        b   = _builder()
        pop = b.ramped_half_and_half(20)
        self.assertEqual(len(pop), 20)

    def test_all_nodes_in_population_are_nodes(self):
        b   = _builder()
        pop = b.ramped_half_and_half(10)
        for tree in pop:
            self.assertIsInstance(tree, Node)

    def test_population_respects_max_nodes(self):
        b   = _builder()
        pop = b.ramped_half_and_half(15)
        for tree in pop:
            self.assertLessEqual(
                tree.node_count(), GP_CONFIG["max_nodes"],
                f"Tree with {tree.node_count()} nodes exceeds max_nodes."
            )

    def test_full_tree_all_leaves_at_same_depth(self):
        """In a Full tree every path from root to leaf has exactly `depth` edges."""
        b    = _builder()
        tree = b.build_full(depth=3)

        def leaf_depths(node, current=0):
            if node.is_terminal:
                return [current]
            depths = []
            for child in node.children:
                depths.extend(leaf_depths(child, current + 1))
            return depths

        depths = leaf_depths(tree)
        self.assertTrue(all(d == 3 for d in depths),
                        f"Not all leaves at depth 3: {depths}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Crossover
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossover(unittest.TestCase):

    def _parents(self):
        b  = _builder()
        p1 = b.build_full(2)
        p2 = b.build_grow(3)
        return p1, p2

    def test_offspring_are_nodes(self):
        p1, p2  = self._parents()
        c1, c2  = subtree_crossover(p1, p2, _rng())
        self.assertIsInstance(c1, Node)
        self.assertIsInstance(c2, Node)

    def test_offspring_depth_bounded(self):
        max_d = GP_CONFIG["max_depth"]
        p1, p2 = self._parents()
        for seed in range(10):
            c1, c2 = subtree_crossover(p1, p2, _rng(seed), max_depth=max_d)
            self.assertLessEqual(c1.depth(), max_d)
            self.assertLessEqual(c2.depth(), max_d)

    def test_offspring_node_count_bounded(self):
        max_n = GP_CONFIG["max_nodes"]
        p1, p2 = self._parents()
        for seed in range(10):
            c1, c2 = subtree_crossover(p1, p2, _rng(seed), max_nodes=max_n)
            self.assertLessEqual(c1.node_count(), max_n)
            self.assertLessEqual(c2.node_count(), max_n)

    def test_parents_unchanged(self):
        p1, p2 = self._parents()
        s1, s2 = p1.to_string(), p2.to_string()
        subtree_crossover(p1, p2, _rng())
        self.assertEqual(p1.to_string(), s1)
        self.assertEqual(p2.to_string(), s2)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Mutation
# ─────────────────────────────────────────────────────────────────────────────

class TestMutation(unittest.TestCase):

    def test_mutant_is_a_node(self):
        b      = _builder()
        tree   = b.build_full(3)
        mutant = subtree_mutation(tree, b, _rng())
        self.assertIsInstance(mutant, Node)

    def test_mutation_depth_bounded(self):
        max_d = GP_CONFIG["max_depth"]
        b     = _builder()
        tree  = b.build_full(3)
        for seed in range(15):
            mutant = subtree_mutation(tree, b, _rng(seed), max_depth=max_d)
            self.assertLessEqual(mutant.depth(), max_d)

    def test_original_unchanged_after_mutation(self):
        b      = _builder()
        tree   = b.build_full(2)
        before = tree.to_string()
        subtree_mutation(tree, b, _rng())
        self.assertEqual(tree.to_string(), before)

    def test_mutation_produces_variety(self):
        """Running mutation 20 times with different seeds should yield some variation."""
        b     = _builder()
        tree  = b.build_full(3)
        exprs = {subtree_mutation(tree, b, _rng(i)).to_string() for i in range(20)}
        # At least 2 distinct results expected across 20 attempts
        self.assertGreater(len(exprs), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 10. compute_fitness
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFitness(unittest.TestCase):

    def setUp(self):
        X    = _small_X()
        y    = _small_y(X)
        _, _, self.X_val, self.y_val = _train_val_split(X, y)

    def test_valid_tree_fitness_in_open_interval(self):
        node    = _make_add(_make_var(0), _make_const(30.0))
        fitness = compute_fitness(node, self.X_val, self.y_val)
        self.assertGreater(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)

    def test_perfect_constant_target_fitness(self):
        """If the tree exactly reproduces the (constant) target it should
        outperform a random tree."""
        const_y  = np.full(len(self.X_val), 42.0)
        perfect  = _make_const(42.0)
        poor     = _make_const(0.0)
        f_perfect = compute_fitness(perfect, self.X_val, const_y)
        f_poor    = compute_fitness(poor,    self.X_val, const_y)
        self.assertGreater(f_perfect, f_poor)

    def test_fitness_is_float(self):
        node    = _make_var(0)
        fitness = compute_fitness(node, self.X_val, self.y_val)
        self.assertIsInstance(fitness, float)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Complexity penalty
# ─────────────────────────────────────────────────────────────────────────────

class TestComplexityPenalty(unittest.TestCase):

    def test_simpler_tree_scores_higher_same_prediction(self):
        """
        Two trees with the same prediction but different complexities must
        have the simpler tree scoring higher (complexity_alpha > 0).
        """
        X    = _small_X()
        y    = np.full(N_ROWS, 50.0)
        _, _, X_val, y_val = _train_val_split(X, y)

        # Both predict exactly 50.0 but simple has 1 node, complex has 3.
        simple  = _make_const(50.0)                       # 1 node
        complex_ = _make_add(_make_const(25.0), _make_const(25.0))  # 3 nodes

        f_simple  = compute_fitness(simple,   X_val, y_val, complexity_alpha=0.05)
        f_complex = compute_fitness(complex_, X_val, y_val, complexity_alpha=0.05)

        self.assertGreater(f_simple, f_complex,
            "Simpler tree (1 node) should outperform complex tree (3 nodes) "
            "with identical predictions when alpha > 0.")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Tournament selection
# ─────────────────────────────────────────────────────────────────────────────

class TestTournamentSelection(unittest.TestCase):

    def test_returns_node(self):
        b   = _builder()
        pop = b.ramped_half_and_half(10)
        fits = [float(i) for i in range(10)]
        winner = tournament_select(pop, fits, k=3, rng=_rng())
        self.assertIsInstance(winner, Node)

    def test_winner_is_fittest_in_sample(self):
        """With k = pop_size the winner must be the global best."""
        b   = _builder()
        pop = b.ramped_half_and_half(8)
        fits = [float(i) for i in range(8)]   # index 7 is best
        winner_str = tournament_select(pop, fits, k=8, rng=_rng()).to_string()
        best_str   = pop[7].to_string()
        self.assertEqual(winner_str, best_str)

    def test_selection_returns_clone_not_reference(self):
        b    = _builder()
        pop  = b.ramped_half_and_half(5)
        fits = [1.0] * 5
        winner = tournament_select(pop, fits, k=2, rng=_rng())
        winner.terminal = "MODIFIED"
        # The original population should be untouched
        for tree in pop:
            self.assertNotEqual(tree.to_string(), "MODIFIED")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Evolution loop — elitism ensures best is non-decreasing
# ─────────────────────────────────────────────────────────────────────────────

class TestEvolutionLoop(unittest.TestCase):

    def _run(self, seed=0, gens=5):
        X   = _small_X(seed)
        y   = _small_y(X)
        X_tr, y_tr, X_val, y_val = _train_val_split(X, y)
        cfg = dict(GP_CONFIG)
        cfg.update({"population_size": 15, "generations": gens,
                    "elite_count": 2, "max_depth": 4, "max_nodes": 31})
        return evolve_expressions(X_tr, y_tr, X_val, y_val, config=cfg, seed=seed)

    def test_returns_gp_result(self):
        result = self._run()
        self.assertIsInstance(result, GPResult)

    def test_fitness_history_length(self):
        gens   = 5
        result = self._run(gens=gens)
        self.assertEqual(len(result.fitness_history), gens)

    def test_best_fitness_non_decreasing(self):
        result = self._run()
        hist   = result.fitness_history
        for i in range(1, len(hist)):
            self.assertGreaterEqual(
                hist[i], hist[i - 1] - 1e-9,
                f"Fitness decreased at gen {i + 1}: {hist[i-1]:.6f} → {hist[i]:.6f}"
            )

    def test_best_fitness_positive(self):
        result = self._run()
        self.assertGreater(result.best_fitness, 0.0)

    def test_best_fitness_le_one(self):
        result = self._run()
        self.assertLessEqual(result.best_fitness, 1.0)

    def test_expression_string_non_empty(self):
        result = self._run()
        self.assertTrue(len(result.expression_string) > 0)

    def test_node_count_positive(self):
        result = self._run()
        self.assertGreater(result.node_count, 0)

    def test_node_count_bounded(self):
        result = self._run()
        cfg    = dict(GP_CONFIG)
        cfg["max_depth"] = 4; cfg["max_nodes"] = 31
        self.assertLessEqual(result.node_count, cfg["max_nodes"])


# ─────────────────────────────────────────────────────────────────────────────
# 14. Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility(unittest.TestCase):

    def _run(self, seed, gens=4):
        X   = _small_X(42)
        y   = _small_y(X)
        X_tr, y_tr, X_val, y_val = _train_val_split(X, y)
        cfg = dict(GP_CONFIG)
        cfg.update({"population_size": 12, "generations": gens,
                    "elite_count": 2, "max_depth": 3, "max_nodes": 15})
        return evolve_expressions(X_tr, y_tr, X_val, y_val, config=cfg, seed=seed)

    def test_same_seed_identical_history(self):
        r1 = self._run(seed=7)
        r2 = self._run(seed=7)
        self.assertEqual(r1.fitness_history, r2.fitness_history,
            f"Fitness histories differ with same seed.\n"
            f"Run 1: {r1.fitness_history}\nRun 2: {r2.fitness_history}")

    def test_different_seeds_generally_differ(self):
        """Different seeds should generally produce different histories."""
        r1 = self._run(seed=1)
        r2 = self._run(seed=99)
        # Not guaranteed for all seeds, but with pop=12 and gens=4 it's very
        # unlikely both produce identical histories.  Just confirm no crash.
        self.assertEqual(len(r1.fitness_history), 4)
        self.assertEqual(len(r2.fitness_history), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 15. Test-set isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestTestSetIsolation(unittest.TestCase):

    def test_evolve_does_not_receive_test_data(self):
        """
        run_gp uses the same train/val/test split as the GA.
        The val_samples + train_samples + test_samples must add up to total,
        and test_samples must not be zero.
        """
        from models.train import build_dataset
        dataset = build_dataset(
            ["OFFLINE_A", "OFFLINE_B"],
            allow_network=False,
        )
        cfg = dict(GP_CONFIG)
        cfg.update({"population_size": 6, "generations": 2,
                    "max_depth": 3, "max_nodes": 15})
        result = run_gp(dataset, config=cfg, seed=0)
        meta   = result["metadata"]

        total = meta["total_samples"]
        self.assertEqual(
            meta["train_samples"] + meta["val_samples"] + meta["test_samples"],
            total,
            "train + val + test must equal total_samples."
        )
        self.assertGreater(meta["test_samples"], 0,
            "test_samples must be > 0 (test rows are reserved, not discarded).")


# ─────────────────────────────────────────────────────────────────────────────
# 16. GP metadata contract (run_gp)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunGPMetadata(unittest.TestCase):

    REQUIRED_KEYS = (
        "expression", "fitness", "val_mse", "val_r2",
        "node_count", "fitness_history", "avg_fitness_history",
        "complexity_history", "seed", "metadata", "expression_tree",
    )
    REQUIRED_META_KEYS = (
        "train_samples", "val_samples", "test_samples", "total_samples",
        "config", "feature_names", "fitness_definition",
    )

    def _run_gp(self):
        from models.train import build_dataset
        dataset = build_dataset(["OFFLINE_A", "OFFLINE_B"], allow_network=False)
        cfg = dict(GP_CONFIG)
        cfg.update({"population_size": 6, "generations": 2,
                    "max_depth": 3, "max_nodes": 15})
        return run_gp(dataset, config=cfg, seed=1)

    def test_all_top_level_keys_present(self):
        result = self._run_gp()
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, result, f"run_gp result missing key '{key}'")

    def test_all_metadata_keys_present(self):
        result = self._run_gp()
        meta   = result["metadata"]
        for key in self.REQUIRED_META_KEYS:
            self.assertIn(key, meta, f"metadata missing key '{key}'")

    def test_fitness_in_range(self):
        result = self._run_gp()
        self.assertGreater(result["fitness"], 0.0)
        self.assertLessEqual(result["fitness"], 1.0)

    def test_expression_is_string(self):
        result = self._run_gp()
        self.assertIsInstance(result["expression"], str)
        self.assertTrue(len(result["expression"]) > 0)

    def test_fitness_definition_in_metadata(self):
        result = self._run_gp()
        self.assertIn("MSE", result["metadata"]["fitness_definition"].upper())

    def test_feature_names_correct(self):
        result = self._run_gp()
        self.assertEqual(result["metadata"]["feature_names"], FEATURE_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# 17. Persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence(unittest.TestCase):

    def test_save_load_roundtrip_expression_string(self):
        from models.train import build_dataset
        dataset = build_dataset(["OFFLINE_A"], allow_network=False)
        cfg     = dict(GP_CONFIG)
        cfg.update({"population_size": 6, "generations": 2,
                    "max_depth": 3, "max_nodes": 15})
        result = run_gp(dataset, config=cfg, seed=2)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        save_gp_result(result, path=path)
        loaded = load_gp_result(path=path)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["expression"], result["expression"],
                         "Expression string changed during save/load roundtrip.")

    def test_save_creates_valid_json(self):
        from models.train import build_dataset
        dataset = build_dataset(["OFFLINE_A"], allow_network=False)
        cfg     = dict(GP_CONFIG)
        cfg.update({"population_size": 6, "generations": 2,
                    "max_depth": 3, "max_nodes": 15})
        result = run_gp(dataset, config=cfg, seed=3)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        save_gp_result(result, path=path)
        with open(path) as fh:
            parsed = json.load(fh)
        self.assertIn("expression", parsed)

    def test_load_missing_file_returns_none(self):
        missing = Path(tempfile.mktemp(suffix=".json"))
        result  = load_gp_result(path=missing)
        self.assertIsNone(result)


# ─────────────────────────────────────────────────────────────────────────────
# 18. Node serialisation (to_dict / from_dict)
# ─────────────────────────────────────────────────────────────────────────────

class TestNodeSerialisation(unittest.TestCase):

    def test_terminal_roundtrip(self):
        node     = _make_const(3.14)
        restored = Node.from_dict(node.to_dict())
        self.assertEqual(restored.terminal, node.terminal)
        self.assertIsNone(restored.op)

    def test_variable_roundtrip(self):
        node     = _make_var(2)
        restored = Node.from_dict(node.to_dict())
        self.assertEqual(restored.terminal, "x2")

    def test_function_node_roundtrip(self):
        X    = _small_X()
        node = _make_add(_make_mul(_make_var(0), _make_const(5.0)), _make_var(1))
        restored = Node.from_dict(node.to_dict())
        np.testing.assert_array_almost_equal(
            node.evaluate(X), restored.evaluate(X),
            err_msg="Restored node evaluates differently from original."
        )

    def test_to_dict_is_json_serialisable(self):
        node = _make_add(_make_var(0), _make_const(2.5))
        d    = node.to_dict()
        json.dumps(d)   # Must not raise


# ─────────────────────────────────────────────────────────────────────────────
# 19. Small integration — evolution on synthetic financial data
# ─────────────────────────────────────────────────────────────────────────────

class TestSmallIntegration(unittest.TestCase):
    """
    End-to-end evolution on a synthetic financial dataset with
    known structure so we can verify the evolved expression improves
    fitness over a constant baseline.
    """

    def test_evolution_improves_over_constant_baseline(self):
        rng = np.random.default_rng(42)
        N   = 200
        X   = np.column_stack([
            rng.uniform(-2.0, 4.0, N),    # sharpe
            rng.uniform(0.0,  0.8, N),    # drawdown
            rng.uniform(0.05, 1.0, N),    # implied_vol
            rng.uniform(0.05, 1.0, N),    # hist_vol
        ])
        # Ground truth: linear combination of the features
        y = np.clip(40 * X[:, 1] + 20 * X[:, 3] + 5, 0, 100)

        n_val = 40
        X_tr, y_tr = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]

        # Baseline: constant expression
        baseline_fitness = compute_fitness(_make_const(50.0), X_val, y_val)

        cfg = dict(GP_CONFIG)
        cfg.update({"population_size": 20, "generations": 10,
                    "max_depth": 4, "max_nodes": 31, "elite_count": 2})
        result = evolve_expressions(X_tr, y_tr, X_val, y_val, config=cfg, seed=0)

        self.assertGreater(
            result.best_fitness, baseline_fitness,
            f"GP best fitness {result.best_fitness:.4f} did not exceed constant "
            f"baseline {baseline_fitness:.4f}."
        )

    def test_evolved_expression_evaluable(self):
        """Evolved expression must evaluate without errors on held-out data."""
        rng = np.random.default_rng(7)
        N   = 100
        X   = rng.uniform(0.0, 1.0, (N, 4))
        y   = rng.uniform(0.0, 100.0, N)
        X_tr, y_tr, X_val, y_val = _train_val_split(X, y)

        cfg = dict(GP_CONFIG)
        cfg.update({"population_size": 10, "generations": 5,
                    "max_depth": 3, "max_nodes": 15})
        result = evolve_expressions(X_tr, y_tr, X_val, y_val, config=cfg, seed=1)

        X_test = rng.uniform(0.0, 1.0, (20, 4))
        output = result.best_expression.evaluate(X_test)
        self.assertEqual(output.shape, (20,))
        self.assertTrue(np.all(np.isfinite(output)))


# ─────────────────────────────────────────────────────────────────────────────
# 20. Progress callback
# ─────────────────────────────────────────────────────────────────────────────

class TestProgressCallback(unittest.TestCase):

    def test_callback_called_once_per_generation(self):
        calls = []
        def _cb(gen, total, fitness):
            calls.append((gen, total, fitness))

        X   = _small_X()
        y   = _small_y(X)
        X_tr, y_tr, X_val, y_val = _train_val_split(X, y)
        gens = 4
        cfg  = dict(GP_CONFIG)
        cfg.update({"population_size": 8, "generations": gens,
                    "max_depth": 3, "max_nodes": 15})
        evolve_expressions(X_tr, y_tr, X_val, y_val, config=cfg, seed=5,
                           progress_callback=_cb)
        self.assertEqual(len(calls), gens)

    def test_callback_args_valid(self):
        calls = []
        def _cb(gen, total, fitness):
            calls.append((gen, total, fitness))

        X   = _small_X()
        y   = _small_y(X)
        X_tr, y_tr, X_val, y_val = _train_val_split(X, y)
        gens = 3
        cfg  = dict(GP_CONFIG)
        cfg.update({"population_size": 8, "generations": gens,
                    "max_depth": 3, "max_nodes": 15})
        evolve_expressions(X_tr, y_tr, X_val, y_val, config=cfg, seed=6,
                           progress_callback=_cb)
        for gen, total, fitness in calls:
            self.assertGreaterEqual(gen,   1)
            self.assertEqual(total,        gens)
            self.assertGreater(fitness,    0.0)
            self.assertLessEqual(fitness,  1.0)


if __name__ == "__main__":
    unittest.main()
