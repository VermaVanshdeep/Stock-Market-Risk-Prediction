"""
Soft Computing Core: Genetic Programming (GP) Engine for Financial Risk Expression Discovery.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GP Overview
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Genetic Programming discovers symbolic mathematical expressions mapping the
four standard financial risk features to a continuous risk score in [0, 100].

Unlike the GA (which optimises 24 real-valued fuzzy-MF parameters over a fixed
model architecture), GP evolves the *structure* of the expression itself. The
result is a fully interpretable closed-form formula such as:

    risk = 30 * drawdown + 15 * hist_vol + 8 * max(0, -sharpe) + ...

Input features (identical to the existing BiLSTM / fuzzy pipeline):
    x0 : sharpe         ∈ [-2.0, 4.0]
    x1 : drawdown       ∈ [0.0, 0.8]
    x2 : implied_vol    ∈ [0.05, 1.0]
    x3 : hist_vol       ∈ [0.05, 1.0]

Target:
    y_score = clip(future_max_drawdown × 200 + HV × 100, 0, 100)

Architecture
────────────
• Representation  : Binary expression tree.  Each node is either a function
                    node (operator + two children) or a terminal node (input
                    variable or ephemeral random constant).
• Operators       : add, sub, mul, protected_div, abs_, min_, max_,
                    protected_log, protected_sqrt.
• Terminals       : x0…x3 and ephemeral random constants in [-5, 5].
• Max depth       : 6 levels (avoids bloat while allowing nested expressions).
• Max nodes       : 63 (2^(max_depth+1) – 1 for a full binary tree).

Evolutionary operators
──────────────────────
• Initialization  : Ramped half-and-half (50% full / 50% grow; depths 2–5).
• Selection       : Tournament selection (k = 3 by default).
• Crossover       : Subtree crossover — swap a random subtree from each parent.
• Mutation        : Subtree mutation — replace a random subtree with a fresh
                    random subtree of depth ≤ 3.
• Elitism         : Best N individuals survive unchanged into the next generation.

Fitness function
────────────────
    fitness = 1 / (1 + MSE + α × complexity)

    MSE         — mean squared error on the **validation** split (normalised to
                  [0, 100] units, same as y_score).
    complexity  — node count of the expression tree (penalises bloat).
    α           — complexity weight (default 0.05).

    Invalid expressions (NaN / Inf outputs) receive fitness = 0.

Separation of concerns (no test-set leakage)
─────────────────────────────────────────────
evolve_expressions() accepts explicit (X_train, y_train, X_val, y_val) arrays.
It never touches the held-out test split.  The caller (run_gp / test code) is
responsible for the split; the function documented below shows the exact usage.

Persistence
───────────
save_gp_result() writes a JSON file to data_cache/gp_best_expression.json
containing the expression string, fitness, complexity, seed, and config.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from config.settings import DATA_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Constants & public configuration
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: List[str] = ["sharpe", "drawdown", "implied_vol", "hist_vol"]
N_FEATURES: int = len(FEATURE_NAMES)   # 4

# Variable terminals: x0 … x3
VARIABLE_TERMINALS: List[str] = [f"x{i}" for i in range(N_FEATURES)]

GP_CONFIG: Dict[str, Any] = {
    "population_size":   80,
    "generations":       30,
    "max_depth":          6,
    "max_nodes":         63,
    "tournament_k":       3,
    "crossover_rate":   0.85,
    "mutation_rate":    0.10,
    "elite_count":        3,
    "complexity_alpha": 0.05,   # MSE penalty weight per tree node
    "const_low":        -5.0,
    "const_high":        5.0,
    "init_depth_min":    2,
    "init_depth_max":    5,
}

GP_SAVE_PATH = Path(DATA_DIR) / "gp_best_expression.json"

# ─────────────────────────────────────────────────────────────────────────────
# Protected numeric operators
# ─────────────────────────────────────────────────────────────────────────────

def _protected_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise a/b; returns 1.0 where |b| < 1e-8."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) < 1e-8, np.ones_like(a), a / b)
    return result


def _protected_log(a: np.ndarray) -> np.ndarray:
    """log(|a|); returns 0.0 where |a| < 1e-8."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(a) < 1e-8, np.zeros_like(a), np.log(np.abs(a) + 1e-8))
    return result


def _protected_sqrt(a: np.ndarray) -> np.ndarray:
    """sqrt(|a|)."""
    return np.sqrt(np.abs(a))


# Operator registry: name → (arity, callable)
OPERATORS: Dict[str, Tuple[int, Callable]] = {
    "add":   (2, lambda a, b: a + b),
    "sub":   (2, lambda a, b: a - b),
    "mul":   (2, lambda a, b: a * b),
    "div":   (2, _protected_div),
    "abs":   (1, lambda a: np.abs(a)),
    "min":   (2, lambda a, b: np.minimum(a, b)),
    "max":   (2, lambda a, b: np.maximum(a, b)),
    "log":   (1, _protected_log),
    "sqrt":  (1, _protected_sqrt),
}

OPERATOR_NAMES: List[str] = list(OPERATORS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Expression tree node
# ─────────────────────────────────────────────────────────────────────────────

class Node:
    """
    A single node in a GP expression tree.

    Terminal nodes have ``op = None``; their value is stored in ``terminal``.
    Function nodes have ``op`` set to one of OPERATOR_NAMES; their children
    are stored in ``children`` (length == arity).
    """

    __slots__ = ("op", "terminal", "children")

    def __init__(
        self,
        op: Optional[str] = None,
        terminal: Optional[Any] = None,
        children: Optional[List["Node"]] = None,
    ) -> None:
        self.op = op                       # None → terminal node
        self.terminal = terminal           # str (var name) or float (constant)
        self.children: List[Node] = children or []

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def is_terminal(self) -> bool:
        return self.op is None

    @property
    def arity(self) -> int:
        return 0 if self.is_terminal else OPERATORS[self.op][0]

    # ------------------------------------------------------------------ #
    # Tree metrics
    # ------------------------------------------------------------------ #

    def depth(self) -> int:
        if self.is_terminal:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def node_count(self) -> int:
        if self.is_terminal:
            return 1
        return 1 + sum(c.node_count() for c in self.children)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate expression on input matrix X of shape (N, 4).

        Returns a 1-D array of shape (N,).
        Invalid (NaN / Inf) values are replaced by 0.0 for safety.
        """
        result = self._eval(X)
        # Sanitise: replace NaN/Inf with 0 (penalty is applied at fitness level)
        result = np.where(np.isfinite(result), result, 0.0)
        return result.ravel()

    def _eval(self, X: np.ndarray) -> np.ndarray:
        if self.is_terminal:
            if isinstance(self.terminal, str):
                # Variable: 'x0' → column 0, etc.
                idx = int(self.terminal[1:])
                return X[:, idx].astype(float)
            else:
                # Ephemeral constant
                return np.full(X.shape[0], float(self.terminal))
        else:
            fn = OPERATORS[self.op][1]
            child_vals = [c._eval(X) for c in self.children]
            return fn(*child_vals)

    # ------------------------------------------------------------------ #
    # String representation (human-readable infix)
    # ------------------------------------------------------------------ #

    def to_string(self) -> str:
        if self.is_terminal:
            if isinstance(self.terminal, str):
                return self.terminal
            return f"{self.terminal:.4g}"
        else:
            arity = OPERATORS[self.op][0]
            if arity == 1:
                child_str = self.children[0].to_string()
                return f"{self.op}({child_str})"
            else:
                left  = self.children[0].to_string()
                right = self.children[1].to_string()
                sym = {"add": "+", "sub": "-", "mul": "*", "div": "/",
                       "min": "min", "max": "max"}.get(self.op, self.op)
                if self.op in ("min", "max"):
                    return f"{sym}({left}, {right})"
                return f"({left} {sym} {right})"

    def __str__(self) -> str:
        return self.to_string()

    # ------------------------------------------------------------------ #
    # Deep copy
    # ------------------------------------------------------------------ #

    def clone(self) -> "Node":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ #
    # Serialisation (for persistence)
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict:
        return {
            "op":       self.op,
            "terminal": self.terminal,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Node":
        node = cls(op=d["op"], terminal=d["terminal"])
        node.children = [cls.from_dict(c) for c in d.get("children", [])]
        return node


# ─────────────────────────────────────────────────────────────────────────────
# Expression-tree builder
# ─────────────────────────────────────────────────────────────────────────────

class ExpressionBuilder:
    """
    Builds random expression trees using either the *full* or *grow* method.

    Full method  — every path from root to leaf has exactly ``depth`` levels.
    Grow method  — each path has between 1 and ``depth`` levels.

    Ramped half-and-half initialisation uses both methods across a range of
    target depths to produce diverse initial populations.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        config: Dict[str, Any] = GP_CONFIG,
    ) -> None:
        self.rng    = rng
        self.config = config

    # ------------------------------------------------------------------ #
    # Core builders
    # ------------------------------------------------------------------ #

    def build_full(self, depth: int) -> Node:
        """Build a full tree of exactly ``depth`` levels."""
        if depth == 0:
            return self._random_terminal()
        op   = self.rng.choice(OPERATOR_NAMES)
        arity = OPERATORS[op][0]
        return Node(op=op, children=[self.build_full(depth - 1) for _ in range(arity)])

    def build_grow(self, depth: int) -> Node:
        """Build a tree that grows randomly up to ``depth`` levels."""
        if depth == 0:
            return self._random_terminal()
        # At non-zero depth: choose operator or terminal with equal probability
        if self.rng.random() < 0.5:
            return self._random_terminal()
        op    = self.rng.choice(OPERATOR_NAMES)
        arity = OPERATORS[op][0]
        return Node(op=op, children=[self.build_grow(depth - 1) for _ in range(arity)])

    def _random_terminal(self) -> Node:
        """Return a variable or ephemeral constant terminal."""
        if self.rng.random() < 0.7:
            # 70% chance: pick an input variable
            var = VARIABLE_TERMINALS[int(self.rng.integers(0, N_FEATURES))]
            return Node(terminal=var)
        else:
            # 30% chance: ephemeral random constant
            lo = self.config["const_low"]
            hi = self.config["const_high"]
            const = float(self.rng.uniform(lo, hi))
            return Node(terminal=const)

    # ------------------------------------------------------------------ #
    # Ramped half-and-half initialisation
    # ------------------------------------------------------------------ #

    def ramped_half_and_half(self, n: int) -> List[Node]:
        """
        Produce ``n`` trees using ramped half-and-half.

        Depths are cycled over [init_depth_min, init_depth_max]; within each
        depth, half are generated with Full and half with Grow.
        """
        d_min = self.config["init_depth_min"]
        d_max = self.config["init_depth_max"]
        depths = list(range(d_min, d_max + 1))
        population: List[Node] = []
        i = 0
        while len(population) < n:
            d = depths[i % len(depths)]
            if (i // len(depths)) % 2 == 0:
                tree = self.build_full(d)
            else:
                tree = self.build_grow(d)
            # Enforce complexity limit before accepting
            if tree.node_count() <= self.config["max_nodes"]:
                population.append(tree)
            i += 1
            if i > n * 20:          # Safety valve against infinite loop
                population.append(self.build_grow(2))
        return population


# ─────────────────────────────────────────────────────────────────────────────
# Fitness evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_fitness(
    tree: Node,
    X_val: np.ndarray,
    y_val: np.ndarray,
    complexity_alpha: float = GP_CONFIG["complexity_alpha"],
) -> float:
    """
    Compute the fitness of an expression tree on the validation set.

    Fitness = 1 / (1 + MSE + α × node_count)

    where:
        MSE         — mean squared error, both operands normalised to [0, 100].
                      y_val must already be in [0, 100]; the tree output is
                      clipped to [0, 100] before MSE is computed.
        node_count  — total nodes in the tree (complexity proxy).
        α           — complexity weight (default 0.05).

    Invalid expressions (any output that remains NaN/Inf after the Node's own
    sanitisation) are caught here and assigned fitness = 0.0.

    Args:
        tree             : Expression tree to evaluate.
        X_val            : Validation inputs, shape (M, 4).
        y_val            : Validation targets, shape (M,) or (M, 1), in [0, 100].
        complexity_alpha : Weight on the complexity penalty term.

    Returns:
        Scalar fitness in (0, 1].
    """
    y_true = np.asarray(y_val, dtype=float).ravel()

    try:
        raw = tree.evaluate(X_val)       # (M,) already sanitised by Node.evaluate
        # Clip to target range before MSE to avoid unbounded score distortion
        pred = np.clip(raw, 0.0, 100.0)

        if not np.all(np.isfinite(pred)):
            return 0.0

        mse         = float(np.mean((pred - y_true) ** 2))
        complexity  = tree.node_count()
        fitness     = 1.0 / (1.0 + mse + complexity_alpha * complexity)
    except Exception:
        fitness = 0.0

    return float(fitness)


def compute_fitness_combined(
    tree: Node,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    complexity_alpha: float = GP_CONFIG["complexity_alpha"],
    train_weight: float = 0.70,
) -> float:
    """
    Dual-split fitness: blends training MSE and validation MSE.

    Fitness = 1 / (1 + blended_MSE + α × node_count)

    where:
        blended_MSE = train_weight × MSE(train) + (1 − train_weight) × MSE(val)

    Using training data as the primary signal prevents the evolutionary search
    from treating the validation set as a surrogate for the true objective —
    a form of selection-induced overfitting sometimes called "GP generalisation
    collapse". The validation fraction ensures the fitness landscape still
    rewards expressions that generalise beyond the training rows.

    Both MSE values are computed with predictions clipped to [0, 100].
    Invalid expressions receive fitness = 0.0.

    Args:
        tree             : Expression tree to evaluate.
        X_train, y_train : Training split (shape N×4, N).
        X_val,   y_val   : Validation split (shape M×4, M).
        complexity_alpha : Complexity penalty weight.
        train_weight     : Weight on training MSE (default 0.70).

    Returns:
        Scalar fitness in (0, 1].
    """
    try:
        # Training MSE
        pred_tr  = np.clip(tree.evaluate(X_train), 0.0, 100.0)
        y_tr     = np.asarray(y_train, dtype=float).ravel()
        mse_tr   = float(np.mean((pred_tr - y_tr) ** 2))

        # Validation MSE
        pred_val = np.clip(tree.evaluate(X_val), 0.0, 100.0)
        y_vl     = np.asarray(y_val, dtype=float).ravel()
        mse_val  = float(np.mean((pred_val - y_vl) ** 2))

        if not (np.isfinite(mse_tr) and np.isfinite(mse_val)):
            return 0.0

        blended_mse = train_weight * mse_tr + (1.0 - train_weight) * mse_val
        complexity  = tree.node_count()
        fitness     = 1.0 / (1.0 + blended_mse + complexity_alpha * complexity)
    except Exception:
        fitness = 0.0

    return float(fitness)


# ─────────────────────────────────────────────────────────────────────────────
# Evolutionary operators
# ─────────────────────────────────────────────────────────────────────────────

def tournament_select(
    population: List[Node],
    fitnesses: List[float],
    k: int,
    rng: np.random.Generator,
) -> Node:
    """
    Tournament selection: sample k individuals, return the fittest clone.

    Args:
        population : Current population of expression trees.
        fitnesses  : Corresponding fitness values.
        k          : Tournament size.
        rng        : Seeded random generator.

    Returns:
        A deep copy of the winning individual.
    """
    indices = rng.choice(len(population), size=k, replace=False)
    winner  = max(indices, key=lambda i: fitnesses[i])
    return population[winner].clone()


def _all_nodes(tree: Node) -> List[Tuple[Node, Optional[Node], int]]:
    """
    Return all (node, parent, child_index) triples for a tree via BFS.

    The root entry has parent=None and child_index=-1.
    """
    result: List[Tuple[Node, Optional[Node], int]] = []
    queue  = [(tree, None, -1)]
    while queue:
        node, parent, idx = queue.pop(0)
        result.append((node, parent, idx))
        for i, child in enumerate(node.children):
            queue.append((child, node, i))
    return result


def subtree_crossover(
    parent1: Node,
    parent2: Node,
    rng: np.random.Generator,
    max_depth: int = GP_CONFIG["max_depth"],
    max_nodes: int = GP_CONFIG["max_nodes"],
) -> Tuple[Node, Node]:
    """
    Subtree crossover: swap a randomly selected subtree from each parent.

    If the resulting offspring violates depth or node-count limits, the
    original parents are returned unchanged (conservative fallback).

    Args:
        parent1, parent2 : Parent expression trees (will be deep-copied first).
        rng              : Seeded random generator.
        max_depth        : Maximum allowed tree depth after crossover.
        max_nodes        : Maximum allowed node count after crossover.

    Returns:
        Two child expression trees.
    """
    child1 = parent1.clone()
    child2 = parent2.clone()

    nodes1 = _all_nodes(child1)
    nodes2 = _all_nodes(child2)

    # Pick crossover points (any node; preferring non-root for meaningful swap)
    def _pick(nodes):
        candidates = nodes[1:] if len(nodes) > 1 else nodes
        return candidates[int(rng.integers(0, len(candidates)))]

    n1, p1, idx1 = _pick(nodes1)
    n2, p2, idx2 = _pick(nodes2)

    # Swap subtrees
    subtree1 = n1.clone()
    subtree2 = n2.clone()

    if p1 is not None:
        p1.children[idx1] = subtree2
    else:
        child1 = subtree2   # Replaced entire tree

    if p2 is not None:
        p2.children[idx2] = subtree1
    else:
        child2 = subtree1

    # Validity check — if violated, fall back to parents
    if child1.depth() > max_depth or child1.node_count() > max_nodes:
        child1 = parent1.clone()
    if child2.depth() > max_depth or child2.node_count() > max_nodes:
        child2 = parent2.clone()

    return child1, child2


def subtree_mutation(
    tree: Node,
    builder: ExpressionBuilder,
    rng: np.random.Generator,
    max_depth: int = GP_CONFIG["max_depth"],
    max_nodes: int = GP_CONFIG["max_nodes"],
    mutation_subtree_depth: int = 3,
) -> Node:
    """
    Subtree mutation: replace a randomly selected node with a fresh random tree.

    If the result violates depth or node-count limits, the original tree is
    returned unchanged.

    Args:
        tree                  : Input expression tree (will be deep-copied first).
        builder               : ExpressionBuilder for generating random subtrees.
        rng                   : Seeded random generator.
        max_depth             : Maximum allowed depth after mutation.
        max_nodes             : Maximum allowed node count after mutation.
        mutation_subtree_depth: Maximum depth of the new random subtree.

    Returns:
        Mutated expression tree.
    """
    mutant = tree.clone()
    nodes  = _all_nodes(mutant)

    # Pick mutation point
    target, parent, idx = nodes[int(rng.integers(0, len(nodes)))]

    # Generate replacement subtree
    new_subtree = builder.build_grow(mutation_subtree_depth)

    if parent is not None:
        parent.children[idx] = new_subtree
    else:
        mutant = new_subtree

    # Validity check
    if mutant.depth() > max_depth or mutant.node_count() > max_nodes:
        mutant = tree.clone()

    return mutant


# ─────────────────────────────────────────────────────────────────────────────
# Main evolution loop
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPResult:
    """Container for the result of a GP run."""

    best_expression:   Node
    best_fitness:      float
    fitness_history:   List[float]        # Best fitness per generation
    avg_fitness_history: List[float]      # Average fitness per generation
    complexity_history:  List[int]        # Best-tree node count per generation
    expression_string:   str
    node_count:          int
    seed:                int
    config:              Dict[str, Any]
    val_mse:             float            # MSE on validation set (raw, no penalty)
    val_r2:              float            # R² on validation set

    def summary(self) -> str:
        lines = [
            "──── GP Evolution Summary ────",
            f"Generations     : {len(self.fitness_history)}",
            f"Best fitness    : {self.best_fitness:.6f}",
            f"Val MSE         : {self.val_mse:.4f}",
            f"Val R²          : {self.val_r2:.4f}",
            f"Tree nodes      : {self.node_count}",
            f"Expression      : {self.expression_string}",
        ]
        return "\n".join(lines)


def evolve_expressions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    config:  Dict[str, Any] = GP_CONFIG,
    seed:    int = 42,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> GPResult:
    """
    Run the GP evolution loop and return the best discovered expression.

    Data contract
    ─────────────
    • X_train, X_val : shape (N, 4) and (M, 4).  Feature order:
                       [sharpe, drawdown, implied_vol, hist_vol].
    • y_train, y_val : shape (N,) or (N,1).  Risk score in [0, 100].
    • Test data is NEVER passed to this function.

    Fitness methodology
    ───────────────────
    During evolution each candidate is scored by compute_fitness_combined()
    which blends training MSE (70%) and validation MSE (30%) so that:
      • Training data is the primary driver of the search,
      • Validation data guards against overfitting / expression memorisation,
      • The test split is never seen by the GP engine.

    For final model selection and the reported fitness_history the pure
    validation fitness (compute_fitness on X_val only) is used, so the
    reported metrics are uncontaminated by the training objective.

    Args:
        X_train          : Training feature matrix.
        y_train          : Training target vector.
        X_val            : Validation feature matrix.
        y_val            : Validation target vector.
        config           : GP hyper-parameter dict (defaults to GP_CONFIG).
        seed             : Random seed for full reproducibility.
        progress_callback: Optional callable(gen, total_gens, best_val_fitness).

    Returns:
        GPResult dataclass with the best expression and all diagnostics.
    """
    rng     = np.random.default_rng(seed)
    random.seed(int(rng.integers(0, 2**31)))   # stdlib random for any helpers

    merged_config = dict(GP_CONFIG)
    if config:
        merged_config.update(config)
    config = merged_config

    builder = ExpressionBuilder(rng=rng, config=config)

    pop_size       = int(config["population_size"])
    generations    = int(config["generations"])
    cx_rate        = float(config["crossover_rate"])
    mut_rate       = float(config["mutation_rate"])
    elite_count    = int(config["elite_count"])
    tourn_k        = int(config["tournament_k"])
    alpha          = float(config["complexity_alpha"])
    max_depth      = int(config["max_depth"])
    max_nodes      = int(config["max_nodes"])

    y_val_arr   = np.asarray(y_val,   dtype=float).ravel()
    y_train_arr = np.asarray(y_train, dtype=float).ravel()
    X_val_arr   = np.asarray(X_val,   dtype=float)
    X_train_arr = np.asarray(X_train, dtype=float)

    # ── Initialisation ──
    # Evolution uses the combined train+val fitness; model-selection tracking
    # uses the pure val fitness so reported metrics are uncontaminated.
    population    = builder.ramped_half_and_half(pop_size)
    evo_fitnesses = [
        compute_fitness_combined(t, X_train_arr, y_train_arr, X_val_arr, y_val_arr, alpha)
        for t in population
    ]
    val_fitnesses = [compute_fitness(t, X_val_arr, y_val_arr, alpha) for t in population]

    fitness_history:     List[float] = []
    avg_fitness_history: List[float] = []
    complexity_history:  List[int]   = []

    best_val_idx  = int(np.argmax(val_fitnesses))
    best_tree     = population[best_val_idx].clone()
    best_val_fit  = val_fitnesses[best_val_idx]

    for gen in range(1, generations + 1):
        # ── Elite preservation (by combined evolutionary fitness) ──
        elite_indices = sorted(range(pop_size), key=lambda i: evo_fitnesses[i], reverse=True)
        elites = [population[i].clone() for i in elite_indices[:elite_count]]

        # ── Generate offspring ──
        offspring: List[Node] = list(elites)

        while len(offspring) < pop_size:
            roll = rng.random()

            if roll < cx_rate and len(population) >= 2:
                # Crossover
                p1 = tournament_select(population, evo_fitnesses, tourn_k, rng)
                p2 = tournament_select(population, evo_fitnesses, tourn_k, rng)
                c1, c2 = subtree_crossover(p1, p2, rng, max_depth, max_nodes)
                offspring.append(c1)
                if len(offspring) < pop_size:
                    offspring.append(c2)

            elif roll < cx_rate + mut_rate:
                # Mutation
                parent = tournament_select(population, evo_fitnesses, tourn_k, rng)
                child  = subtree_mutation(parent, builder, rng, max_depth, max_nodes)
                offspring.append(child)

            else:
                # Reproduction (copy)
                parent = tournament_select(population, evo_fitnesses, tourn_k, rng)
                offspring.append(parent.clone())

        population    = offspring[:pop_size]
        # Combined fitness drives selection; pure val fitness drives model selection
        evo_fitnesses = [
            compute_fitness_combined(t, X_train_arr, y_train_arr, X_val_arr, y_val_arr, alpha)
            for t in population
        ]
        val_fitnesses = [compute_fitness(t, X_val_arr, y_val_arr, alpha) for t in population]

        # ── Track best by val fitness (pure generalisation signal) ──
        gen_best_val_idx = int(np.argmax(val_fitnesses))
        gen_best_val_fit = val_fitnesses[gen_best_val_idx]

        if gen_best_val_fit > best_val_fit:
            best_val_fit = gen_best_val_fit
            best_tree    = population[gen_best_val_idx].clone()

        fitness_history.append(best_val_fit)
        avg_fitness_history.append(float(np.mean(val_fitnesses)))
        complexity_history.append(best_tree.node_count())

        if progress_callback is not None:
            progress_callback(gen, generations, best_val_fit)

    # ── Final validation metrics ──
    pred    = np.clip(best_tree.evaluate(X_val_arr), 0.0, 100.0)
    val_mse = float(np.mean((pred - y_val_arr) ** 2))
    ss_res  = float(np.sum((pred - y_val_arr) ** 2))
    ss_tot  = float(np.sum((y_val_arr - y_val_arr.mean()) ** 2))
    val_r2  = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

    return GPResult(
        best_expression     = best_tree,
        best_fitness        = best_val_fit,
        fitness_history     = fitness_history,
        avg_fitness_history = avg_fitness_history,
        complexity_history  = complexity_history,
        expression_string   = best_tree.to_string(),
        node_count          = best_tree.node_count(),
        seed                = seed,
        config              = dict(config),
        val_mse             = val_mse,
        val_r2              = val_r2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# High-level entry point (mirrors optimize_model in genetic_optimizer.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_gp(
    dataset: Dict[str, np.ndarray],
    config:  Dict[str, Any] = GP_CONFIG,
    seed:    int = 42,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Dict[str, Any]:
    """
    High-level interface: run GP on a dataset dict and return a serialisable result.

    The dataset must contain the keys produced by ``models.train.build_dataset``:
        X_vec   : (N, 4) float32 — per-day feature vector
        y_score : (N, 1) float32 — risk score in [0, 100]

    The function performs its own three-way split using the same
    ``get_train_val_test_indices`` logic as the GA, ensuring the GP validation
    slice is also disjoint from the held-out test set.

    Args:
        dataset          : Dict with at least 'X_vec' and 'y_score'.
        config           : GP hyper-parameter dict.
        seed             : Random seed.
        progress_callback: Optional callable(gen, total_gens, best_fitness).

    Returns:
        Serialisable dict with 'expression', 'fitness', 'metadata', etc.
    """
    from models.train import (
        get_train_val_test_indices,
        VALIDATION_SPLIT, TEST_SPLIT, TRAIN_SEED,
    )

    X = np.asarray(dataset["X_vec"],   dtype=float)
    y = np.asarray(dataset["y_score"], dtype=float).ravel()

    n = len(X)
    if "split_indices" in dataset:
        train_idx = dataset["split_indices"]["train"]
        val_idx   = dataset["split_indices"]["val"]
        test_idx  = dataset["split_indices"]["test"]
    else:
        train_idx, val_idx, test_idx = get_train_val_test_indices(
            n,
            validation_split=VALIDATION_SPLIT,
            test_split=TEST_SPLIT,
            seed=TRAIN_SEED,
        )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    # test_idx is never passed to evolve_expressions

    result = evolve_expressions(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config,
        seed=seed,
        progress_callback=progress_callback,
    )

    return {
        "expression":        result.expression_string,
        "fitness":           result.best_fitness,
        "val_mse":           result.val_mse,
        "val_r2":            result.val_r2,
        "node_count":        result.node_count,
        "fitness_history":   result.fitness_history,
        "avg_fitness_history": result.avg_fitness_history,
        "complexity_history":  result.complexity_history,
        "seed":              seed,
        "metadata": {
            "train_samples":  int(len(train_idx)),
            "val_samples":    int(len(val_idx)),
            "test_samples":   int(len(test_idx)),
            "total_samples":  int(n),
            "config":         dict(config),
            "feature_names":  FEATURE_NAMES,
            "fitness_definition": (
                "1 / (1 + blended_MSE + alpha*nodes); "
                "blended_MSE = 0.70*MSE_train + 0.30*MSE_val; "
                "model selection by pure val fitness"
            ),
        },
        "expression_tree": result.best_expression.to_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_gp_result(
    result_dict: Dict[str, Any],
    path: Path = GP_SAVE_PATH,
) -> None:
    """Save GP result dict (without the expression_tree Node) to JSON."""
    serialisable = {k: v for k, v in result_dict.items() if k != "expression_tree"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(serialisable, fh, indent=2, default=str)


def load_gp_result(path: Path = GP_SAVE_PATH) -> Optional[Dict[str, Any]]:
    """Load a previously saved GP result.  Returns None if the file is absent."""
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


# ────────────────────────────────────────────────────────────────────────────────
# Model integration helpers
# ────────────────────────────────────────────────────────────────────────────────

def load_expression_from_dict(tree_dict: Dict) -> Node:
    """Deserialise a Node from the ``expression_tree`` dict stored in a saved result."""
    return Node.from_dict(tree_dict)


def attach_gp_to_model(model, gp_result_dict: Dict[str, Any]) -> None:
    """
    Attach the best evolved expression from *gp_result_dict* to *model*.

    After this call ``model.gp_expression_string`` returns the expression and
    the GP signal genuinely influences every forward pass / predict_risk call.

    Args:
        model          : DeepNeuroFuzzyRiskModel instance.
        gp_result_dict : Dict returned by run_gp() — must contain
                         ``'expression_tree'`` key (a Node.to_dict() payload).
    """
    tree_dict = gp_result_dict.get("expression_tree")
    if tree_dict is None:
        raise ValueError(
            "gp_result_dict does not contain 'expression_tree'. "
            "Pass the dict returned directly by run_gp(), not the JSON-loaded version."
        )
    node = load_expression_from_dict(tree_dict)
    model.set_gp_expression(node)


def save_expression_tree(tree_dict: Dict, path: Path) -> None:
    """Save only the expression-tree dict (for model reload without full GP result)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(tree_dict, fh, indent=2)


def load_expression_tree(path: Path) -> Optional[Node]:
    """Load and reconstruct a Node from a saved expression-tree JSON. Returns None if absent."""
    if not path.exists():
        return None
    with open(path) as fh:
        return Node.from_dict(json.load(fh))
