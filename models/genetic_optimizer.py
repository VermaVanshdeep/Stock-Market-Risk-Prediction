"""
Soft Computing Core: Genetic Algorithm (GA) Evolutionary Optimizer.

Chromosome Encoding (24 genes)
────────────────────────────────────────────────────────────────
  Genes  0-11  : Gaussian MF centres  - means[4][3]  (4 features x 3 terms)
  Genes 12-23  : Gaussian MF spreads  - sigmas[4][3] (4 features x 3 terms)

Every gene maps to a real parameter in PyTorchTakagiSugenoFuzzyLayer.
Feature order : [Sharpe, Drawdown, ImpliedVol, HistVol]
Term   order  : [Low, Medium, High]  - enforced Low < Medium < High

Defects fixed in this version (audit 2026-10-19)
──────────────────────────────────────────────────
D1  Fitness oracle — evaluates the FULL end-to-end hybrid model (BiLSTM +
    dense encoder + fuzzy layer + fusion net + risk-score head) rather than
    the fuzzy sub-layer alone.  Sequence inputs are supplied where required.
    fitness = 1 / (1 + MSE(predicted_risk_score, y_val))
    Both operands are in [0, 100] score-point units.

D2  MF centre ordering — chromosome_to_params() sorts Low<Medium<High and
    applies the same permutation to sigmas, preserving the gene pairing.
    Repair is applied consistently after decode, crossover, and mutation.

D3  GA init bounds — feature-specific bounds are read from the project's
    authoritative configuration (train._BOUNDS / feature_extractor.feature_bounds)
    so the search space always covers the full valid input domain.

D4  Reproducibility — initialize_population() accepts a seed and passes it to
    np.random.default_rng(seed).  All random operators inside optimize() use the
    same seeded Generator; np.random.seed() is also set for legacy callers.

D5  Validation independence (extended) — optimize_model() uses
    get_train_val_test_indices() to produce a GA validation set that is
    provably disjoint from BOTH the NN training rows AND the held-out final
    test set.  The last TEST_SPLIT fraction of rows is never seen by the GA.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console

from config.settings import DATA_DIR, GA_CONFIG

console = Console()

# Path where the best chromosome and metadata are persisted.
GA_PARAMS_PATH: Path = Path(DATA_DIR) / "ga_optimized_params.json"

# Chromosome length: 12 means + 12 sigmas.
CHROMOSOME_LEN: int = 24

# Feature order must match DeepNeuroFuzzyRiskModel / build_dataset / fuzzy_layer.
_FEATURE_ORDER = ["sharpe", "drawdown", "implied_vol", "hist_vol"]

# Authoritative feature bounds (mirrors train._BOUNDS and RiskFeatureExtractor).
# Read lazily to avoid circular imports at module load time.
def _get_feature_bounds() -> Dict[str, Tuple[float, float]]:
    from models.train import _BOUNDS
    return _BOUNDS  # {"sharpe": (-2.0, 4.0), "drawdown": (0.0, 0.8), ...}


# ─────────────────────────────────────────────────────────────────────────────
# Chromosome <-> model parameter mapping  (Defects D2 + D3 addressed here)
# ─────────────────────────────────────────────────────────────────────────────

def _clip_means(means: np.ndarray, bounds: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Clip each row of means to its feature's valid domain."""
    m = means.copy()
    for i, key in enumerate(_FEATURE_ORDER):
        lo, hi = bounds[key]
        m[i] = np.clip(m[i], lo, hi)
    return m


def _sort_means_sigmas(means: np.ndarray, sigmas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort (means, sigmas) pairs within each feature row so that
    centre_Low < centre_Medium < centre_High.

    The sort permutation is applied identically to both arrays so that
    each centre stays paired with its spread (D2 fix).
    """
    m = means.copy()
    s = sigmas.copy()
    for i in range(m.shape[0]):
        idx = np.argsort(m[i])
        m[i] = m[i][idx]
        s[i] = s[i][idx]
    return m, s


def chromosome_to_params(chromosome: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode a 24-gene chromosome into (means, sigmas) arrays of shape (4, 3).

    Repairs applied (D2, D3):
    1. Sigmas clipped to [0.01, 2.0].
    2. Means clipped to each feature's valid domain.
    3. Within each feature, means (and corresponding sigmas) sorted Low->High.
    """
    if len(chromosome) != CHROMOSOME_LEN:
        raise ValueError(
            f"Chromosome must have exactly {CHROMOSOME_LEN} genes "
            f"(12 means + 12 sigmas); got {len(chromosome)}."
        )
    bounds = _get_feature_bounds()

    means  = np.asarray(chromosome[:12],  dtype=np.float32).reshape(4, 3)
    sigmas = np.clip(
        np.asarray(chromosome[12:24], dtype=np.float32).reshape(4, 3),
        0.01, 2.0,
    )
    means  = _clip_means(means, bounds)
    means, sigmas = _sort_means_sigmas(means, sigmas)
    return means, sigmas


def apply_chromosome(model, chromosome: np.ndarray) -> None:
    """
    Apply a 24-gene chromosome to the fuzzy engine inside *model*.

    Works for both the PyTorch and NumPy fallback variants.
    Constraints (ordering, domain, sigma positivity) are enforced via
    chromosome_to_params before writing.
    """
    means, sigmas = chromosome_to_params(chromosome)
    fuzzy = model.fuzzy_engine

    try:
        import torch
        with torch.no_grad():
            fuzzy.means.copy_(torch.tensor(means, dtype=torch.float32))
            fuzzy.sigmas.copy_(torch.tensor(sigmas, dtype=torch.float32))
    except Exception:
        fuzzy.means  = means.copy()
        fuzzy.sigmas = sigmas.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_ga_params(
    chromosome: np.ndarray,
    metadata: Dict,
    path: Path = GA_PARAMS_PATH,
) -> Path:
    """
    Save the best chromosome and optimisation metadata as JSON.

    The decoded (constrained, ordered) means and sigmas are saved alongside the
    raw chromosome so the file is self-documenting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    means, sigmas = chromosome_to_params(chromosome)
    payload = {
        "chromosome": [float(v) for v in chromosome],
        "chromosome_len": CHROMOSOME_LEN,
        "decoded_means":  means.tolist(),
        "decoded_sigmas": sigmas.tolist(),
        "feature_order":  _FEATURE_ORDER,
        **{str(k): v for k, v in metadata.items()},
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_ga_params(
    model,
    path: Path = GA_PARAMS_PATH,
) -> Optional[Dict]:
    """
    Load a saved chromosome from *path* and apply it to *model*.

    Returns the full metadata dict on success, or None if the file does not
    exist or is malformed.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        chromo = np.asarray(payload["chromosome"], dtype=np.float32)
        if len(chromo) != CHROMOSOME_LEN:
            return None
        apply_chromosome(model, chromo)
        return payload
    except Exception:
        return None


def ga_params_exist(path: Path = GA_PARAMS_PATH) -> bool:
    return path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Genetic Algorithm
# ─────────────────────────────────────────────────────────────────────────────

class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm optimizer for fuzzy membership-function parameters.

    Chromosome Encoding (24 float genes):
        Genes  0-11  -> MF centres  means[4][3]
        Genes 12-23  -> MF spreads  sigmas[4][3]  (clipped to [0.01, 2.0])

    Fitness (D1 fix):
        The full end-to-end hybrid model (BiLSTM + dense encoder + fuzzy layer
        + fusion net + risk-score head) is called for each candidate chromosome.
        fitness = 1 / (1 + MSE(predicted_risk_score, y_val))
        Both sides are in [0, 100] score-point units.

    Reproducibility (D4 fix):
        A single seeded np.random.default_rng is used for all random operations.
    """

    def __init__(
        self,
        population_size: int = GA_CONFIG["population_size"],
        generations: int = GA_CONFIG["generations"],
        mutation_rate: float = GA_CONFIG["mutation_rate"],
        crossover_rate: float = GA_CONFIG["crossover_rate"],
    ):
        self.pop_size      = population_size
        self.generations   = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.chromosome_len = CHROMOSOME_LEN

    # ---------------------------------------------------------------------- #
    # Population  (D3 + D4 fix)                                               #
    # ---------------------------------------------------------------------- #

    def initialize_population(self, rng: np.random.Generator) -> np.ndarray:
        """
        Draw an initial population with domain-aware random bounds.

        Bounds come from _BOUNDS (authoritative config) so the full feature
        domain is always covered.  rng must be a seeded Generator (D4 fix).

        Each individual is immediately repaired (clip + sort) so every member
        of the initial population satisfies all constraints.
        """
        bounds = _get_feature_bounds()
        pop = np.zeros((self.pop_size, self.chromosome_len), dtype=np.float32)

        keys = _FEATURE_ORDER  # ["sharpe", "drawdown", "implied_vol", "hist_vol"]
        for i, key in enumerate(keys):
            lo, hi = bounds[key]
            # Sample 3 centres uniformly over the full valid domain
            pop[:, i*3:(i+1)*3] = rng.uniform(lo, hi, (self.pop_size, 3))

        # Sigmas: positive, bounded
        pop[:, 12:24] = rng.uniform(0.02, 0.5, (self.pop_size, 12))

        # Repair ordering and domain in-place
        pop = self._repair_population(pop)
        return pop

    def _repair_population(self, pop: np.ndarray) -> np.ndarray:
        """Apply decode-and-re-encode repair to every individual in *pop*."""
        repaired = np.empty_like(pop)
        for i, ind in enumerate(pop):
            m, s = chromosome_to_params(ind)
            repaired[i, :12]  = m.ravel()
            repaired[i, 12:]  = s.ravel()
        return repaired

    def _repair_individual(self, chromosome: np.ndarray) -> np.ndarray:
        """Repair a single chromosome (ordering + domain + sigma clip)."""
        m, s = chromosome_to_params(chromosome)
        out = np.empty(CHROMOSOME_LEN, dtype=np.float32)
        out[:12] = m.ravel()
        out[12:] = s.ravel()
        return out

    # ---------------------------------------------------------------------- #
    # Fitness  (D1 fix: full model forward pass)                              #
    # ---------------------------------------------------------------------- #

    def compute_fitness(
        self,
        chromosome: np.ndarray,
        X_vec: np.ndarray,
        X_seq: np.ndarray,
        y_val: np.ndarray,
        model,
    ) -> float:
        """
        Evaluate one chromosome against the real model's end-to-end pipeline.

        Objective
        ---------
        fitness = 1 / (1 + MSE(risk_score_predicted, y_val))

        Both *risk_score_predicted* and *y_val* are in [0, 100] score-point
        units — the same scale the dashboard shows and the same target that
        train_model() minimises.

        Procedure
        ---------
        1. Snapshot current fuzzy parameters.
        2. Write candidate means and sigmas to the model's fuzzy_engine.
        3. Run the full forward pass in eval + no_grad mode:
               BiLSTM branch  (X_seq)
               Dense encoder  (X_vec)
               Fuzzy branch   (X_vec)
               Fusion net -> risk-score head  -> [0, 100]
        4. Restore original fuzzy parameters.
        5. Return 1 / (1 + MSE).

        Neural network weights (BiLSTM, dense, fusion, heads) are FIXED
        throughout GA evaluation — only means and sigmas change.
        """
        means, sigmas = chromosome_to_params(chromosome)
        fuzzy = model.fuzzy_engine

        try:
            import torch
            orig_m = fuzzy.means.data.clone()
            orig_s = fuzzy.sigmas.data.clone()
            try:
                with torch.no_grad():
                    fuzzy.means.copy_(torch.tensor(means, dtype=torch.float32))
                    fuzzy.sigmas.copy_(torch.tensor(sigmas, dtype=torch.float32))
                    model.eval()
                    x_v = torch.tensor(X_vec, dtype=torch.float32)
                    x_s = torch.tensor(X_seq, dtype=torch.float32)
                    risk_out, _, _, _ = model(x_v, x_s)
                    pred = risk_out.squeeze(-1).numpy()   # (N,) in [0, 100]
            finally:
                # Always restore — even if forward() raises
                with torch.no_grad():
                    fuzzy.means.copy_(orig_m)
                    fuzzy.sigmas.copy_(orig_s)
        except Exception:
            # NumPy fallback: same save/restore pattern
            orig_m_np = fuzzy.means.copy()
            orig_s_np = fuzzy.sigmas.copy()
            try:
                fuzzy.means  = means.copy()
                fuzzy.sigmas = sigmas.copy()
                risk_out, _, _, _ = model.forward(X_vec)
                pred = np.clip(np.asarray(risk_out).ravel(), 0.0, 100.0)
            finally:
                fuzzy.means  = orig_m_np
                fuzzy.sigmas = orig_s_np

        mse = float(np.mean((pred - y_val.ravel()) ** 2))
        return 1.0 / (1.0 + mse)

    # ---------------------------------------------------------------------- #
    # Evolutionary operators  (D4 fix: use rng parameter)                     #
    # ---------------------------------------------------------------------- #

    def selection(
        self,
        population: np.ndarray,
        fitness_scores: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Tournament selection (k=2)."""
        selected = []
        n = len(population)
        for _ in range(self.pop_size):
            i, j = rng.choice(n, 2, replace=False)
            winner = i if fitness_scores[i] >= fitness_scores[j] else j
            selected.append(population[winner].copy())
        return np.asarray(selected, dtype=np.float32)

    def crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover, followed by constraint repair."""
        if rng.random() < self.crossover_rate:
            pt = int(rng.integers(1, self.chromosome_len - 1))
            c1 = np.concatenate([p1[:pt], p2[pt:]])
            c2 = np.concatenate([p2[:pt], p1[pt:]])
        else:
            c1, c2 = p1.copy(), p2.copy()
        # Repair ordering / domain after gene mixing
        return self._repair_individual(c1), self._repair_individual(c2)

    def mutate(
        self,
        chromosome: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Gaussian mutation with feature-proportional noise (D6 fix).

        Mean genes use sigma = 0.05 * feature_range so exploration scales to the
        domain size.  Sigma genes use a fixed smaller noise (0.02).
        After mutation, constraints are enforced by _repair_individual.
        """
        bounds = _get_feature_bounds()
        mutated = chromosome.copy()

        gene_idx = 0
        for i, key in enumerate(_FEATURE_ORDER):
            lo, hi = bounds[key]
            feature_range = hi - lo
            noise_std = 0.05 * feature_range  # proportional to domain
            for _ in range(3):                # 3 terms per feature
                if rng.random() < self.mutation_rate:
                    mutated[gene_idx] += rng.normal(0.0, noise_std)
                gene_idx += 1

        # Sigma genes: smaller fixed noise
        for idx in range(12, CHROMOSOME_LEN):
            if rng.random() < self.mutation_rate:
                mutated[idx] += rng.normal(0.0, 0.02)

        return self._repair_individual(mutated)

    # ---------------------------------------------------------------------- #
    # Main evolution loop  (D4 fix: single seeded Generator)                  #
    # ---------------------------------------------------------------------- #

    def optimize(
        self,
        X_vec: np.ndarray,
        X_seq: np.ndarray,
        y_val: np.ndarray,
        model,
        progress_callback: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Run the evolutionary loop.

        Parameters
        ----------
        X_vec : (N, 4) feature vectors [Sharpe, Drawdown, IV, HV]
        X_seq : (N, seq_len, 4) feature sequences for the BiLSTM branch
        y_val : (N,) ground-truth risk scores in [0, 100]
        model : DeepNeuroFuzzyRiskModel (trained, weights fixed)
        seed  : integer seed — two calls with the same seed produce identical
                chromosome and fitness histories (D4 fix)

        Returns
        -------
        (best_chromosome, fitness_history)
        """
        # One seeded Generator controls ALL random operations (D4 fix)
        rng = np.random.default_rng(seed)
        # Also seed legacy API for any external callers that use np.random.*
        if seed is not None:
            np.random.seed(seed)

        console.print("[bold cyan]Running GA optimisation of fuzzy MF parameters...[/bold cyan]")

        population = self.initialize_population(rng)
        fitness_history: List[float] = []
        best_chromosome: Optional[np.ndarray] = None
        best_fitness: float = -1.0

        for gen in range(self.generations):
            scores = np.array([
                self.compute_fitness(ind, X_vec, X_seq, y_val, model)
                for ind in population
            ])

            gen_best_idx = int(np.argmax(scores))
            if scores[gen_best_idx] > best_fitness:
                best_fitness = float(scores[gen_best_idx])
                best_chromosome = population[gen_best_idx].copy()

            fitness_history.append(best_fitness)

            if progress_callback is not None:
                try:
                    progress_callback(gen + 1, self.generations, best_fitness)
                except Exception:
                    pass

            selected = self.selection(population, scores, rng)
            next_gen = []
            for i in range(0, self.pop_size, 2):
                p1 = selected[i]
                p2 = selected[min(i + 1, self.pop_size - 1)]
                c1, c2 = self.crossover(p1, p2, rng)
                next_gen.append(self.mutate(c1, rng))
                next_gen.append(self.mutate(c2, rng))

            population = np.asarray(next_gen[: self.pop_size], dtype=np.float32)

        console.print(
            f"[bold green]GA complete · best fitness {best_fitness:.6f}[/bold green]"
        )
        return best_chromosome, fitness_history


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end model optimisation entry point
# ─────────────────────────────────────────────────────────────────────────────

def optimize_model(
    model,
    dataset: Dict,
    population_size: int = GA_CONFIG["population_size"],
    generations: int = GA_CONFIG["generations"],
    seed: int = 42,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    """
    Full GA optimisation of the fuzzy layer embedded in *model*.

    Validation split (D5 fix — extended)
    ─────────────────────────────────────
    Uses get_train_val_test_indices() to build a GA validation set that is
    provably disjoint from BOTH the neural-network training rows AND the
    held-out final test set.  The last TEST_SPLIT fraction of rows (by
    concatenation order) is excluded from all optimisation and selection.

    Fitness objective (D1 fix)
    ──────────────────────────
    fitness = 1 / (1 + MSE(model_risk_score, y_val))
    Both quantities are in [0, 100] risk-score units.
    """
    from models.train import (
        get_train_val_test_indices,
        VALIDATION_SPLIT, TEST_SPLIT, TRAIN_SEED,
    )

    X_vec = dataset["X_vec"]            # (N, 4)
    X_seq = dataset["X_seq"]            # (N, seq_len, 4)
    y     = dataset["y_score"].ravel()  # (N,)

    n = len(X_vec)

    # Three-way disjoint split — GA never sees test rows.
    if "split_indices" in dataset:
        train_idx = dataset["split_indices"]["train"]
        val_idx   = dataset["split_indices"]["val"]
        test_idx  = dataset["split_indices"]["test"]
        n_train   = int(len(train_idx))
    else:
        train_idx, val_idx, test_idx = get_train_val_test_indices(
            n,
            validation_split=VALIDATION_SPLIT,
            test_split=TEST_SPLIT,
            seed=TRAIN_SEED,
        )
        n_trainval = n - len(test_idx)
        n_train    = int(n_trainval - len(val_idx))

    X_val_vec = X_vec[val_idx]   # (M, 4)
    X_val_seq = X_seq[val_idx]   # (M, seq_len, 4)
    y_val     = y[val_idx]       # (M,)

    # Evolutionary training subset (prevents excessive execution times while maintaining diversity)
    max_ga_train = min(1024, len(train_idx))
    X_tr_ga_vec  = X_vec[train_idx[:max_ga_train]]
    X_tr_ga_seq  = X_seq[train_idx[:max_ga_train]]
    y_tr_ga      = y[train_idx[:max_ga_train]]

    console.print(
        f"[bold cyan]GA optimization: {len(X_tr_ga_vec)} training samples, "
        f"{len(val_idx)} validation samples (held-out test: {len(test_idx)} samples)[/bold cyan]"
    )

    # Baseline — full model forward pass on validation set with current fuzzy parameters
    baseline_fitness = _full_model_fitness(model, X_val_vec, X_val_seq, y_val)

    # Snapshot for rollback
    orig_means, orig_sigmas = _snapshot_params(model.fuzzy_engine)

    ga = GeneticAlgorithmOptimizer(
        population_size=population_size,
        generations=generations,
    )
    best_chromosome, fitness_history = ga.optimize(
        X_vec=X_tr_ga_vec,
        X_seq=X_tr_ga_seq,
        y_val=y_tr_ga,
        model=model,
        progress_callback=progress_callback,
        seed=seed,
    )

    # Test evolved chromosome strictly on validation set
    apply_chromosome(model, best_chromosome)
    evolved_fitness = _full_model_fitness(model, X_val_vec, X_val_seq, y_val)
    improved = evolved_fitness > baseline_fitness

    if improved:
        console.print(
            f"[bold green]GA-optimised parameters improved validation fitness "
            f"({baseline_fitness:.6f} -> {evolved_fitness:.6f}). Applied to model.[/bold green]"
        )
    else:
        _restore_params(model.fuzzy_engine, orig_means, orig_sigmas)
        console.print(
            "[yellow]GA did not improve validation fitness "
            f"(evolved {evolved_fitness:.6f} vs baseline {baseline_fitness:.6f}). "
            "Original parameters kept.[/yellow]"
        )

    metadata = {
        "seed":               seed,
        "population_size":    population_size,
        "generations":        generations,
        "baseline_fitness":   float(baseline_fitness),
        "best_fitness":       float(evolved_fitness),
        "improved":           bool(improved),
        "fitness_history":    [float(f) for f in fitness_history],
        "val_samples":        int(len(val_idx)),
        "train_samples":      int(n_train),
        "test_samples":       int(len(test_idx)),
        "total_samples":      int(n),
        "fitness_definition": "1 / (1 + MSE(model_risk_score_0_100, y_val_0_100))",
        "objective":          "full_end_to_end_model",
        "split_design":       "chronological_tail_test_shuffled_trainval",
        "feature_bounds":     {k: list(v) for k, v in _get_feature_bounds().items()},
    }

    if improved:
        save_ga_params(best_chromosome, metadata)

    return {
        "chromosome": best_chromosome,
        "history":    fitness_history,
        "metadata":   metadata,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _full_model_fitness(
    model,
    X_vec: np.ndarray,
    X_seq: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """
    1 / (1 + MSE) of the full model's risk score on the given validation slice.
    Both predicted and target scores are in [0, 100].
    """
    try:
        import torch
        model.eval()
        with torch.no_grad():
            risk_out, _, _, _ = model(
                torch.tensor(X_vec, dtype=torch.float32),
                torch.tensor(X_seq, dtype=torch.float32),
            )
            pred = risk_out.squeeze(-1).numpy()
    except Exception:
        risk_out, _, _, _ = model.forward(X_vec)
        pred = np.clip(np.asarray(risk_out).ravel(), 0.0, 100.0)
    mse = float(np.mean((pred - y_val.ravel()) ** 2))
    return 1.0 / (1.0 + mse)


def _snapshot_params(fuzzy_engine) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import torch
        return (
            fuzzy_engine.means.data.clone().numpy(),
            fuzzy_engine.sigmas.data.clone().numpy(),
        )
    except Exception:
        return fuzzy_engine.means.copy(), fuzzy_engine.sigmas.copy()


def _restore_params(fuzzy_engine, means: np.ndarray, sigmas: np.ndarray) -> None:
    try:
        import torch
        with torch.no_grad():
            fuzzy_engine.means.copy_(torch.tensor(means, dtype=torch.float32))
            fuzzy_engine.sigmas.copy_(torch.tensor(sigmas, dtype=torch.float32))
    except Exception:
        fuzzy_engine.means  = means.copy()
        fuzzy_engine.sigmas = sigmas.copy()
