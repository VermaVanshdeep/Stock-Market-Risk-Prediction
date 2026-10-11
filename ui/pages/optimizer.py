"""
Genetic Algorithm optimizer page: evolve fuzzy membership parameters and feature
weights with a live progress bar and convergence diagnostics.
"""

import numpy as np
import pandas as pd
import streamlit as st

from ui import charts, services
from ui.components import callout, page_header, render_cards, section_title

FEATURE_NAMES = ["Sharpe Ratio", "Max Drawdown", "Implied Vol", "Historical Vol"]
TERM_NAMES = ["Low", "Medium", "High"]


def _chromosome_table(chromosome: np.ndarray) -> pd.DataFrame:
    means = np.asarray(chromosome[0:12], dtype=float).reshape(4, 3)
    spreads = np.clip(np.asarray(chromosome[12:24], dtype=float), 0.01, 1.0).reshape(4, 3)

    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        row = {"Feature": name}
        for j, term in enumerate(TERM_NAMES):
            row[f"Center μ · {term}"] = round(float(means[i, j]), 4)
        for j, term in enumerate(TERM_NAMES):
            row[f"Spread σ · {term}"] = round(float(spreads[i, j]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def render() -> None:
    page_header(
        "Evolutionary optimisation",
        "Genetic Algorithm Optimizer",
        "Evolve the 28-gene chromosome — 12 membership centres, 12 spreads and 4 feature "
        "importance weights — using tournament selection, simulated binary crossover and "
        "Gaussian mutation.",
    )

    # ------------------------------------------------------------------ #
    # Hyper-parameters                                                    #
    # ------------------------------------------------------------------ #
    section_title("Hyper-parameters", "Configure the search, then start the run.")

    cols = st.columns([1, 1, 1, 1.6], gap="medium")
    with cols[0]:
        pop_size = st.slider("Population size", 4, 120, 20, step=2,
                             help="Number of candidate chromosomes per generation.")
    with cols[1]:
        generations = st.slider("Generations", 1, 80, 15, step=1,
                                help="Evolutionary iterations to perform.")
    with cols[2]:
        seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42,
                               help="Fixes the synthetic validation benchmark for reproducible runs.")
    with cols[3]:
        st.markdown("")
        run = st.button("Run evolution", type="primary", width="stretch",
                        help="Evaluates population × generations chromosomes in total.")

    total_evals = pop_size * generations
    st.caption(
        f"Approximately **{total_evals:,}** chromosome evaluations · "
        f"fitness = 1 / (1 + MSE) where MSE compares the weighted risk proxy against "
        f"the benchmark labels."
    )

    if run:
        st.session_state.pop("ga_result", None)
        bar = st.progress(0.0, text="Preparing initial population…")

        def _on_progress(gen: int, total: int, best: float) -> None:
            bar.progress(gen / max(total, 1),
                         text=f"Generation {gen}/{total} · best fitness {best:.6f}")

        with st.spinner("Evolving the population…"):
            try:
                result = services.run_genetic_algorithm(
                    population_size=pop_size,
                    generations=generations,
                    seed=int(seed),
                    progress_callback=_on_progress,
                )
            except Exception as exc:
                st.error(f"Optimization failed — {exc}")
                result = None

        if result is not None:
            st.session_state["ga_result"] = result
            bar.progress(1.0, text="Evolution complete.")

    result = st.session_state.get("ga_result")
    if not result:
        callout(
            "No run yet — press <b>Run evolution</b> above to start the evolutionary "
            "search. Results are kept here while you adjust other parts of the app."
        )
        return

    history = result["history"]
    chromosome = np.asarray(result["chromosome"], dtype=float)
    weights = chromosome[24:28]

    best = history[-1] if history else 0.0
    first = history[0] if history else 0.0
    improvement = (best - first) if first > 0 else best

    st.divider()
    section_title(
        "Optimization results",
        f"Last run: population {result['population_size']}, "
        f"{result['generations']} generations, seed {result['seed']}.",
    )

    render_cards(
        [
            {
                "eyebrow": "Best fitness",
                "value": f"{best:.6f}",
                "hint": "1 / (1 + MSE) of the best chromosome found.",
            },
            {
                "eyebrow": "Fitness gain",
                "value": f"+{improvement:.6f}",
                "hint": "Improvement over the best individual of generation 1.",
            },
            {
                "eyebrow": "Evaluations",
                "value": f"{result['population_size'] * result['generations']:,}",
                "hint": "Total chromosome fitness evaluations performed.",
            },
        ],
        columns=3,
    )

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.plotly_chart(charts.fitness_chart(history), theme=None, key="ga_fitness")
    with right:
        st.plotly_chart(charts.feature_weights_chart(weights), theme=None, key="ga_weights")

    callout(
        "<b>Interpretation.</b> The fitness curve flattening means the population has "
        "converged — further generations buy little. The weights chart shows how much "
        "influence the optimizer assigned to each of the four antecedent metrics."
    )

    tab_params, tab_vector = st.tabs(["Membership parameters", "Raw chromosome"])
    with tab_params:
        st.dataframe(_chromosome_table(chromosome), hide_index=True)
        st.caption(
            "Centres (μ) and spreads (σ) of the Gaussian membership functions after "
            "evolution, reshaped as 4 features × 3 linguistic terms."
        )
    with tab_vector:
        st.dataframe(
            pd.DataFrame(
                {"gene": np.arange(len(chromosome)), "value": np.round(chromosome, 6)},
            ),
            hide_index=True,
        )
        st.caption(
            "Genes 0–11: centres · genes 12–23: spreads · genes 24–27: feature weights."
        )

    st.divider()
    st.download_button(
        "Export optimized chromosome (CSV)",
        data=pd.DataFrame({
            "gene": np.arange(len(chromosome)),
            "value": chromosome,
        }).to_csv(index=False).encode("utf-8"),
        file_name="ga_optimized_chromosome.csv",
        mime="text/csv",
    )
