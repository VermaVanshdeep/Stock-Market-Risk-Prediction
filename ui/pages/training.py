"""
Model Training page: fit the risk head so the score actually tracks risk, and
show the evidence that it now responds to input.
"""

from typing import Dict

import streamlit as st

from ui import charts, services
from ui.components import callout, page_header, render_cards, section_title, subtle_note


def _status_cards(status: Dict[str, object], probe: Dict[str, float]) -> list:
    profiles = probe.get("profiles", {})
    return [
        {
            "eyebrow": "Model state",
            "value": "Trained" if status["trained"] else "Untrained",
            "hint": "Weights loaded from disk." if status["trained"]
                    else "No persisted weights yet.",
        },
        {
            "eyebrow": "Score range",
            "value": f"{probe['min']:.1f} – {probe['max']:.1f}",
            "hint": "Defensive profile through crisis profile.",
        },
        {
            "eyebrow": "Response spread",
            "value": f"{probe['spread']:.1f}",
            "hint": "Points between the calmest and most extreme regime.",
        },
    ]


def render() -> None:
    page_header(
        "Model training",
        "Train the Risk Head",
        "The network previously shipped with random weights, which made the risk "
        "score sit at a near-constant value. This page fits the three prediction "
        "heads on real market history and shows exactly how much the score now moves.",
    )

    status = services.model_training_status()

    if not status["torch"]:
        callout(
            "PyTorch is not installed, so the neural head cannot be trained. "
            "Install it with <code>pip install torch</code> — the NumPy fallback "
            "engine has no gradient-based training loop."
        )

    probe = services.training_probe() if status["torch"] else {
        "profiles": {}, "min": 0.0, "max": 0.0, "spread": 0.0
    }

    render_cards(_status_cards(status, probe), columns=3)

    if status["trained"]:
        callout(
            f"Trained weights are loaded from <code>{status['weights_path']}</code>. "
            f"The score now spans <b>{probe['spread']:.1f} points</b> between the "
            f"calmest and most extreme regimes, which is what the dashboard gauge reflects."
        )
    else:
        callout(
            "No trained weights found — the score is currently near-constant. "
            "Press <b>Train model</b> below to fit the heads (about two minutes "
            "the first time, then cached on disk)."
        )

    st.divider()
    section_title("Regime response",
                  "How the fitted model scores a fixed set of market regimes.")
    st.plotly_chart(
        charts.regime_response_chart(probe),
        theme=None, key="training_regime",
    )
    if probe.get("profiles"):
        rows = "\n".join(
            f"{name:10s} {data['score']:6.2f} / 100   {data['category']}"
            for name, data in probe["profiles"].items()
        )
        st.code(rows, language="text")

    st.divider()
    section_title("Train / retrain",
                  "Rebuilds the dataset from live market history and refits all heads.")

    controls = st.columns([1.2, 1.6, 1.6], gap="medium")
    with controls[0]:
        epochs = st.slider("Epochs", 1, 40, 10, step=1,
                            help="More epochs lowers training error but costs time.")
    with controls[1]:
        st.markdown("")
        train_clicked = st.button(
            "Train model", type="primary", width="stretch",
            help="Runs dataset construction, training and weight persistence.",
        )
    with controls[2]:
        st.markdown("")
        clear_clicked = st.button("Reset to untrained", width="stretch",
                                  help="Deletes the persisted weights file.")

    if clear_clicked:
        from models import train as trainer

        try:
            trainer.WEIGHTS_PATH.unlink(missing_ok=True)
            get_model_cached_clear()
            st.session_state.pop("training_history", None)
            st.success("Weights deleted — the next prediction falls back to an untrained head.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not delete the weights — {exc}")

    if train_clicked and status["torch"]:
        bar = st.progress(0.0, text="Preparing training data…")

        def _on_progress(epoch: int, total: int, stats: Dict[str, float]) -> None:
            bar.progress(
                epoch / max(total, 1),
                text=(
                    f"Epoch {epoch}/{total} · "
                    f"train MAE {stats.get('train_mae', 0):.2f} · "
                    f"val accuracy {stats.get('val_acc', 0):.1f}%"
                ),
            )

        with st.spinner("Building the dataset and fitting the heads…"):
            try:
                history = services.retrain(
                    epochs=int(epochs), progress_callback=_on_progress
                )
            except Exception as exc:
                st.error(f"Training failed — {exc}")
                history = None

        if history is not None:
            st.session_state["training_history"] = history
            get_model_cached_clear()
            bar.progress(1.0, text="Training complete.")
            st.success(
                f"Fitted on {history.get('samples', 0):,} samples across "
                f"{len(history.get('tickers', []))} tickers. "
                f"Final validation accuracy {history.get('val_acc', [0])[-1]:.1f}%."
            )

    history = st.session_state.get("training_history")
    if history:
        section_title("Training history",
                      "Mean absolute score error in risk-score points "
                      "(0 = perfect prediction).")
        st.plotly_chart(
            charts.training_curve(history),
            theme=None, key="training_curve",
        )

        render_cards(
            [
                {"eyebrow": "Samples", "value": f"{history.get('samples', 0):,}",
                 "hint": "Total supervised windows built from price history."},
                {"eyebrow": "Train / validation",
                 "value": f"{history.get('train_samples', 0):,} / {history.get('val_samples', 0):,}",
                 "hint": "Chronological-style split used for honest evaluation."},
                {"eyebrow": "Final val. MAE",
                 "value": f"{history.get('val_mae', [0])[-1]:.2f}",
                 "hint": "Average absolute error in the 0-100 risk score."},
            ],
            columns=3,
        )

    st.divider()
    subtle_note(
        "<b>How the targets are built.</b> For every 20-day window the engine looks "
        "10 days ahead, measures the maximum peak-to-trough drop, and scores risk as "
        "<code>future_drawdown x 200 + historical_volatility x 100</code>. The "
        "classification head learns Low / Moderate / High / Extreme buckets from the "
        "same target, and the drawdown head regresses the forward-looking drop directly."
    )

    st.caption(
        "Weights persist to <code>data_cache/risk_model_weights.pt</code>, so the CLI "
        "and every browser session share the same fitted model."
    )


def get_model_cached_clear() -> None:
    """Drops the cached model singleton so the next render reloads weights."""
    try:
        services.get_model.clear()
    except Exception:
        pass