"""
Manual Input page: score an arbitrary set of risk values with live re-prediction
as the sliders move.
"""

from typing import Dict

import numpy as np
import streamlit as st

from ui import charts, services
from ui.components import (
    RISK_BLURB,
    callout,
    fuzzy_pills,
    model_status_note,
    page_header,
    render_cards,
    risk_gauge,
    section_title,
    subtle_note,
)

# Sliders are expressed in friendly display units; the model consumes decimals.
FIELDS = [
    {
        "key": "sharpe",
        "label": "Sharpe Ratio",
        "min": -2.0, "max": 4.0, "step": 0.05, "default": 1.20,
        "help": "Annualised excess return divided by annualised volatility (Rf = 4%).",
        "hint": "Above 1.0 is generally considered good; below 0 means risk-free rate beats the asset.",
    },
    {
        "key": "drawdown",
        "label": "Max Drawdown",
        "min": 0.0, "max": 80.0, "step": 0.5, "default": 15.0,
        "suffix": "%",
        "help": "Largest peak-to-trough decline observed over the trading window.",
        "hint": "Deeper historical drawdowns push the fuzzy engine toward the High term.",
    },
    {
        "key": "implied_vol",
        "label": "Implied Volatility",
        "min": 5.0, "max": 100.0, "step": 0.5, "default": 25.0,
        "suffix": "%",
        "help": "Garman-Klass option-implied volatility proxy.",
        "hint": "Pricing-derived uncertainty for the next horizon.",
    },
    {
        "key": "hist_vol",
        "label": "Historical Volatility",
        "min": 5.0, "max": 100.0, "step": 0.5, "default": 20.0,
        "suffix": "%",
        "help": "Annualised 21-day standard deviation of log returns.",
        "hint": "Realised volatility actually observed in the series.",
    },
]

PRESETS = {
    "Defensive": {"sharpe": 2.00, "drawdown": 5.0, "implied_vol": 12.0, "hist_vol": 10.0},
    "Balanced": {"sharpe": 1.20, "drawdown": 15.0, "implied_vol": 25.0, "hist_vol": 20.0},
    "Stressed": {"sharpe": 0.10, "drawdown": 40.0, "implied_vol": 55.0, "hist_vol": 50.0},
    "Crisis": {"sharpe": -1.20, "drawdown": 65.0, "implied_vol": 85.0, "hist_vol": 75.0},
}


def _apply_preset(name: str) -> None:
    for key, value in PRESETS[name].items():
        st.session_state[f"mi_{key}"] = value


def _reset_defaults() -> None:
    for field in FIELDS:
        st.session_state[f"mi_{field['key']}"] = field["default"]


def _card_row(values: Dict[str, float]) -> list:
    suffixes = {"sharpe": "", "drawdown": "%", "implied_vol": "%", "hist_vol": "%"}
    cards = []
    for field in FIELDS:
        key = field["key"]
        cards.append({
            "eyebrow": field["label"],
            "value": f"{values[key]:.2f}{suffixes[key]}",
            "hint": field["hint"],
        })
    return cards


def render() -> None:
    page_header(
        "Manual input",
        "Custom Risk Scorer",
        "Have your own numbers? Enter a Sharpe ratio, drawdown and volatility pair "
        "to score them with the same ANFIS + BiLSTM pipeline used for live tickers. "
        "The prediction updates on every slider move.",
    )

    # ------------------------------------------------------------------ #
    # Presets                                                             #
    # ------------------------------------------------------------------ #
    section_title("Quick presets",
                  "Jump to a representative market regime, then fine-tune the sliders.")
    preset_cols = st.columns(len(PRESETS), gap="small")
    for col, name in zip(preset_cols, PRESETS):
        with col:
            st.button(name, key=f"preset_{name}", width="stretch",
                      on_click=_apply_preset, args=(name,),
                      help=f"Load the '{name}' profile into the sliders.")

    st.write("")

    # ------------------------------------------------------------------ #
    # Inputs                                                              #
    # ------------------------------------------------------------------ #
    section_title("Risk inputs",
                  "Ranges mirror the feature bounds the fuzzy layer was configured with.")

    grid = st.columns(2, gap="large")
    values: Dict[str, float] = {}

    # Seed defaults on first visit so session-state driven presets can take over.
    for field in FIELDS:
        st.session_state.setdefault(f"mi_{field['key']}", field["default"])

    for idx, field in enumerate(FIELDS):
        with grid[idx % 2]:
            widget_key = f"mi_{field['key']}"
            values[field["key"]] = st.slider(
                field["label"],
                min_value=field["min"],
                max_value=field["max"],
                step=field["step"],
                key=widget_key,
                help=field["help"],
            )

    st.button("Reset to defaults", key="mi_reset", on_click=_reset_defaults)

    # ------------------------------------------------------------------ #
    # Prediction                                                          #
    # ------------------------------------------------------------------ #
    vector = np.array([
        values["sharpe"],
        values["drawdown"] / 100.0,
        values["implied_vol"] / 100.0,
        values["hist_vol"] / 100.0,
    ], dtype=np.float32)

    with st.spinner("Scoring…"):
        result = services.predict_risk(vector)

    score = float(result["score"])
    category = str(result["category"])

    st.write("")
    section_title("Prediction",
                  "Output of the hybrid Deep Neuro-Fuzzy risk network for these inputs.")

    gauge_col, verdict_col = st.columns([3, 8], gap="large")
    with gauge_col:
        st.plotly_chart(risk_gauge(score, category), theme=None, key="mi_gauge")
    with verdict_col:
        st.markdown(
            '<div class="srp-verdict">'
            f'<div class="srp-card-eyebrow">Assessment</div>'
            f'<div class="srp-verdict-score">{score:.2f}'
            '<span class="srp-verdict-max">/100</span></div>'
            f'<div class="srp-verdict-cat">{category.upper()} RISK</div>'
            f'<div class="srp-verdict-text">{RISK_BLURB.get(category, "")}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        callout(
            f"Forecast next-horizon (10-day) maximum drawdown: "
            f"<b>{float(result['predicted_mdd']) * 100:.2f}%</b> (model projection over 10 trading days) · "
            f"headline category derived from the fusion network's classification head."
        )
        subtle_note(model_status_note())

    st.write("")
    fuzzy_pills(result["fuzzy_labels"])

    st.divider()
    section_title("Your inputs",
                  "The exact values fed into the fuzzy layer.")
    render_cards(_card_row(values), columns=2)

    # ------------------------------------------------------------------ #
    # Explainability                                                      #
    # ------------------------------------------------------------------ #
    st.divider()
    section_title("Soft-computing explainability",
                  "Where your inputs land on the Gaussian membership curves and which "
                  "TSK rules dominate the fused output.")

    degrees = services.membership_degrees(vector)
    bounds = services.get_extractor().feature_bounds
    from config.settings import FUZZY_INPUT_SPECS

    current = {
        "sharpe": values["sharpe"],
        "drawdown": values["drawdown"] / 100.0,
        "implied_vol": values["implied_vol"] / 100.0,
        "hist_vol": values["hist_vol"] / 100.0,
    }

    st.plotly_chart(
        charts.membership_chart(
            feature_keys=list(services.FEATURE_KEYS),
            feature_labels=[services.FEATURE_LABELS[k] for k in services.FEATURE_KEYS],
            bounds=bounds,
            specs=FUZZY_INPUT_SPECS,
            current=current,
            degrees=degrees,
        ),
        theme=None, key="mi_membership",
    )

    strengths = services.rule_firing_strengths(vector)
    catalog = services.rule_catalog()
    top_idx = int(np.argmax(strengths))
    top = catalog[top_idx]

    callout(
        f"<b>Dominant rule R{top['id']}</b> contributes "
        f"<b>{strengths[top_idx] * 100:.2f}%</b> of the fuzzy output — "
        f"“{services.describe_rule(top)}.”"
    )

    st.plotly_chart(
        charts.rule_firing_chart(catalog, strengths),
        theme=None, key="mi_rules",
    )

    st.caption(
        f"Category class probabilities — "
        + ", ".join(
            f"{k}: {v:.1%}"
            for k, v in services.class_probabilities(vector).items()
        )
    )
