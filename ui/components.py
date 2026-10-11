"""
Reusable presentational components (headers, cards, gauges, pills, footer).
Restricted styling palette: Standard Default and Blue accents only.
"""

from typing import Dict, List

import plotly.graph_objects as go
import streamlit as st

from config.settings import RISK_CATEGORIES

# Blue-only tint ramp used to communicate risk intensity.
RISK_TINT = {
    "Low": "#93c5fd",
    "Moderate": "#3b82f6",
    "High": "#1d4ed8",
    "Extreme": "#172554",
}

RISK_VALUE_CLASS = {
    "Low": "is-low",
    "Moderate": "is-mid",
    "High": "is-high",
    "Extreme": "is-extreme",
}

RISK_BLURB = {
    "Low": "Risk-adjusted returns look healthy; drawdown exposure is contained.",
    "Moderate": "Balanced profile — normal monitoring of volatility and drawdowns is advised.",
    "High": "Elevated tail risk. Tighten position sizing and review hedging.",
    "Extreme": "Severe risk regime. Drawdown and volatility inputs are both stressed.",
}


# --------------------------------------------------------------------------- #
# Layout helpers                                                               #
# --------------------------------------------------------------------------- #
def page_header(eyebrow: str, title: str, description: str = "") -> None:
    html = f'<div class="srp-hero"><div class="srp-hero-eyebrow">{eyebrow}</div><h1>{title}</h1>'
    if description:
        html += f"<p>{description}</p>"
    html += '<div class="srp-rule"></div></div>'
    st.markdown(html, unsafe_allow_html=True)


def section_title(title: str, note: str = "") -> None:
    html = f'<div class="srp-section-title">{title}</div>'
    if note:
        html += f'<p class="srp-section-note">{note}</p>'
    st.markdown(html, unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f'<div class="srp-note">{text}</div>', unsafe_allow_html=True)


MODEL_STATUS_TRAINED = (
    "<b>Model status — trained.</b> The risk head is fitted on "
    "{universe} tickers of real market history, so the score responds to the "
    "inputs you provide. Use <b>Model Training</b> to refit it."
)

MODEL_STATUS_UNTRAINED = (
    "<b>Model status — untrained.</b> The BiLSTM fusion head is randomly "
    "initialised, so the headline score sits near a constant until it is "
    "fitted. Open <b>Model Training</b> and press <b>Train model</b>; the "
    "fuzzy membership degrees and rule firings below are already correct."
)


def model_status_note() -> str:
    """Returns the caveat matching the current training state."""
    from ui.services import model_training_status

    try:
        status = model_training_status()
    except Exception:
        return MODEL_STATUS_UNTRAINED

    if status.get("trained"):
        count = len(status.get("universe", []))
        return MODEL_STATUS_TRAINED.format(universe=count)

    return MODEL_STATUS_UNTRAINED


def subtle_note(text: str) -> None:
    """Renders a low-emphasis inline note (used for model caveats)."""
    st.markdown(f'<div class="srp-hint">{text}</div>', unsafe_allow_html=True)


def info_card(eyebrow: str, value: str, hint: str = "", value_class: str = "") -> str:
    """Returns HTML for a single stat card."""
    return (
        '<div class="srp-card">'
        f'<div class="srp-card-eyebrow">{eyebrow}</div>'
        f'<div class="srp-card-value {value_class}">{value}</div>'
        f'<div class="srp-card-hint">{hint}</div>'
        "</div>"
    )


def render_cards(cards: List[Dict[str, str]], columns: int = 3) -> None:
    """
    Renders a responsive grid of stat cards.

    Uses a CSS grid rather than nested ``st.columns`` so cards keep a usable
    width inside any container. The column count is reduced by media queries in
    ``styles.py`` as the viewport narrows.
    """
    cells = "".join(
        info_card(c["eyebrow"], c["value"], c.get("hint", ""), c.get("class", ""))
        for c in cards
    )
    st.markdown(
        f'<div class="srp-grid" data-cols="{columns}" '
        f'style="grid-template-columns: repeat({columns}, minmax(0, 1fr));">'
        f"{cells}</div>",
        unsafe_allow_html=True,
    )


def fuzzy_pills(fuzzy_labels: Dict[str, str]) -> None:
    """Renders the linguistic term assigned to each metric as inline pills."""
    from ui.services import FEATURE_LABELS

    pills = []
    for key, label in FEATURE_LABELS.items():
        term = fuzzy_labels.get(key, "N/A")
        pills.append(
            f'<span class="srp-pill"><span class="srp-pill-dot"></span>'
            f'{label}: <b>{term}</b></span>'
        )
    st.markdown("".join(pills), unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        '<div class="srp-footer">'
        "<span>Stock Market Risk Prediction System · Soft Computing + Deep Learning</span>"
        "<span>Research tool only — not financial advice.</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def risk_badge(category: str) -> str:
    return RISK_TINT.get(category, "#3b82f6")


# --------------------------------------------------------------------------- #
# Charts                                                                       #
# --------------------------------------------------------------------------- #
def risk_gauge(score: float, category: str, height: int = 250) -> go.Figure:
    """0-100 risk gauge rendered in the blue-only tint ramp."""
    color = RISK_TINT.get(category, "#3b82f6")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=max(0.0, min(100.0, float(score))),
            number={"suffix": "", "font": {"size": 46, "color": "#0f172a"}},
            title={
                "text": f"<span style='font-size:15px;color:{color};font-weight:700'>"
                        f"{category.upper()} RISK</span><br>"
                        "<span style='font-size:11px;color:#64748b'>score out of 100</span>",
                "font": {"size": 14},
            },
            gauge={
                "axis": {"range": [0, 100], "showticklabels": False, "ticks": ""},
                "bar": {"color": color, "thickness": 0.32},
                "bgcolor": "#ffffff",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#eff6ff"},
                    {"range": [25, 50], "color": "#dbeafe"},
                    {"range": [50, 75], "color": "#bfdbfe"},
                    {"range": [75, 100], "color": "#93c5fd"},
                ],
                "threshold": {
                    "line": {"color": "#1d4ed8", "width": 3},
                    "thickness": 0.85,
                    "value": max(0.0, min(100.0, float(score))),
                },
            },
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=34, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif", color="#0f172a"),
    )
    return fig


def category_bar(probs: Dict[str, float]) -> go.Figure:
    """Horizontal distribution across the four risk categories."""
    fig = go.Figure(
        go.Bar(
            x=[float(probs.get(c, 0.0)) for c in RISK_CATEGORIES],
            y=RISK_CATEGORIES,
            orientation="h",
            marker_color=[RISK_TINT[c] for c in RISK_CATEGORIES],
            text=[f"{float(probs.get(c, 0.0)):.1%}" for c in RISK_CATEGORIES],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=40, t=18, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif", size=12, color="#0f172a"),
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1.18]),
        yaxis=dict(automargin=True, categoryorder="array", categoryarray=RISK_CATEGORIES[::-1]),
    )
    return fig
