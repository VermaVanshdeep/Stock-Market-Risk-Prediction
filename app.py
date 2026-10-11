"""
Stock Market Risk Prediction System — Streamlit Web Interface.

Run with:
    streamlit run app.py

The web UI is a presentation layer only: every metric, model, optimizer and report
is the same engine module used by the CLI (`python main.py`).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Stock Market Risk Prediction",
    page_icon=str(ROOT / "assets" / "logo.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import services  # noqa: E402
from ui.components import MODEL_STATUS_UNTRAINED, MODEL_STATUS_TRAINED, footer  # noqa: E402
from ui.pages import (
    dashboard,
    docs,
    manual_input,
    optimizer,
    reports,
    test_suite,
    training,
)
from ui.styles import inject_styles  # noqa: E402

NAVIGATION = {
    "Risk Dashboard": dashboard.render,
    "Manual Input": manual_input.render,
    "Model Training": training.render,
    "GA Optimizer": optimizer.render,
    "Test Suite": test_suite.render,
    "Reports": reports.render,
    "Documentation": docs.render,
}


def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div class="srp-brand">'
            '<div class="srp-brand-mark">SR</div>'
            "<div>"
            '<div class="srp-brand-name">Stock Market Risk</div>'
            '<div class="srp-brand-sub">Soft Computing · Deep Learning</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigate",
            list(NAVIGATION),
            label_visibility="collapsed",
            key="nav_page",
        )

        st.markdown('<div class="srp-rule"></div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="srp-card-eyebrow">Engine status</div>',
            unsafe_allow_html=True,
        )
        for item in services.engine_status():
            st.markdown(
                f'<span class="srp-pill"><span class="srp-pill-dot"></span>'
                f"{item['name']}: <b>{item['value']}</b></span>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.caption(
            "Prefer a terminal? The full menu is still available with "
            "`python main.py`."
        )

        with st.expander("Model status"):
            if services.model_is_trained():
                trained_on = len(services.model_training_status()["universe"])
                st.markdown(MODEL_STATUS_TRAINED.format(universe=trained_on),
                            unsafe_allow_html=True)
            else:
                st.markdown(MODEL_STATUS_UNTRAINED, unsafe_allow_html=True)
            st.caption(
                "Metrics, charts, the genetic algorithm, the test suite and the PDF "
                "report are all fully functional regardless."
            )

    return page


def main() -> None:
    inject_styles()
    page = _sidebar()
    NAVIGATION[page]()
    footer()


if __name__ == "__main__":
    main()
