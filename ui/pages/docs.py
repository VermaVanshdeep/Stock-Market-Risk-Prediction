"""
Documentation page: renders ABOUT.md / README.md alongside quick-start commands
and the current engine status.
"""

import streamlit as st

from ui import services
from ui.components import callout, page_header, render_cards, section_title

COMMANDS = [
    ("Web interface (this app)", "streamlit run app.py"),
    ("Interactive CLI menu", "python main.py"),
    ("Automated test suite", "python test_runner.py"),
    ("PDF research report", "python generate_pdf_report.py"),
    ("Install dependencies", "pip install -r requirements.txt"),
]


def render() -> None:
    page_header(
        "Reference",
        "Project Documentation",
        "Technical implementation manual for the hybrid Soft Computing + Deep Learning "
        "risk prediction system — concepts, formulas, file map and run instructions.",
    )

    status = services.engine_status()
    render_cards(
        [{"eyebrow": s["name"], "value": s["value"], "hint": "Detected at runtime"}
         for s in status],
        columns=3,
    )

    st.divider()
    section_title("Quick start", "Everything needed to run the project locally.")

    for label, command in COMMANDS:
        st.markdown(f"**{label}**")
        st.code(command, language="bash")

    st.divider()
    tab_about, tab_readme, tab_data, tab_map = st.tabs(
        ["Technical manual", "README", "Data sources", "Repository map"]
    )

    with tab_about:
        st.markdown(services.load_markdown("ABOUT.md"))

    with tab_readme:
        st.markdown(services.load_markdown("README.md"))

    with tab_data:
        st.markdown("#### Yahoo Finance")
        st.markdown(
            "Listed tickers are downloaded through **yfinance**. If the network is "
            "unavailable the loader automatically switches to the built-in Merton Jump "
            "Diffusion simulator and the UI clearly labels the data as synthetic."
        )

        st.markdown("#### Google Sheets")
        st.markdown(
            "Any custom price series can be analysed by publishing it as a Google Sheet "
            "and pasting the link into **Risk Dashboard → Google Sheets**. The app reads "
            "the sheet's CSV export directly — no API key, no login, no local upload."
        )
        st.markdown(
            "1. Create a sheet with one row per trading day.\n"
            "2. Name the header columns; only **Date** and one of "
            "*Close / Adj Close / Price / Value* are required.\n"
            "3. Share it: **File › Share › Anyone with the link › Viewer**.\n"
            "4. Paste either the full URL, the `export?format=csv` URL, or just the "
            "sheet ID from the URL.\n"
            "5. Press **Load sheet**."
        )
        st.markdown(
            "`Open`, `High`, `Low` and `Volume` columns are optional — when omitted the "
            "app derives them from Close and tells you so. At least 30 usable rows are "
            "required so the 21-day volatility and drawdown windows can be computed."
        )
        st.markdown("**Example sheet layout**")
        st.code(
            "Date,Open,High,Low,Close,Volume\n"
            "2024-01-02,185.64,185.88,183.43,184.25,58414500\n"
            "2024-01-03,182.16,183.09,180.88,180.88,64014500",
            language="csv",
        )
        st.caption(services.SHEET_HELP)

    with tab_map:
        callout(
            "<b>config/settings.py</b> thresholds &amp; hyperparameters · "
            "<b>data/data_loader.py</b> yfinance + synthetic fallback · "
            "<b>data/feature_extractor.py</b> feature vectors · "
            "<b>models/fuzzy_layer.py</b> TSK ANFIS layer · "
            "<b>models/deep_neuro_fuzzy_net.py</b> BiLSTM fusion network · "
            "<b>models/genetic_optimizer.py</b> evolutionary optimizer · "
            "<b>utils/financial_metrics.py</b> risk maths · "
            "<b>utils/visualizer.py</b> Rich terminal &amp; matplotlib output · "
            "<b>app.py</b> Streamlit web UI · "
            "<b>main.py</b> CLI menu · "
            "<b>test_runner.py</b> suite runner · "
            "<b>generate_pdf_report.py</b> report compiler."
        )
        st.caption(
            "The web UI reuses the exact same engine modules as the CLI — no duplicated "
            "business logic lives under ui/."
        )
