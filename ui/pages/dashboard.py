"""
Risk Dashboard page: pick a company, fetch market data, run the hybrid model and
explore the result through interactive charts and fuzzy-rule explanations.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
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
    category_bar,
    subtle_note,
)

PERIODS = {"6 months": "6mo", "1 year": "1y", "2 years": "2y"}


def _metric_cards(metrics: Dict[str, float]) -> list:
    return [
        {
            "eyebrow": "Sharpe Ratio",
            "value": f"{metrics.get('sharpe', 0.0):.3f}",
            "hint": "Excess return per unit of total volatility (Rf = 4%).",
        },
        {
            "eyebrow": "Max Drawdown",
            "value": f"{metrics.get('drawdown', 0.0) * 100:.2f}%",
            "hint": "Largest peak-to-trough decline over the window.",
        },
        {
            "eyebrow": "Implied Volatility",
            "value": f"{metrics.get('implied_vol', 0.0) * 100:.2f}%",
            "hint": "Garman-Klass option-implied volatility proxy.",
        },
        {
            "eyebrow": "Historical Volatility",
            "value": f"{metrics.get('hist_vol', 0.0) * 100:.2f}%",
            "hint": "Annualised 21-day standard deviation of log returns.",
        },
        {
            "eyebrow": "Sortino Ratio",
            "value": f"{metrics.get('sortino', 0.0):.3f}",
            "hint": "Risk-adjusted return using downside deviation only.",
        },
        {
            "eyebrow": "VaR 95%",
            "value": f"{metrics.get('var_95', 0.0) * 100:.2f}%",
            "hint": "Expected worst 1-day loss at 95% confidence.",
        },
    ]


def _render_results(result: Dict[str, object], metrics: Dict[str, float]) -> None:
    score = float(result["score"])
    category = str(result["category"])
    vector = result["vector"]
    fuzzy_labels = result["fuzzy_labels"]
    blurb = RISK_BLURB.get(category, "")

    gauge_col, verdict_col = st.columns([3, 8], gap="large")

    with gauge_col:
        st.plotly_chart(
            risk_gauge(score, category),
            theme=None,
            key="dashboard_gauge",
        )

    with verdict_col:
        st.markdown(
            '<div class="srp-verdict">'
            f'<div class="srp-card-eyebrow">Assessment</div>'
            f'<div class="srp-verdict-score">{score:.2f}'
            '<span class="srp-verdict-max">/100</span></div>'
            f'<div class="srp-verdict-cat">{category.upper()} RISK</div>'
            f'<div class="srp-verdict-text">{blurb}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        callout(
            f"Forecast next-horizon maximum drawdown: "
            f"<b>{float(result['predicted_mdd']) * 100:.2f}%</b> · "
            f"headline category derived from the fusion network's classification head."
        )
        subtle_note(model_status_note())

    section_title("Fuzzy linguistic interpretation",
                  "How the ANFIS layer fuzzified each crisp input into a linguistic term.")
    fuzzy_pills(fuzzy_labels)

    st.divider()
    section_title("Extracted risk metrics",
                  "Sharpe, drawdown and volatility figures computed from the loaded series.")
    render_cards(_metric_cards(metrics), columns=3)

    st.divider()
    _render_explainer(vector)


def _render_explainer(vector: np.ndarray) -> None:
    section_title("Why this score?",
                  "Membership curves and rule firings produced by the Takagi-Sugeno layer.")

    degrees = services.membership_degrees(vector)
    current = dict(zip(services.FEATURE_KEYS, [float(v) for v in vector]))

    bounds = services.get_extractor().feature_bounds
    from config.settings import FUZZY_INPUT_SPECS

    fig = charts.membership_chart(
        feature_keys=list(services.FEATURE_KEYS),
        feature_labels=[services.FEATURE_LABELS[k] for k in services.FEATURE_KEYS],
        bounds=bounds,
        specs=FUZZY_INPUT_SPECS,
        current=current,
        degrees=degrees,
    )
    st.plotly_chart(fig, theme=None, key="dash_membership")

    strengths = services.rule_firing_strengths(vector)
    catalog = services.rule_catalog()

    top_idx = int(np.argmax(strengths))
    top = catalog[top_idx]
    callout(
        f"<b>Dominant rule R{top['id']}</b> fires at <b>{strengths[top_idx] * 100:.2f}%</b> "
        f"of the normalized fuzzy output — “{services.describe_rule(top)} THEN risk is elevated.”"
    )

    tab_bar, tab_table = st.tabs(["Firing strength chart", "Rule base table"])
    with tab_bar:
        st.plotly_chart(
            charts.rule_firing_chart(catalog, strengths),
            theme=None, key="dash_rules",
        )
    with tab_table:
        table = pd.DataFrame([
            {
                "Rule": f"R{c['id']}",
                "Antecedent": services.describe_rule(c),
                "Firing %": f"{strengths[i] * 100:.3f}",
            }
            for i, c in enumerate(catalog)
        ])
        st.dataframe(table, hide_index=True)


def _select_featured(symbol: str) -> None:
    """Preset callback — runs before the script body, so widget keys are safe."""
    st.session_state["search_query"] = symbol
    st.session_state["resolved_query"] = symbol
    st.session_state["resolution"] = services.resolve_free_text(symbol)
    st.session_state["search_open"] = False


def _pick_suggestion(ticker: str) -> None:
    """Autocomplete row callback — selects the listing and closes the dropdown."""
    st.session_state["search_query"] = ticker
    st.session_state["resolved_query"] = ticker
    st.session_state["resolution"] = services.resolve_free_text(ticker)
    st.session_state["search_open"] = False


def _suggestion_rows(query: str) -> None:
    """Renders the Google-style autocomplete panel beneath the search box."""
    rows = services.company_suggestions(query, limit=8)
    if not rows:
        return

    open_panel = st.session_state.get("search_open", True)
    if not open_panel:
        return

    for row in rows:
        st.button(
            f"{row['label']}  ·  {row['ticker']}",
            key=f"sugg_{row['ticker']}",
            width="stretch",
            help=f"{row['label']} — select to analyse",
            on_click=_pick_suggestion,
            args=(row["ticker"],),
        )


def _google_quote_card(ticker: str, exchange: str) -> None:
    """Live Google Finance quote strip shown above the risk assessment."""
    quote = services.google_finance_quote(ticker, exchange)

    if not quote or quote.get("price") is None:
        st.caption(
            "Google Finance quote unavailable for this listing right now — "
            "risk metrics still come from the full price history below."
        )
        return

    cards = [
        {
            "eyebrow": f"Google Finance · {quote['exchange']}",
            "value": f"{quote['currency']} {quote['price']:,.2f}",
            "hint": f"Live quote for {quote['name']} at {quote['retrieved_at']}.",
        },
        {
            "eyebrow": "Company",
            "value": quote["name"],
            "hint": f"Resolved ticker {quote['ticker']}.",
        },
        {
            "eyebrow": "Source",
            "value": "Google",
            "hint": "Read from the public Google Finance quote page.",
        },
    ]
    render_cards(cards, columns=3)
    st.markdown(
        f'<div class="srp-hint">Quote retrieved from Google Finance at '
        f'{quote["retrieved_at"]}. Risk metrics below are computed from the full '
        f'price history, not this single price.</div>',
        unsafe_allow_html=True,
    )


def _load_yahoo() -> Tuple[pd.DataFrame, str, str, str]:
    """
    Resolves the Yahoo Finance source.

    Returns (df, ticker, company_name, note).
    """
    st.markdown(
        '<div class="srp-card-eyebrow">Step 1 — find the company</div>',
        unsafe_allow_html=True,
    )

    query_row = st.columns([4.6, 1.8, 1.6], gap="medium")
    with query_row[0]:
        query = st.text_input(
            "Company name or ticker",
            key="search_query",
            placeholder="Search a company — try NVIDIA, JPM or Costco",
            label_visibility="collapsed",
            help="Start typing and pick a suggestion, exactly like a search engine.",
        )
        # The panel hangs directly beneath the search box, matching its width.
        st.session_state.setdefault("search_open", True)
        _suggestion_rows(query)
    with query_row[1]:
        st.markdown("")
        analyse = st.button("Analyse", width="stretch", type="primary")
    with query_row[2]:
        st.markdown("")
        refresh = st.button("Refresh", width="stretch")

    featured = st.columns(6, gap="small")
    for col, symbol in zip(featured, services.FEATURED_STOCKS[:6]):
        with col:
            st.button(
                symbol,
                key=f"featured_{symbol}",
                width="stretch",
                help=services.resolve_free_text(symbol)["name"],
                on_click=_select_featured,
                args=(symbol,),
            )

    query_changed = query != st.session_state.get("resolved_query")

    if analyse:
        # An explicit confirmation closes the panel.
        st.session_state["search_open"] = False
    elif query_changed:
        # Editing the box always brings the suggestions back.
        st.session_state["search_open"] = True

    if analyse or (len(query) >= 2 and query_changed):
        # The analysis keeps updating live while typing; the dropdown only
        # closes on an explicit choice (suggestion row, preset or Analyse).
        st.session_state["resolved_query"] = query
        st.session_state["resolution"] = services.resolve_free_text(query)

    resolution = st.session_state.get("resolution")
    if resolution is None:
        resolution = services.resolve_free_text(services.FEATURED_STOCKS[0])
        st.session_state["resolution"] = resolution
        st.session_state["resolved_query"] = services.FEATURED_STOCKS[0]

    suggestions = resolution.get("suggestions") or []
    if suggestions:
        st.caption("Google suggestions — " + " · ".join(s[:38] for s in suggestions[:4]))

    ticker = resolution["ticker"]
    company_name = resolution["name"]
    if not ticker:
        st.info("Enter a company name or ticker to begin.")
        st.stop()

    if resolution.get("matched_by") == "unresolved — treated as a raw ticker":
        callout(
            f"Google and the built-in map did not recognise <b>{ticker}</b>. "
            "It will be tried as a raw ticker symbol."
        )

    period_row = st.columns([2.4, 7.6], gap="medium")
    with period_row[0]:
        period_label = st.selectbox("History window", list(PERIODS.keys()), index=1)
    with period_row[1]:
        st.markdown("")
        _google_quote_card(ticker, resolution.get("exchange", "NASDAQ"))

    if refresh:
        services.fetch_stock.clear()
        services.google_finance_quote.clear()
        st.rerun()

    period = PERIODS[period_label]
    fetch_ticker = services.fetch_symbol(ticker)

    try:
        with st.spinner(f"Loading {period_label} of price history for {fetch_ticker}…"):
            df, is_synthetic = services.fetch_stock(fetch_ticker, period)
    except Exception as exc:
        st.error(f"Could not load data for **{ticker}** — {exc}")
        st.stop()

    if df is None or df.empty:
        st.error(f"No price history returned for **{ticker}**.")
        st.stop()

    note = "synthetic fallback series" if is_synthetic else "live yfinance series"
    return df, ticker, company_name, note


def _load_google() -> Tuple[pd.DataFrame, str, str, str]:
    """Resolves a Google Sheets source. Returns (df, ticker, company_name, note)."""
    control = st.columns([6.0, 3.4], gap="medium")
    with control[0]:
        url = st.text_input(
            "Google Sheets URL or sheet ID",
            value=st.session_state.get("gs_url", ""),
            placeholder="https://docs.google.com/spreadsheets/d/1AbC…/edit",
            help=services.SHEET_HELP,
        )
    with control[1]:
        name = st.text_input(
            "Label shown on charts",
            value=st.session_state.get("gs_name", "Custom Portfolio"),
            help="Used as the ticker name in chart titles and the CSV export.",
        )

    st.caption(services.SHEET_HELP)

    left, middle, right = st.columns([1.4, 1.2, 3], gap="medium")
    with left:
        load = st.button("Load sheet", type="primary", width="stretch")
    with middle:
        refresh = st.button("Reload", width="stretch")
        if refresh:
            services.fetch_google_sheet.clear()
            st.rerun()

    if load:
        st.session_state["gs_url"] = url
        st.session_state["gs_name"] = name
        try:
            with st.spinner("Reading the sheet from Google…"):
                df, notes = services.fetch_google_sheet(url)
            st.session_state["gs_df"] = df
            st.session_state["gs_notes"] = notes
            st.session_state["gs_error"] = None
        except Exception as exc:
            st.session_state["gs_error"] = str(exc)

    error = st.session_state.get("gs_error")
    if error:
        st.error(error)
        st.stop()

    df = st.session_state.get("gs_df")
    if df is None:
        callout(
            "No sheet loaded yet. Paste a link above and choose <b>Load sheet</b> — the "
            "app reads the CSV export directly from Google, then runs the identical "
            "risk pipeline used for live tickers. Expected columns: "
            "<b>Date</b> plus <b>Close</b> (Open / High / Low / Volume are optional "
            "and derived from Close when absent)."
        )
        st.stop()

    notes = st.session_state.get("gs_notes", [])
    label = name.strip() or "Custom Portfolio"
    return df, label.upper().replace(" ", "_")[:12], label, "google sheets dataset"


def render() -> None:
    page_header(
        "Risk dashboard",
        "Market Risk Dashboard",
        "Analyse any listed company with the hybrid Deep Neuro-Fuzzy network. "
        "Pull prices live from Yahoo Finance, read your own dataset straight from "
        "Google Sheets, or fall back to a Merton Jump Diffusion simulator when the "
        "network is unavailable.",
    )

    source = st.radio(
        "Data source",
        ["Yahoo Finance", "Google Sheets"],
        horizontal=True,
        help="Yahoo Finance covers listed tickers. Google Sheets lets you analyse any "
             "custom price series you publish to the web.",
    )

    if source == "Yahoo Finance":
        df, ticker, company_name, note = _load_yahoo()
    else:
        df, ticker, company_name, note = _load_google()

    if not services.model_is_trained() and services.model_training_status()["torch"]:
        with st.spinner(
            "First run: fitting the risk head so the score responds to input "
            "(this happens once and is cached)…"
        ):
            services.ensure_trained()

    with st.spinner("Extracting risk metrics and running the hybrid model…"):
        try:
            metrics = services.get_extractor().extract_metrics_from_df(df)
            vector = services.metrics_to_vector(metrics)
            sequence = services.sequence_from_frame(df)
            result = services.predict_risk(vector, sequence)
        except Exception as exc:
            st.error(f"Could not compute risk metrics from this dataset — {exc}")
            st.stop()

    st.caption(
        f"**{company_name}** · {len(df)} trading bars · "
        f"{df.index.min().date()} → {df.index.max().date()} · {note}"
    )
    if "synthetic" in note:
        callout(
            "Offline mode: a high-fidelity <b>Merton Jump Diffusion</b> simulator generated "
            "this series because live market data was unreachable. Metrics remain fully valid "
            "for demonstrating the pipeline."
        )
    if source == "Google Sheets":
        notes = st.session_state.get("gs_notes", [])
        if notes:
            callout(
                "Column mapping applied — "
                + ", ".join(n.lower() for n in notes)
                + ". Supply true Open / High / Low columns for a faithful "
                "Garman-Klass implied volatility."
            )

    _render_results(result, metrics)

    st.divider()
    section_title("Interactive market charts",
                  "Hover, zoom and drag the range slider — all rows share one x-axis.")

    tab_price, tab_vol, tab_breakdown, tab_raw = st.tabs(
        ["Price & drawdown", "Volatility", "Risk breakdown", "Metrics & data"]
    )

    with tab_price:
        chart_type = st.radio(
            "Price rendering",
            ["Line", "Candlestick"],
            horizontal=True,
            help="Candlestick requires OHLC columns, which the loader always provides.",
        )
        st.plotly_chart(
            charts.price_drawdown_chart(df, ticker, metrics, chart_type),
            theme=None, key="dash_price",
        )
        st.plotly_chart(
            charts.monthly_returns_heatmap(df, ticker),
            theme=None, key="dash_monthly",
        )

    with tab_vol:
        st.plotly_chart(
            charts.volatility_chart(df, metrics),
            theme=None, key="dash_vol",
        )
        st.caption(
            "Rolling HV uses a 21-day window annualised by √252. The dashed line is the "
            "Garman-Klass implied-volatility proxy over the same period."
        )

    with tab_breakdown:
        st.plotly_chart(
            category_bar(services.class_probabilities(vector)),
            theme=None, key="dash_probs",
        )
        st.caption(
            "Softmax probability assigned by the classification head to each of the four "
            "risk levels. The headline category is the arg-max of this distribution."
        )

    with tab_raw:
        show_metrics = st.checkbox("Show extracted risk metrics table", value=True)
        if show_metrics:
            metrics_table = pd.DataFrame(
                [
                    ("Sharpe Ratio", metrics.get("sharpe", 0.0), "Return vs total risk"),
                    ("Maximum Drawdown", metrics.get("drawdown", 0.0), "Peak-to-trough decline"),
                    ("Implied Volatility", metrics.get("implied_vol", 0.0), "Garman-Klass proxy"),
                    ("Historical Volatility", metrics.get("hist_vol", 0.0), "21-day log-return σ"),
                    ("Sortino Ratio", metrics.get("sortino", 0.0), "Downside-adjusted return"),
                    ("Value at Risk 95%", metrics.get("var_95", 0.0), "Worst 1-day loss"),
                ],
                columns=["Metric", "Value", "Description"],
            )
            st.dataframe(metrics_table, hide_index=True)

        st.download_button(
            "Download price history (CSV)",
            data=df.to_csv().encode("utf-8"),
            file_name=f"{ticker}_prices.csv",
            mime="text/csv",
        )
