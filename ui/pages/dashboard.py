"""
Risk Dashboard page: pick a company, fetch market data, run the hybrid model and
explore the result through interactive charts and fuzzy-rule explanations.
"""

from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from ui import charts, services
from config.settings import get_currency_config
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
            "eyebrow": "Historical Drawdown (Observed)",
            "value": f"{metrics.get('drawdown', 0.0) * 100:.2f}%",
            "hint": "Largest peak-to-trough decline over the selected trailing window.",
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


def _render_results(result: Dict[str, object], metrics: Dict[str, float], ticker: str) -> None:
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
            key=f"dashboard_gauge_{ticker}",
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
            f"Forecast next-horizon (10-day) maximum drawdown: "
            f"<b>{float(result['predicted_mdd']) * 100:.2f}%</b> · "
            f"Historical observed drawdown: <b>{metrics.get('drawdown', 0.0) * 100:.2f}%</b> · "
            f"headline category derived from the fusion network's classification head."
        )
        gp_expr = result.get("gp_expression", "(none)")
        st.markdown(
            f'<div style="display:flex; flex-wrap:wrap; gap:6px; margin: 8px 0 10px 0;">'
            f'<span class="srp-pill" style="background:#f1f5f9; color:#0f172a; border-color:#cbd5e1; font-size:0.78rem;">'
            f'<span class="srp-pill-dot" style="background:#2563eb;"></span><b>BiLSTM</b>: 128-D Temporal</span>'
            f'<span class="srp-pill" style="background:#f1f5f9; color:#0f172a; border-color:#cbd5e1; font-size:0.78rem;">'
            f'<span class="srp-pill-dot" style="background:#10b981;"></span><b>TSK Fuzzy</b>: 16 Rules</span>'
            f'<span class="srp-pill" style="background:#f1f5f9; color:#0f172a; border-color:#cbd5e1; font-size:0.78rem;">'
            f'<span class="srp-pill-dot" style="background:#f59e0b;"></span><b>GA</b>: 24 Tuned MF Params</span>'
            f'<span class="srp-pill" style="background:#f1f5f9; color:#0f172a; border-color:#cbd5e1; font-size:0.78rem;">'
            f'<span class="srp-pill-dot" style="background:#8b5cf6;"></span><b>GP Signal</b>: <code>{gp_expr}</code></span>'
            f'</div>',
            unsafe_allow_html=True,
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
    _render_explainer(vector, ticker)


def _render_explainer(vector: np.ndarray, ticker: str) -> None:
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
    st.plotly_chart(fig, theme=None, key=f"dash_membership_{ticker}")

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
            theme=None, key=f"dash_rules_{ticker}",
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


def _google_quote_card(ticker: str, exchange: str, df: Optional[pd.DataFrame] = None) -> None:
    """Live quote strip shown above the risk assessment."""
    curr_cfg = get_currency_config(ticker)
    curr_sym = curr_cfg["symbol"]
    curr_code = curr_cfg["code"]
    quote = services.google_finance_quote(ticker, exchange)

    price = quote.get("price") if quote else None
    source_label = quote.get("source_name", "Google Finance") if quote else "Market Feed"
    exch_label = quote.get("exchange", exchange) if quote else exchange
    name_label = quote.get("name") if (quote and quote.get("name")) else services.COMPANY_NAME_MAP.get(ticker, ticker)
    retrieved_time = quote.get("retrieved_at", datetime.now().strftime("%H:%M:%S")) if quote else datetime.now().strftime("%H:%M:%S")

    # If live quote failed, use latest price from historical data df as reliable fallback
    if (price is None or price <= 0) and df is not None and not df.empty and "Close" in df:
        close_series = df["Close"].dropna()
        if not close_series.empty:
            price = float(close_series.iloc[-1])
            source_label = "Historical Close"

    if price is None:
        st.caption(
            f"Live market quote unavailable for this listing right now — "
            f"risk metrics and prices ({curr_code}) still come from the full price history below."
        )
        return

    cards = [
        {
            "eyebrow": f"{source_label} · {exch_label}",
            "value": f"{curr_sym}{price:,.2f} ({curr_code})",
            "hint": f"Quote for {name_label} at {retrieved_time}.",
        },
        {
            "eyebrow": "Company",
            "value": name_label,
            "hint": f"Resolved ticker {ticker}.",
        },
        {
            "eyebrow": "Currency",
            "value": f"{curr_code} ({curr_sym})",
            "hint": f"Pricing currency: {curr_cfg['locale']}.",
        },
    ]
    render_cards(cards, columns=3)
    st.markdown(
        f'<div class="srp-hint">Quote retrieved via {source_label} at '
        f'{retrieved_time}. Currency: <b>{curr_code} ({curr_sym})</b>. Risk metrics below are computed from the full '
        f'price history, not this single price.</div>',
        unsafe_allow_html=True,
    )


def _load_yahoo() -> Tuple[pd.DataFrame, str, str, str, str]:
    """
    Resolves the Yahoo Finance source.

    Returns (df, ticker, company_name, status_code, status_label).
    """
    st.markdown(
        '<div class="srp-card-eyebrow">Step 1 — find the company</div>',
        unsafe_allow_html=True,
    )

    curr_res = st.session_state.get("resolution")
    active_ticker = (
        curr_res.get("ticker")
        if curr_res and curr_res.get("ticker")
        else st.session_state.get("resolved_query") or services.FEATURED_STOCKS[0]
    )

    st.session_state.setdefault("search_query", active_ticker)

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

    st.markdown(
        '<div style="margin: 6px 0 3px 0; font-size: 0.76rem; font-weight: 700; '
        'color: var(--srp-slate-600); text-transform: uppercase; letter-spacing: 0.6px;">'
        'Quick-Select Stocks (Faculty Demonstration)</div>',
        unsafe_allow_html=True,
    )

    # 10 quick-select stock buttons in 2 clean, responsive rows of 5
    stocks = services.FEATURED_STOCKS[:10]
    row1 = st.columns(5, gap="small")
    for col, symbol in zip(row1, stocks[:5]):
        is_sel = (symbol == active_ticker)
        company_label = services.COMPANY_NAME_MAP.get(symbol, symbol)
        with col:
            st.button(
                symbol,
                key=f"featured_{symbol}",
                width="stretch",
                type="primary" if is_sel else "secondary",
                help=f"{symbol} — {company_label} (click to analyse)",
                on_click=_select_featured,
                args=(symbol,),
            )

    row2 = st.columns(5, gap="small")
    for col, symbol in zip(row2, stocks[5:10]):
        is_sel = (symbol == active_ticker)
        company_label = services.COMPANY_NAME_MAP.get(symbol, symbol)
        with col:
            st.button(
                symbol,
                key=f"featured_{symbol}",
                width="stretch",
                type="primary" if is_sel else "secondary",
                help=f"{symbol} — {company_label} (click to analyse)",
                on_click=_select_featured,
                args=(symbol,),
            )

    st.markdown(
        '<div style="margin: 10px 0 3px 0; font-size: 0.76rem; font-weight: 700; '
        'color: var(--srp-slate-600); text-transform: uppercase; letter-spacing: 0.6px;">'
        '🇮🇳 Indian Markets (NSE Benchmark Stocks)</div>',
        unsafe_allow_html=True,
    )
    from config.settings import NSE_FEATURED_STOCKS
    nse_row1 = st.columns(4, gap="small")
    for col, symbol in zip(nse_row1, NSE_FEATURED_STOCKS[:4]):
        is_sel = (symbol == active_ticker)
        company_label = services.COMPANY_NAME_MAP.get(symbol, symbol)
        short_name = symbol.replace(".NS", "")
        with col:
            st.button(
                f"{short_name} (NSE)",
                key=f"featured_{symbol}",
                width="stretch",
                type="primary" if is_sel else "secondary",
                help=f"{symbol} — {company_label} (click to analyse)",
                on_click=_select_featured,
                args=(symbol,),
            )

    nse_row2 = st.columns(4, gap="small")
    for col, symbol in zip(nse_row2, NSE_FEATURED_STOCKS[4:8]):
        is_sel = (symbol == active_ticker)
        company_label = services.COMPANY_NAME_MAP.get(symbol, symbol)
        short_name = symbol.replace(".NS", "")
        with col:
            st.button(
                f"{short_name} (NSE)",
                key=f"featured_{symbol}",
                width="stretch",
                type="primary" if is_sel else "secondary",
                help=f"{symbol} — {company_label} (click to analyse)",
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

    if refresh:
        services.clear_market_cache()
        services.google_finance_quote.clear()
        st.rerun()

    period = PERIODS[period_label]
    fetch_ticker = services.fetch_symbol(ticker)

    try:
        with st.spinner(f"Loading {period_label} of price history for {fetch_ticker}…"):
            df, status_code, status_label = services.fetch_stock_with_meta(fetch_ticker, period)
    except Exception as exc:
        st.error(f"Could not load data for **{ticker}** — {exc}")
        st.stop()

    if df is None or df.empty or len(df) < 30:
        st.error(f"Insufficient historical price data for **{ticker}** (minimum 30 trading bars required).")
        st.stop()

    with period_row[1]:
        st.markdown("")
        _google_quote_card(ticker, resolution.get("exchange", "NASDAQ"), df=df)

    return df, ticker, company_name, status_code, status_label


def _load_google() -> Tuple[pd.DataFrame, str, str, str, str]:
    """Resolves a Google Sheets source. Returns (df, ticker, company_name, status_code, status_label)."""
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
    return df, label.upper().replace(" ", "_")[:12], label, "google", "Google Sheets dataset"


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
        df, ticker, company_name, status_code, status_label = _load_yahoo()
    else:
        df, ticker, company_name, status_code, status_label = _load_google()

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

    status_badges = {
        "live": (
            '<span class="srp-pill" style="color:#15803d; background:#dcfce7; border-color:#86efac; font-weight:650;">'
            '<span class="srp-pill-dot" style="background:#22c55e;"></span>Live Yahoo Finance</span>'
        ),
        "cached": (
            '<span class="srp-pill" style="color:#1d4ed8; background:#dbeafe; border-color:#93c5fd; font-weight:650;">'
            '<span class="srp-pill-dot" style="background:#3b82f6;"></span>Cached Data</span>'
        ),
        "synthetic": (
            '<span class="srp-pill" style="color:#c2410c; background:#ffedd5; border-color:#fdba74; font-weight:650;">'
            '<span class="srp-pill-dot" style="background:#f97316;"></span>Synthetic Fallback</span>'
        ),
    }
    badge_html = status_badges.get(status_code, f'<span class="srp-pill">{status_label}</span>')

    st.markdown(
        f'<div style="display:flex; align-items:center; gap:8px; margin: 4px 0 14px 0; flex-wrap:wrap;">'
        f'{badge_html}'
        f'<span style="font-size:0.88rem; color:var(--srp-slate-600);">'
        f'<b>{company_name} ({ticker})</b> · <b>{len(df)}</b> trading bars · '
        f'<b>{df.index.min().date()}</b> → <b>{df.index.max().date()}</b> · {status_label}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    if status_code == "synthetic":
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

    _render_results(result, metrics, ticker)

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
            key=f"chart_type_{ticker}",
        )
        st.plotly_chart(
            charts.price_drawdown_chart(df, ticker, metrics, chart_type),
            theme=None, key=f"dash_price_{ticker}_{chart_type}",
        )
        st.plotly_chart(
            charts.monthly_returns_heatmap(df, ticker),
            theme=None, key=f"dash_monthly_{ticker}",
        )

    with tab_vol:
        st.plotly_chart(
            charts.volatility_chart(df, metrics),
            theme=None, key=f"dash_vol_{ticker}",
        )
        st.caption(
            "Rolling HV uses a 21-day window annualised by √252. The dashed line is the "
            "Garman-Klass implied-volatility proxy over the same period."
        )

    with tab_breakdown:
        st.plotly_chart(
            category_bar(services.class_probabilities(vector)),
            theme=None, key=f"dash_probs_{ticker}",
        )
        st.caption(
            "Softmax probability assigned by the classification head to each of the four "
            "risk levels. The headline category is the arg-max of this distribution."
        )

    with tab_raw:
        show_metrics = st.checkbox("Show extracted risk metrics table", value=True, key=f"show_metrics_{ticker}")
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
            key=f"download_{ticker}",
        )

    st.divider()
    _render_cross_stock_comparison(ticker)

    st.divider()
    _render_algorithm_architecture()

    st.divider()
    _render_model_performance()


def _render_cross_stock_comparison(active_ticker: str) -> None:
    section_title(
        "Cross-Stock Risk Comparison",
        "Empirical proof of model sensitivity: identical trained hybrid model evaluated across 10 liquid demonstration stocks."
    )
    with st.spinner("Loading cross-stock risk benchmarks…"):
        df_comp = services.fetch_featured_stock_comparison()

    if df_comp is not None and not df_comp.empty:
        col_chart, col_table = st.columns([6, 5], gap="medium")
        with col_chart:
            st.plotly_chart(
                charts.risk_comparison_bar_chart(df_comp, highlight_ticker=active_ticker),
                theme=None,
                key=f"dash_comp_bar_{active_ticker}",
                width="stretch",
            )
        with col_table:
            st.markdown(
                '<div style="font-size:0.85rem; font-weight:700; color:var(--srp-slate-600); margin-bottom:8px; text-transform:uppercase; letter-spacing:0.5px;">'
                'Comparison Table (Sorted by Risk Score)</div>',
                unsafe_allow_html=True,
            )
            df_display = df_comp.sort_values(by="Risk Score", ascending=True).copy()
            st.dataframe(
                df_display,
                hide_index=True,
                column_config={
                    "Risk Score": st.column_config.NumberColumn("Risk Score", format="%.2f"),
                    "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                },
                width="stretch",
            )
            callout(
                f"<b>Currently analyzing: {active_ticker}</b>. Low-volatility defensive stocks sit at lower scores, "
                f"while higher-drawdown / high-beta stocks scale dynamically into higher risk categories."
            )


def _render_algorithm_architecture() -> None:
    section_title(
        "Hybrid Architecture & Core Algorithms",
        "How the four computing paradigms collaborate to predict risk without black-box opacity."
    )
    summary = services.get_model_performance_summary()
    algos = summary.get("algorithms", [])

    cols = st.columns(2, gap="medium")
    for i, algo in enumerate(algos[:4]):
        col = cols[i % 2]
        with col:
            st.markdown(
                f'<div class="srp-card" style="margin-bottom:12px; height:100%;">'
                f'<div class="srp-card-eyebrow" style="color:#2563eb;">{algo["role"]}</div>'
                f'<div style="font-size:1.02rem; font-weight:700; color:#0f172a; margin: 4px 0 6px 0;">{algo["name"]}</div>'
                f'<div style="font-size:0.86rem; color:#475569; line-height:1.45;">{algo["desc"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if len(algos) > 4:
        st.markdown(
            f'<div class="srp-card" style="background:#f8fafc; border:1px solid #cbd5e1; margin-top:6px;">'
            f'<div class="srp-card-eyebrow" style="color:#0f172a;">{algos[4]["role"]}</div>'
            f'<div style="font-size:1.02rem; font-weight:700; color:#0f172a; margin: 4px 0 6px 0;">{algos[4]["name"]}</div>'
            f'<div style="font-size:0.86rem; color:#475569; line-height:1.45;">{algo["desc"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_model_performance() -> None:
    section_title(
        "Model Performance & Benchmark Results",
        "Pre-computed empirical evaluation metrics demonstrating model accuracy against conventional baselines."
    )
    summary = services.get_model_performance_summary()
    metrics = summary.get("metrics", {})

    cards = [
        {
            "eyebrow": "Held-Out Test MAE",
            "value": f"{metrics.get('test_mae', 7.19):.2f}",
            "hint": "Overall held-out test error (0-100 scale) across all 8 test equities.",
        },
        {
            "eyebrow": "Test Accuracy",
            "value": f"{metrics.get('test_accuracy', 66.5):.1f}%",
            "hint": "4-way risk category classification accuracy on test partition.",
        },
        {
            "eyebrow": "Sector Error Range",
            "value": f"{metrics.get('financials_mae', 3.96):.2f} – {metrics.get('consumer_disc_mae', 11.64):.2f}",
            "hint": "Low error in Financials (JPM: 3.96) vs higher in Consumer Disc (TSLA: 11.64).",
        },
        {
            "eyebrow": "Training Universe",
            "value": f"{metrics.get('universe_size', 20)} Stocks",
            "hint": "Diversified equities across 6 sectors (12 supervised epochs).",
        },
    ]
    render_cards(cards, columns=4)

    baselines_df = summary.get("baselines_df")
    if baselines_df is not None:
        st.markdown(
            '<div style="font-size:0.88rem; font-weight:600; color:var(--srp-slate-600); margin: 16px 0 6px 0;">'
            'Benchmark Comparison (Identical Held-Out Partition)</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(baselines_df, hide_index=True, width="stretch")
        st.caption(
            "<b>Evaluation protocol & analysis:</b> Evaluated on an identical strictly chronological held-out test partition "
            "with forward look-ahead boundaries to eliminate data leakage. "
            "<br><b>Academic Assessment:</b> The Full Hybrid model (Score MAE 7.194, Accuracy 66.5%) does not outperform every "
            "simpler baseline; unconstrained tabular models like Linear Regression (MAE 5.923) and Random Forest (Accuracy 69.4%) "
            "achieve lower quantitative error. The hybrid architecture was designed for <b>soft-computing explainability</b> "
            "— providing transparent Gaussian membership curves, explicit rule firing strengths, and symbolic GP expressions — "
            "rather than black-box score minimization. Note that headline sector MAE spans from 3.96 on stable assets (JPM) "
            "to 11.64 on high-volatility assets (TSLA).",
            unsafe_allow_html=True,
        )

