"""
Global styling for the Streamlit UI.
Restricted styling palette: Standard Default and Blue accents only.
"""

import streamlit as st

CSS = """
<style>
/* ------------------------------------------------------------------ */
/* Tokens                                                              */
/* ------------------------------------------------------------------ */
:root {
    --srp-blue-950: #172554;
    --srp-blue-900: #1e3a8a;
    --srp-blue-800: #1e40af;
    --srp-blue-700: #1d4ed8;
    --srp-blue-600: #2563eb;
    --srp-blue-500: #3b82f6;
    --srp-blue-300: #93c5fd;
    --srp-blue-200: #bfdbfe;
    --srp-blue-100: #dbeafe;
    --srp-blue-50:  #eff6ff;
    --srp-slate-900: #0f172a;
    --srp-slate-600: #475569;
    --srp-slate-400: #94a3b8;
    --srp-blue-900: #1e3a8a;
    --srp-border: #dbeafe;
    --srp-radius: 14px;
}

/* ------------------------------------------------------------------ */
/* Header / toolbar cleanup                                            */
/* ------------------------------------------------------------------ */
header[data-testid="stHeader"] {
    background: transparent;
    height: 0;
}
footer[data-testid="stFooter"] { display: none; }

/* ------------------------------------------------------------------ */
/* Sidebar                                                             */
/* ------------------------------------------------------------------ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--srp-blue-50) 0%, #ffffff 42%);
    border-right: 1px solid var(--srp-border);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.4rem;
}
.srp-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 4px 14px 4px;
    border-bottom: 1px solid var(--srp-border);
    margin-bottom: 14px;
}
.srp-brand-mark {
    width: 38px; height: 38px;
    border-radius: 11px;
    background: linear-gradient(135deg, var(--srp-blue-600) 0%, var(--srp-blue-800) 100%);
    color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 17px; letter-spacing: -0.5px;
    flex-shrink: 0;
    box-shadow: 0 3px 10px rgba(29, 78, 216, 0.28);
}
.srp-brand-name {
    font-size: 14.5px; font-weight: 700; color: var(--srp-slate-900);
    line-height: 1.2; letter-spacing: -0.2px;
}
.srp-brand-sub {
    font-size: 11px; color: var(--srp-blue-700); font-weight: 600;
    letter-spacing: 0.4px; text-transform: uppercase; margin-top: 2px;
}

/* Radio navigation */
section[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 2px;
}
section[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 10px;
    padding: 7px 10px;
    transition: background 0.15s ease;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: var(--srp-blue-50);
}
section[data-testid="stSidebar"] [role="radiogroup"] label p {
    font-size: 0.78rem !important;
    font-weight: 650 !important;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    color: var(--srp-slate-600) !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] p,
section[data-testid="stSidebar"] [role="radiogroup"] [aria-checked="true"] p {
    color: var(--srp-blue-800) !important;
}

/* ------------------------------------------------------------------ */
/* Page hero header                                                    */
/* ------------------------------------------------------------------ */
.srp-hero {
    padding: 2px 0 6px 0;
    margin-bottom: 6px;
}
.srp-hero-eyebrow {
    font-size: 11px; font-weight: 700; letter-spacing: 1.4px;
    text-transform: uppercase; color: var(--srp-blue-600);
    margin-bottom: 6px;
}
.srp-hero h1 {
    font-size: 1.85rem; font-weight: 750; color: var(--srp-slate-900);
    margin: 0; letter-spacing: -0.6px; line-height: 1.15;
}
.srp-hero p {
    font-size: 0.94rem; color: var(--srp-slate-600);
    margin: 8px 0 0 0; max-width: 78ch; line-height: 1.5;
}
.srp-rule {
    height: 3px; width: 62px; border-radius: 3px;
    background: linear-gradient(90deg, var(--srp-blue-600), var(--srp-blue-300));
    margin: 14px 0 4px 0;
}

.srp-section-title {
    font-size: 1.02rem; font-weight: 700; color: var(--srp-slate-900);
    letter-spacing: -0.2px; margin: 22px 0 4px 0;
}
.srp-section-note {
    font-size: 0.83rem; color: var(--srp-slate-600); margin: 0 0 12px 0;
}

/* ------------------------------------------------------------------ */
/* Cards & pills                                                       */
/* ------------------------------------------------------------------ */
.srp-grid {
    display: grid;
    gap: 12px;
    align-items: stretch;
}
.srp-card {
    background: #ffffff;
    border: 1px solid var(--srp-border);
    border-radius: var(--srp-radius);
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    min-width: 0;
    overflow-wrap: anywhere;
}
.srp-card-eyebrow {
    font-size: 10.5px; font-weight: 700; letter-spacing: 1.1px;
    text-transform: uppercase; color: var(--srp-blue-600); margin-bottom: 7px;
}
.srp-card-value {
    font-size: 1.65rem; font-weight: 750; color: var(--srp-slate-900);
    letter-spacing: -0.8px; line-height: 1.1;
}
.srp-card-hint {
    font-size: 0.76rem; color: var(--srp-slate-600); margin-top: 5px; line-height: 1.4;
}
.srp-card-value.is-low    { color: var(--srp-blue-500); }
.srp-card-value.is-mid    { color: var(--srp-blue-600); }
.srp-card-value.is-high   { color: var(--srp-blue-800); }
.srp-card-value.is-extreme{ color: var(--srp-blue-950); }

.srp-verdict {
    background: var(--srp-blue-50);
    border: 1px solid var(--srp-blue-200);
    border-radius: var(--srp-radius);
    padding: 16px 20px;
    margin-bottom: 10px;
}
.srp-verdict-score {
    font-size: 2.1rem; font-weight: 750; color: var(--srp-slate-900);
    letter-spacing: -1px; line-height: 1.05;
}
.srp-verdict-max {
    font-size: 1rem; font-weight: 600; color: var(--srp-slate-400);
    letter-spacing: 0; margin-left: 4px;
}
.srp-verdict-cat {
    display: inline-block;
    margin-top: 6px; padding: 3px 11px; border-radius: 999px;
    background: var(--srp-blue-700); color: #fff;
    font-size: 11px; font-weight: 700; letter-spacing: 1px;
}
.srp-verdict-text {
    margin-top: 10px; font-size: 0.88rem; line-height: 1.5;
    color: var(--srp-blue-950);
}

.srp-pill {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 5px 12px; border-radius: 999px;
    background: var(--srp-blue-50); border: 1px solid var(--srp-blue-200);
    font-size: 12.5px; font-weight: 600; color: var(--srp-blue-800);
    margin: 0 8px 8px 0; white-space: nowrap;
}
.srp-pill b { color: var(--srp-blue-950); font-weight: 750; }
.srp-pill-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--srp-blue-500); flex-shrink: 0;
}

.srp-note {
    background: var(--srp-blue-50);
    border: 1px solid var(--srp-blue-200);
    border-left: 4px solid var(--srp-blue-600);
    border-radius: 10px;
    padding: 12px 15px;
    font-size: 0.86rem; color: var(--srp-blue-950); line-height: 1.55;
    margin: 6px 0 14px 0;
}

.srp-hint {
    border: 1px dashed var(--srp-blue-200);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.8rem; color: var(--srp-slate-600); line-height: 1.5;
    margin: 4px 0 14px 0;
}
.srp-hint b { color: var(--srp-blue-800); font-weight: 700; }

/* ------------------------------------------------------------------ */
/* st.metric restyle                                                   */
/* ------------------------------------------------------------------ */
div[data-testid="stMetric"] {
    background: #f8fbff;
    border: 1px solid var(--srp-border);
    border-radius: 12px;
    padding: 12px 14px;
}
div[data-testid="stMetric"] label {
    font-size: 11.5px !important; font-weight: 650 !important;
    letter-spacing: 0.5px; text-transform: uppercase;
    color: var(--srp-blue-700) !important;
}
div[data-testid="stMetricValue"] {
    font-size: 1.35rem !important; font-weight: 750 !important;
    color: var(--srp-slate-900) !important;
}

/* ------------------------------------------------------------------ */
/* Tabs                                                                */
/* ------------------------------------------------------------------ */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid var(--srp-blue-100);
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 0.88rem;
}
div[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--srp-blue-50) !important;
    border-bottom: 2px solid var(--srp-blue-700) !important;
}

/* ------------------------------------------------------------------ */
/* Buttons                                                             */
/* ------------------------------------------------------------------ */
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    border-radius: 10px;
    font-weight: 650;
    letter-spacing: 0.2px;
    transition: transform 0.08s ease, box-shadow 0.15s ease;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(29, 78, 216, 0.22);
}

/* ------------------------------------------------------------------ */
/* Charts & tables                                                     */
/* ------------------------------------------------------------------ */
div[data-testid="stPlotlyChart"] {
    border: 1px solid var(--srp-border);
    border-radius: var(--srp-radius);
    padding: 6px;
    background: #ffffff;
}
div[data-testid="stDataFrame"],
div[data-testid="stDataEditor"] {
    border: 1px solid var(--srp-border);
    border-radius: 12px;
    overflow: hidden;
}

/* ------------------------------------------------------------------ */
/* Autocomplete dropdown (company search)                              */
/* ------------------------------------------------------------------ */
/* Streamlit tags keyed widgets with a st-key-<key> class, so the
   suggestion rows can be styled in isolation without touching the rest
   of the page. */
div[class*="st-key-sugg_"] {
    margin: 2px 0;
}

div[class*="st-key-sugg_"] button {
    width: 100%;
    justify-content: flex-start;
    text-align: left;
    background: #ffffff;
    color: var(--srp-slate-900);
    border: 1px solid var(--srp-blue-100);
    border-radius: 10px;
    font-weight: 550;
    font-size: 0.85rem;
    letter-spacing: 0;
    padding: 8px 12px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    transform: none;
    transition: background 0.12s ease, border-color 0.12s ease;
}

div[class*="st-key-sugg_"] button:hover,
div[class*="st-key-sugg_"] button:focus-visible {
    background: var(--srp-blue-50);
    border-color: var(--srp-blue-300);
    color: var(--srp-blue-900);
    box-shadow: 0 2px 8px rgba(29, 78, 216, 0.12);
    transform: none;
}

/* ------------------------------------------------------------------ */
/* Footer                                                              */
/* ------------------------------------------------------------------ */
.srp-footer {
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid var(--srp-border);
    font-size: 0.76rem; color: var(--srp-slate-400);
    display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap;
}

/* ------------------------------------------------------------------ */
/* Responsive                                                          */
/* ------------------------------------------------------------------ */
@media (max-width: 900px) {
    .srp-hero h1 { font-size: 1.45rem; }
    .srp-card-value { font-size: 1.4rem; }
}
@media (max-width: 1500px) {
    .srp-grid[data-cols="3"],
    .srp-grid[data-cols="4"] {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    }
}
@media (max-width: 860px) {
    .srp-grid[data-cols="2"],
    .srp-grid[data-cols="3"],
    .srp-grid[data-cols="4"] {
        grid-template-columns: 1fr !important;
    }
    .srp-footer { flex-direction: column; gap: 4px; }
    .srp-verdict-score { font-size: 1.7rem; }
}
</style>
"""


def inject_styles() -> None:
    """Injects the global stylesheet into the current Streamlit page."""
    st.markdown(CSS, unsafe_allow_html=True)
