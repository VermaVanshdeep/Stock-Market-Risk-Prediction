"""
Interactive Plotly chart builders for the Streamlit UI.
Restricted styling palette: Standard Default and Blue accents only.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NAVY = "#0f172a"
BLUE = "#1d4ed8"
BLUE_600 = "#2563eb"
BLUE_500 = "#3b82f6"
BLUE_400 = "#60a5fa"
BLUE_300 = "#93c5fd"
BLUE_200 = "#bfdbfe"
BLUE_100 = "#dbeafe"
BLUE_50 = "#eff6ff"
GRID = "#eef3fa"
MUTED = "#64748b"

FONT = "Segoe UI, sans-serif"


# --------------------------------------------------------------------------- #
# Shared layout                                                                #
# --------------------------------------------------------------------------- #
def _style(
    fig: go.Figure,
    title: str = "",
    height: int = 340,
    y_title: str = "",
    hovermode: str = "x unified",
) -> go.Figure:
    """Shared layout with enough headroom that titles never collide with legends."""
    fig.update_layout(
        height=height,
        # Left padding must clear the y tick labels *and* the axis title.
        margin=dict(
            l=(76 if y_title else 12),
            r=20,
            t=(66 if title else 30),
            b=54,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=NAVY),
        title=dict(
            text=title,
            x=0.01, xanchor="left",
            y=0.985, yanchor="top",
            font=dict(size=15, color=NAVY),
        ),
        hovermode=hovermode,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zerolinecolor="#cbd5e1", linewidth=1)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zerolinecolor="#cbd5e1", linewidth=1,
                     title=y_title)
    return fig


# --------------------------------------------------------------------------- #
# Market charts                                                                #
# --------------------------------------------------------------------------- #
def price_drawdown_chart(df: pd.DataFrame, ticker: str, metrics: Dict[str, float],
                         chart_type: str = "Line") -> go.Figure:
    """Two-row linked figure: price history on top, drawdown profile below."""
    prices = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    dates = df.index

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=(f"{ticker} — Closing Price", "Drawdown from Running Peak"),
        row_heights=[0.6, 0.4],
    )

    if chart_type == "Candlestick" and all(c in df.columns for c in ("Open", "High", "Low", "Close")):
        fig.add_trace(
            go.Candlestick(
                x=dates, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"],
                name=ticker, increasing_line_color=BLUE_600,
                decreasing_line_color=BLUE_400, showlegend=False,
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=dates, y=prices, mode="lines", name="Close",
                line=dict(color=BLUE, width=2),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.10)",
                hovertemplate="%{y:$,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax * 100.0
    fig.add_trace(
        go.Scatter(
            x=dates, y=drawdown, mode="lines", name="Drawdown",
            line=dict(color=BLUE_500, width=1.6),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.25)",
            hovertemplate="%{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_color="#cbd5e1", line_width=1, row=2, col=1)

    worst = float(drawdown.min()) if len(drawdown) else 0.0
    fig.add_hline(
        y=worst, line_dash="dot", line_color=BLUE_400, row=2, col=1,
        annotation_text=f"Worst: {worst:.2f}%", annotation_position="bottom left",
        annotation_font=dict(size=11, color=BLUE_600),
    )

    fig.update_layout(
        height=540, margin=dict(l=10, r=18, t=64, b=24),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=NAVY),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis2=dict(rangeslider=dict(visible=True, thickness=0.05), type="date"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, linewidth=1)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, linewidth=1, ticksuffix="")
    return fig


def volatility_chart(df: pd.DataFrame, metrics: Dict[str, float]) -> go.Figure:
    """21-day rolling historical volatility versus the implied volatility proxy."""
    prices = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    dates = df.index

    log_rets = np.log(prices / prices.shift(1))
    rolling_hv = log_rets.rolling(21).std() * np.sqrt(252) * 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates, y=rolling_hv, mode="lines", name="Rolling HV (21d)",
            line=dict(color=BLUE, width=2),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.12)",
            hovertemplate="%{y:.2f}%<extra>HV</extra>",
        )
    )

    iv = float(metrics.get("implied_vol", 0.0)) * 100.0
    fig.add_hline(
        y=iv, line_dash="dash", line_color=BLUE_500, line_width=2,
        annotation_text=f"Implied Volatility {iv:.1f}%",
        annotation_position="top right",
        annotation_font=dict(size=11, color=BLUE_600),
    )

    avg_hv = float(rolling_hv.mean()) if len(rolling_hv) else 0.0
    if np.isfinite(avg_hv):
        fig.add_hline(
            y=avg_hv, line_dash="dot", line_color=BLUE_300,
            annotation_text=f"Average HV {avg_hv:.1f}%",
            annotation_position="bottom right",
            annotation_font=dict(size=11, color=MUTED),
        )

    return _style(fig, title="Volatility Profile — Historical vs Implied", height=380, y_title="Volatility (%)")


def monthly_returns_heatmap(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Calendar-style monthly return heatmap (blue-only sequential ramp)."""
    prices = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    monthly = prices.resample("ME").last()
    returns = monthly.pct_change().dropna() * 100.0
    if returns.empty:
        return _style(go.Figure(), title="Monthly Returns (insufficient history)", height=300)

    labels = returns.index.strftime("%b %Y")
    fig = go.Figure(
        go.Heatmap(
            z=[returns.values],
            x=labels,
            y=["Monthly return"],
            colorscale=[
                [0.0, "#eff6ff"],
                [0.25, "#bfdbfe"],
                [0.5, "#93c5fd"],
                [0.75, "#3b82f6"],
                [1.0, "#172554"],
            ],
            zmid=0,
            colorbar=dict(title="%", thickness=14, outlinewidth=0, len=0.62, nticks=4, xpad=8),
            hovertemplate="%{x}<br>%{z:.2f}%<extra></extra>",
        )
    )
    # Thin the month labels so they never collide on narrow viewports.
    step = max(1, len(labels) // 10)
    fig.update_layout(
        height=260,
        margin=dict(l=8, r=8, t=54, b=64),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=NAVY),
        title=dict(text=f"{ticker} — Monthly Returns", x=0.01, xanchor="left",
                   y=0.97, yanchor="top", font=dict(size=15, color=NAVY)),
        xaxis=dict(showgrid=False, tickangle=-45,
                   tickmode="array",
                   tickvals=list(range(0, len(labels), step)),
                   ticktext=list(labels[::step]),
                   tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    return fig


# --------------------------------------------------------------------------- #
# Soft-computing explainability charts                                        #
# --------------------------------------------------------------------------- #
def membership_chart(
    feature_keys: List[str],
    feature_labels: List[str],
    bounds: Dict[str, tuple],
    specs: Dict[str, Dict[str, Dict[str, float]]],
    current: Dict[str, float],
    degrees: np.ndarray,
) -> go.Figure:
    """Gaussian membership curves with the live input marked on every subplot."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=feature_labels,
        horizontal_spacing=0.12, vertical_spacing=0.16,
    )

    shades = [BLUE_300, BLUE_500, BLUE]
    terms = ["Low", "Medium", "High"]
    # Stagger the value labels: two terms can land at the same membership
    # level, and centred labels would print on top of each other.
    label_positions = ["top center", "bottom center", "middle right"]

    for idx, key in enumerate(feature_keys):
        row = idx // 2 + 1
        col = idx % 2 + 1
        lo, hi = bounds.get(key, (0.0, 1.0))
        x = np.linspace(lo, hi, 300)

        for t_idx, (term, shade) in enumerate(zip(terms, shades)):
            params = specs[key][term.lower()]
            mu = np.exp(-0.5 * ((x - params["mean"]) / params["sigma"]) ** 2)
            fig.add_trace(
                go.Scatter(
                    x=x, y=mu, mode="lines", name=term,
                    legendgroup=term, showlegend=(idx == 0),
                    line=dict(color=shade, width=2),
                    hovertemplate=f"{feature_labels[idx]}<br>x=%{{x:.3f}}<br>"
                                  f"{term}: μ=%{{y:.3f}}<extra></extra>",
                ),
                row=row, col=col,
            )

            deg = float(degrees[idx][t_idx])
            fig.add_trace(
                go.Scatter(
                    x=[current[key]], y=[deg], mode="markers+text",
                    text=[f"{deg:.2f}"], textposition=label_positions[t_idx],
                    textfont=dict(size=9.5, color=NAVY),
                    marker=dict(color=shade, size=8, line=dict(color="#ffffff", width=1.5)),
                    legendgroup=term, showlegend=False,
                    cliponaxis=False,
                    hovertemplate=f"{term}: {deg:.3f}<extra></extra>",
                ),
                row=row, col=col,
            )

        fig.add_vline(
            x=current[key], line_dash="dash", line_color=NAVY, line_width=1.4,
            row=row, col=col,
        )

    fig.update_layout(
        height=560,
        margin=dict(l=64, r=18, t=104, b=96),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=11, color=NAVY),
        title=dict(text="Gaussian Fuzzy Membership Functions",
                   x=0.01, xanchor="left", y=0.98, yanchor="top",
                   font=dict(size=15, color=NAVY)),
        legend=dict(orientation="h", yanchor="top", y=-0.09, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, linewidth=1)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, range=[-0.05, 1.12], linewidth=1)

    # Axis titles only on the left column, so ticks never collide with them.
    for col in (2,):
        fig.update_yaxes(title_text="", row=1, col=col)
        fig.update_yaxes(title_text="", row=2, col=col)
    fig.update_yaxes(title_text="Degree of membership", row=1, col=1)
    fig.update_yaxes(title_text="Degree of membership", row=2, col=1)
    return fig


def rule_firing_chart(catalog: List[Dict[str, str]], strengths: np.ndarray) -> go.Figure:
    """Horizontal bar chart of normalized TSK rule firing strengths."""
    order = np.argsort(strengths)[::-1]
    labels, values, colors = [], [], []

    for i in order:
        entry = catalog[i]
        labels.append(
            f"R{entry['id']}  ·  Sharpe {entry['sharpe']} · DD {entry['drawdown']} · "
            f"IV {entry['implied_vol']} · HV {entry['hist_vol']}"
        )
        values.append(float(strengths[i]))
        colors.append(BLUE_500 if i == order[0] else BLUE_300)

    fig = go.Figure(
        go.Bar(
            x=[v * 100.0 for v in values], y=labels, orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Firing: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=470,
        margin=dict(l=8, r=24, t=58, b=68),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=11, color=NAVY),
        title=dict(text="Rule Firing Strengths (normalized contribution)",
                   x=0.01, xanchor="left", y=0.98, yanchor="top",
                   font=dict(size=15, color=NAVY)),
        xaxis=dict(title=dict(text="Share of final fuzzy output (%)", standoff=14),
                   gridcolor=GRID),
        yaxis=dict(automargin=True, gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
    )
    return fig


# --------------------------------------------------------------------------- #
# Optimization & testing charts                                                #
# --------------------------------------------------------------------------- #
def fitness_chart(history: List[float]) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=list(range(1, len(history) + 1)), y=history,
            mode="lines+markers", name="Best fitness",
            line=dict(color=BLUE, width=2.5),
            marker=dict(size=6, color=BLUE_400),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.12)",
            hovertemplate="Gen %{x}<br>Fitness %{y:.6f}<extra></extra>",
        )
    )
    return _style(fig, title="Genetic Algorithm — Fitness Convergence", height=360,
                  y_title="Fitness")


def training_curve(history: Dict[str, object]) -> go.Figure:
    """Training/validation MAE and category accuracy across epochs."""
    epochs = history.get("epoch", [])
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("train_mae", []), mode="lines+markers",
        name="Train MAE (score points)", line=dict(color=BLUE, width=2.4),
        marker=dict(size=5), hovertemplate="Epoch %{x}<br>Train MAE %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("val_mae", []), mode="lines+markers",
        name="Validation MAE", line=dict(color=BLUE_300, width=2.4, dash="dash"),
        marker=dict(size=5), hovertemplate="Epoch %{x}<br>Val MAE %{y:.3f}<extra></extra>",
    ))

    if history.get("val_acc"):
        axis = "yaxis2"
        fig.add_trace(go.Scatter(
            x=epochs, y=history.get("val_acc", []), mode="lines+markers",
            name="Validation category accuracy (%)", line=dict(color=BLUE_500, width=2),
            marker=dict(size=5), yaxis=axis,
            hovertemplate="Epoch %{x}<br>Accuracy %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            yaxis2=dict(title="Accuracy (%)", overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)", range=[0, 100]),
        )

    return _style(fig, title="Training Progress", height=380, y_title="Mean absolute score error")


def regime_response_chart(probe: Dict[str, object]) -> go.Figure:
    """Risk score across the four regime presets — proof the model is responsive."""
    profiles = probe.get("profiles", {})
    labels = list(profiles)
    scores = [profiles[k]["score"] for k in labels]

    figure = go.Figure(go.Bar(
        x=labels, y=scores,
        marker_color=[BLUE_300, BLUE_500, BLUE_600, BLUE],
        text=[f"{s:.1f}" for s in scores], textposition="outside",
        customdata=[profiles[k]["category"] for k in labels],
        hovertemplate="%{x}<br>Score %{y:.1f} / 100<br>%{customdata}<extra></extra>",
    ))

    for threshold, label in [(25, "Low"), (50, "Moderate"), (75, "High")]:
        figure.add_hline(
            y=threshold, line_dash="dot", line_color=BLUE_300,
            annotation_text=label, annotation_position="top right",
            annotation_font=dict(size=10, color=MUTED),
        )

    figure.update_layout(
        height=360, margin=dict(l=76, r=18, t=62, b=46),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=NAVY),
        title=dict(text="Model response across risk regimes", x=0.01, xanchor="left",
                   y=0.97, yanchor="top", font=dict(size=15, color=NAVY)),
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(title=dict(text="Risk score (0-100)", standoff=10),
                   range=[0, 105], gridcolor=GRID),
    )
    return figure


def feature_weights_chart(weights: np.ndarray) -> go.Figure:
    names = ["Sharpe", "Drawdown", "Implied Vol", "Historical Vol"]
    # Explicit line breaks keep the four labels from colliding in a narrow column.
    ticks = ["Sharpe", "Max<br>Drawdown", "Implied<br>Vol", "Historical<br>Vol"]
    fig = go.Figure(
        go.Bar(
            x=names, y=np.asarray(weights, dtype=float),
            marker_color=[BLUE_300, BLUE_500, BLUE_600, BLUE],
            text=[f"{w:.4f}" for w in np.asarray(weights, dtype=float)],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: %{y:.5f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=330, margin=dict(l=76, r=18, t=62, b=46),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=NAVY),
        title=dict(text="Optimized Feature Importance Weights", x=0.01, xanchor="left",
                   y=0.97, yanchor="top", font=dict(size=15, color=NAVY)),
        xaxis=dict(gridcolor=GRID, tickmode="array", tickvals=names, ticktext=ticks,
                   tickfont=dict(size=10)),
        yaxis=dict(title=dict(text="Weight", standoff=10), gridcolor=GRID),
    )
    return fig
