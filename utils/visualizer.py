"""
Visualization & Formatting Utilities.
Provides Rich Terminal UI outputs and Matplotlib graphical plots for Soft Computing
membership functions, stock risk metrics, and deep model predictions.
Restricted styling palette: Standard Default and Blue accents only.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(soft_wrap=True)


def print_project_header():
    """Prints simple header for CLI interface using Blue palette."""
    header_text = Text(
        "STOCK MARKET RISK PREDICTION SYSTEM\n"
        "Deep Learning & Soft Computing Risk Predictor",
        style="bold blue"
    )
    console.print(Panel(header_text, border_style="blue", expand=False))


def display_metrics_table(metrics: Dict[str, float], title: str = "Stock Risk Values"):
    """Displays financial risk metrics in a clean table with Blue accents."""
    table = Table(title=title, show_header=True, header_style="bold blue", border_style="dim")
    table.add_column("Risk Metric", style="blue", width=25)
    table.add_column("Value", justify="right", style="bold white", width=15)
    table.add_column("Description", style="white")

    table.add_row("Sharpe Ratio", f"{metrics.get('sharpe', 0.0):.4f}", "Return vs risk level")
    table.add_row("Maximum Drawdown (MDD)", f"{metrics.get('drawdown', 0.0)*100:.2f}%", "Biggest drop from peak")
    table.add_row("Implied Volatility (IV)", f"{metrics.get('implied_vol', 0.0)*100:.2f}%", "Expected market volatility")
    table.add_row("Historical Volatility (HV)", f"{metrics.get('hist_vol', 0.0)*100:.2f}%", "Past 21-day volatility")
    table.add_row("Sortino Ratio", f"{metrics.get('sortino', 0.0):.4f}", "Downside risk-adjusted return")
    table.add_row("Value at Risk (VaR 95%)", f"{metrics.get('var_95', 0.0)*100:.2f}%", "Expected max 1-day loss")

    console.print(table)


def display_risk_prediction(risk_score: float, category: str, predicted_mdd: float, fuzzy_terms: Dict[str, str]):
    """Displays Risk Assessment Results using Default and Blue styling."""
    panel_content = (
        f"[bold blue]Risk Level: {category.upper()}[/bold blue]\n"
        f"Risk Score (0-100): [bold white]{risk_score:.2f} / 100.00[/bold white]\n"
        f"Predicted Next Drawdown: [bold white]{predicted_mdd*100:.2f}%[/bold white]\n\n"
        f"[bold blue]Fuzzy Logic Levels:[/bold blue]\n"
        f"  - Sharpe Ratio: {fuzzy_terms.get('sharpe', 'N/A')}\n"
        f"  - Max Drawdown: {fuzzy_terms.get('drawdown', 'N/A')}\n"
        f"  - Implied Volatility: {fuzzy_terms.get('implied_vol', 'N/A')}\n"
        f"  - Historical Volatility: {fuzzy_terms.get('hist_vol', 'N/A')}"
    )
    
    console.print(Panel(panel_content, title="Stock Risk Assessment Results", border_style="blue"))


def plot_fuzzy_membership_functions(save_path: Optional[str] = None):
    """
    Plots Gaussian Soft Computing Fuzzy Membership Functions using Blue spectrum curves.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Fuzzy Logic Membership Curves", fontsize=14, fontweight='bold', color='navy')

    specs = {
        'Sharpe Ratio': (np.linspace(-1, 3, 200), [('Low', 0.0, 0.5), ('Medium', 1.0, 0.5), ('High', 2.0, 0.5)]),
        'Max Drawdown': (np.linspace(0, 0.6, 200), [('Low', 0.05, 0.05), ('Medium', 0.20, 0.08), ('High', 0.40, 0.10)]),
        'Implied Volatility': (np.linspace(0, 0.8, 200), [('Low', 0.12, 0.04), ('Medium', 0.25, 0.06), ('High', 0.45, 0.10)]),
        'Historical Volatility': (np.linspace(0, 0.8, 200), [('Low', 0.12, 0.04), ('Medium', 0.25, 0.06), ('High', 0.45, 0.10)])
    }

    blue_shades = ['royalblue', 'mediumblue', 'darkblue']

    for ax, (title, (x, curves)) in zip(axes.flat, specs.items()):
        for (label, mean, sigma), color in zip(curves, blue_shades):
            y = np.exp(-0.5 * ((x - mean) / sigma) ** 2)
            ax.plot(x, y, label=f"{label} (mean={mean}, std={sigma})", color=color, linewidth=2)
        ax.set_title(title, fontweight='bold', color='navy')
        ax.set_xlabel("Value")
        ax.set_ylabel("Fuzzy Level")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        console.print(f"[bold blue]Saved plot to {save_path}[/bold blue]")
    plt.show()


def plot_stock_risk_analysis(df: pd.DataFrame, ticker: str, metrics: Dict[str, float], risk_score: float, save_path: Optional[str] = None):
    """
    Plots Stock Price, Drawdown Waterfall, and Volatility Comparison using Blue spectrum.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Stock Risk Analysis: {ticker} (Score: {risk_score:.1f}/100)", fontsize=14, fontweight='bold', color='navy')

    prices = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
    dates = df.index

    # 1. Price History
    axes[0].plot(dates, prices, color='royalblue', label='Close Price ($)')
    axes[0].set_ylabel("Price ($)")
    axes[0].set_title(f"Stock Price - {ticker}", color='navy')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')

    # 2. Drawdown Waterfall
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    axes[1].fill_between(dates, drawdown, 0, color='darkblue', alpha=0.4, label='Drawdown %')
    axes[1].set_ylabel("Drawdown %")
    axes[1].set_title(f"Drawdown Profile (Max Drop: {metrics.get('drawdown', 0.0)*100:.2f}%)", color='navy')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='lower left')

    # 3. Rolling Volatility
    log_rets = np.log(prices / prices.shift(1))
    rolling_hv = log_rets.rolling(21).std() * np.sqrt(252)
    axes[2].plot(dates, rolling_hv, color='navy', label='21-Day Historical Volatility (HV)')
    axes[2].axhline(y=metrics.get('implied_vol', 0.2), color='royalblue', linestyle='--', label=f"Implied Volatility ({metrics.get('implied_vol', 0.2)*100:.1f}%)")
    axes[2].set_ylabel("Volatility")
    axes[2].set_xlabel("Date")
    axes[2].set_title("Volatility (HV vs IV)", color='navy')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        console.print(f"[bold blue]Saved analysis chart to {save_path}[/bold blue]")
    plt.show()
