"""
Main CLI Application Interface for Stock Market Risk Prediction System.
Restricted styling palette: Standard Default and Blue accents only.
"""

import sys
import numpy as np
import pandas as pd

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from rich.console import Console
from rich.prompt import Prompt, FloatPrompt, IntPrompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.theme import Theme

custom_theme = Theme({
    "info": "blue",
    "warning": "blue",
    "danger": "blue",
    "prompt.choices": "blue",
    "prompt.default": "bold blue",
    "markdown.h1": "bold blue",
    "markdown.h2": "bold blue",
    "markdown.h3": "bold blue",
    "markdown.h4": "bold blue",
    "markdown.code": "blue",
    "markdown.link": "blue",
    "repr.number": "none",
    "repr.str": "none",
    "repr.path": "none",
    "table.header": "bold blue",
    "table.title": "bold blue"
})

console = Console(theme=custom_theme, soft_wrap=True, highlight=False)

from config.settings import DEFAULT_STOCKS, COMPANY_NAME_MAP
from data.data_loader import StockDataLoader, SyntheticStockGenerator
from data.feature_extractor import RiskFeatureExtractor
from models.deep_neuro_fuzzy_net import DeepNeuroFuzzyRiskModel
from models.genetic_optimizer import GeneticAlgorithmOptimizer
from utils.visualizer import (
    print_project_header,
    display_metrics_table,
    display_risk_prediction,
    plot_fuzzy_membership_functions,
    plot_stock_risk_analysis
)
from test_runner import run_all_tests


def resolve_company_info(user_input: str) -> tuple[str, str]:
    """
    Parses user input (e.g., 'Apple' or 'AAPL') into (ticker_symbol, full_company_name).
    """
    clean_input = user_input.strip()
    clean_upper = clean_input.upper()

    # Direct match in symbol map
    if clean_upper in COMPANY_NAME_MAP:
        symbol = clean_upper
        full_name = f"{COMPANY_NAME_MAP[symbol]} ({symbol})"
        return symbol, full_name

    # Name partial match
    for sym, name in COMPANY_NAME_MAP.items():
        if clean_input.lower() in name.lower() or sym.lower() in clean_input.lower():
            return sym, f"{name} ({sym})"

    # Fallback to uppercase input as symbol
    return clean_upper, f"Company ({clean_upper})"


class StockRiskApplication:

    def __init__(self):
        self.data_loader = StockDataLoader(use_offline_fallback=True)
        self.feature_extractor = RiskFeatureExtractor()
        self.model = DeepNeuroFuzzyRiskModel()

    def option_run_tests(self):
        """Option 1: Run Automated Test Suite."""
        run_all_tests()

    def option_analyze_ticker(self):
        """Option 2: Check Risk by Company Name."""
        console.print("\n[bold blue]Select Company / Stock Name:[/bold blue]")
        console.print("Sample Companies:")
        for company_label in DEFAULT_STOCKS:
            console.print(f"  - {company_label}")

        user_input = Prompt.ask("\nEnter Stock Symbol or Company Name", default="AAPL", console=console)
        symbol, full_company_name = resolve_company_info(user_input)

        try:
            df, is_synthetic = self.data_loader.fetch_stock_data(symbol, period="1y")
            
            # Extract Metrics
            metrics = self.feature_extractor.extract_metrics_from_df(df)
            display_metrics_table(metrics, title=f"Risk Values for {full_company_name}")

            # Predict Risk
            input_vec = self.feature_extractor.metrics_to_feature_vector(metrics)
            risk_score, category, pred_mdd, fuzzy_labels = self.model.predict_risk(input_vec)

            display_risk_prediction(risk_score, category, pred_mdd, fuzzy_labels)

            # Option to plot
            if Prompt.ask("View risk charts?", choices=["y", "n"], default="y", console=console) == "y":
                plot_stock_risk_analysis(df, full_company_name, metrics, risk_score)

        except Exception as e:
            console.print(f"[bold blue]Error analyzing {full_company_name}: {str(e)}[/bold blue]")

    def option_manual_input(self):
        """Option 3: Input Custom Risk Values."""
        console.print(Panel("[bold blue]MANUAL INPUT MODE[/bold blue]\n"
                            "Enter risk values manually to get risk prediction.", border_style="blue"))

        sharpe = FloatPrompt.ask("1. Sharpe Ratio", default=1.2, console=console)
        drawdown_pct = FloatPrompt.ask("2. Max Drawdown %", default=15.0, console=console)
        iv_pct = FloatPrompt.ask("3. Implied Volatility %", default=25.0, console=console)
        hv_pct = FloatPrompt.ask("4. Historical Volatility %", default=20.0, console=console)

        # Convert percentages to decimals
        drawdown = drawdown_pct / 100.0
        implied_vol = iv_pct / 100.0
        hist_vol = hv_pct / 100.0

        metrics = {
            'sharpe': sharpe,
            'drawdown': drawdown,
            'implied_vol': implied_vol,
            'hist_vol': hist_vol,
            'sortino': sharpe * 1.2,
            'var_95': hist_vol * 0.15
        }

        display_metrics_table(metrics, title="Custom Input Values")

        input_vec = np.array([sharpe, drawdown, implied_vol, hist_vol], dtype=np.float32)
        risk_score, category, pred_mdd, fuzzy_labels = self.model.predict_risk(input_vec)

        display_risk_prediction(risk_score, category, pred_mdd, fuzzy_labels)

    def option_run_ga_optimization(self):
        """Option 4: Run Genetic Algorithm."""
        console.print(Panel("[bold blue]GENETIC ALGORITHM OPTIMIZER[/bold blue]", border_style="blue"))
        
        pop_size = IntPrompt.ask("Population Size", default=15, console=console)
        generations = IntPrompt.ask("Number of Generations", default=10, console=console)

        # Generate sample validation data
        np.random.seed(42)
        X_val = np.random.uniform(0.05, 0.5, size=(30, 4))
        y_val = np.random.uniform(10, 90, size=30)

        ga = GeneticAlgorithmOptimizer(population_size=pop_size, generations=generations)
        best_chromo, history = ga.optimize(X_val, y_val)

        console.print(f"[bold blue]Optimized Weights: {best_chromo[24:28].round(4)}[/bold blue]")

    def option_view_fuzzy_curves(self):
        """Option 5: View Fuzzy Logic Curves."""
        console.print("[bold blue]Plotting Fuzzy Logic Membership Curves...[/bold blue]")
        plot_fuzzy_membership_functions()

    def option_view_about_docs(self):
        """Option 6: View Documentation."""
        try:
            with open("ABOUT.md", "r", encoding="utf-8") as f:
                content = f.read()
            console.print(Markdown(content))
        except Exception as e:
            console.print(f"[bold blue]Could not load documentation: {str(e)}[/bold blue]")

    def run_menu(self):
        """Main application execution loop."""
        print_project_header()

        while True:
            console.print("\n[bold blue]==================== MAIN MENU ====================[/bold blue]")
            console.print("[1] Run Tests")
            console.print("[2] Check Risk by Company Name")
            console.print("[3] Input Custom Risk Values (Sharpe, Drawdown, IV, HV)")
            console.print("[4] Run Genetic Algorithm")
            console.print("[5] View Fuzzy Logic Curves")
            console.print("[6] View Documentation (About)")
            console.print("[7] Exit")
            
            choice = Prompt.ask("\nSelect option [1-7]", choices=["1", "2", "3", "4", "5", "6", "7"], default="1", console=console)

            if choice == "1":
                self.option_run_tests()
            elif choice == "2":
                self.option_analyze_ticker()
            elif choice == "3":
                self.option_manual_input()
            elif choice == "4":
                self.option_run_ga_optimization()
            elif choice == "5":
                self.option_view_fuzzy_curves()
            elif choice == "6":
                self.option_view_about_docs()
            elif choice == "7":
                console.print("[bold blue]Exiting program. Goodbye![/bold blue]")
                sys.exit(0)


if __name__ == "__main__":
    app = StockRiskApplication()
    app.run_menu()
