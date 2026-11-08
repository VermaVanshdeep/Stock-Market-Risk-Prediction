"""
Main CLI Application Interface for Stock Market Risk Prediction System.
Restricted styling palette: Standard Default and Blue accents only.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

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
from utils.visualizer import (
    print_project_header,
    display_metrics_table,
    display_risk_prediction,
    plot_fuzzy_membership_functions,
    plot_stock_risk_analysis
)
from test_runner import run_all_tests
from generate_pdf_report import build_pdf_report


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
        """Option 4: Run Genetic Algorithm Optimizer.

        Uses optimize_model() which:
          - builds validation data from the trained model's feature space
          - runs the 24-gene GA (12 MF centres + 12 MF spreads)
          - applies best chromosome only when it beats the baseline
          - never exposes the held-out test split to the GA
        """
        console.print(Panel("[bold blue]GENETIC ALGORITHM OPTIMIZER[/bold blue]", border_style="blue"))

        pop_size   = IntPrompt.ask("Population Size",       default=15, console=console)
        generations = IntPrompt.ask("Number of Generations", default=10, console=console)

        from models.genetic_optimizer import optimize_model
        from models.train import build_dataset, TRAINING_UNIVERSE

        console.print("[bold blue]Building offline validation dataset…[/bold blue]")
        try:
            dataset = build_dataset(
                tickers=TRAINING_UNIVERSE[:6],
                period="5y",
                allow_network=True,
            )
        except Exception as e:
            console.print(f"[bold blue]Network fetch failed ({e}), using offline fallback.[/bold blue]")
            dataset = build_dataset(
                tickers=["OFFLINE_A", "OFFLINE_B", "OFFLINE_C"],
                allow_network=False,
            )

        console.print("[bold blue]Running GA optimisation…[/bold blue]")
        result = optimize_model(
            self.model,
            dataset,
            population_size=pop_size,
            generations=generations,
            seed=42,
        )

        meta      = result["metadata"]
        best_chromo = result["chromosome"]

        console.print(f"[bold blue]GA complete.[/bold blue]")
        console.print(f"  Baseline fitness : {meta['baseline_fitness']:.6f}")
        console.print(f"  Best fitness     : {meta['best_fitness']:.6f}")
        console.print(f"  Improved         : {meta['improved']}")
        console.print(f"  Chromosome genes : {len(best_chromo)} (12 MF centres + 12 MF spreads)")

        # Display the 12 optimised centres (genes 0-11) rounded for readability
        centres = best_chromo[:12].reshape(4, 3).round(4)
        spreads = best_chromo[12:24].reshape(4, 3).round(4)
        feature_labels = ["Sharpe", "Drawdown", "ImpliedVol", "HistVol"]
        console.print("[bold blue]Optimised Membership Centres (Low / Med / High):[/bold blue]")
        for i, label in enumerate(feature_labels):
            console.print(f"  {label:<12} {centres[i]}")
        console.print("[bold blue]Optimised Membership Spreads  (Low / Med / High):[/bold blue]")
        for i, label in enumerate(feature_labels):
            console.print(f"  {label:<12} {spreads[i]}")

    def option_run_hybrid_ga_gp(self):
        """Option 5: Run GA + GP Hybrid Pipeline."""
        from models.train import build_dataset, TRAINING_UNIVERSE
        from models.hybrid_pipeline import run_hybrid_pipeline
        from config.settings import GA_CONFIG
        from models.genetic_programming import GP_CONFIG

        console.print("\n[bold blue]Hybrid GA + GP Pipeline[/bold blue]")
        pop_ga  = IntPrompt.ask("GA population size",  default=GA_CONFIG["population_size"], console=console)
        gen_ga  = IntPrompt.ask("GA generations",      default=GA_CONFIG["generations"],      console=console)
        pop_gp  = IntPrompt.ask("GP population size",  default=30,  console=console)
        gen_gp  = IntPrompt.ask("GP generations",      default=10,  console=console)

        console.print("[bold blue]Building dataset…[/bold blue]")
        try:
            dataset = build_dataset(
                tickers=TRAINING_UNIVERSE[:6], period="5y", allow_network=True
            )
        except Exception as exc:
            console.print(f"[bold blue]Network failed ({exc}), using offline data.[/bold blue]")
            dataset = build_dataset(
                tickers=["OFFLINE_A", "OFFLINE_B", "OFFLINE_C"], allow_network=False
            )

        ga_cfg = dict(GA_CONFIG)
        ga_cfg["population_size"] = pop_ga
        ga_cfg["generations"]     = gen_ga

        gp_cfg = dict(GP_CONFIG)
        gp_cfg["population_size"] = pop_gp
        gp_cfg["generations"]     = gen_gp
        gp_cfg["max_depth"]       = 5
        gp_cfg["max_nodes"]       = 63

        console.print("[bold blue]Running GA → GP pipeline…[/bold blue]")
        result = run_hybrid_pipeline(
            self.model, dataset,
            ga_config=ga_cfg, gp_config=gp_cfg,
            seed=42,
            ga_progress=lambda g, t, f: console.print(
                f"  GA gen {g:3d}/{t}  fitness={f:.6f}") if g == t or g % 5 == 0 else None,
            gp_progress=lambda g, t, f: console.print(
                f"  GP gen {g:3d}/{t}  val_fit={f:.6f}") if g == t or g % 5 == 0 else None,
        )

        console.print("\n" + result.summary())
        expr_str = self.model.gp_expression_string
        if expr_str != "(none)":
            console.print(f"[bold blue]GP expression attached to model: YES[/bold blue]")
            console.print(f"  {expr_str[:120]}")
        else:
            console.print("[bold blue]GP expression: not attached[/bold blue]")

    def option_view_fuzzy_curves(self):
        """Option 6: View Fuzzy Logic Curves."""
        console.print("[bold blue]Plotting Fuzzy Logic Membership Curves...[/bold blue]")
        plot_fuzzy_membership_functions()

    def option_view_about_docs(self):
        """Option 7: View Documentation."""
        try:
            about_path = Path(__file__).resolve().parent / "ABOUT.md"
            with open(about_path, "r", encoding="utf-8") as f:
                content = f.read()
            console.print(Markdown(content))
        except Exception as e:
            console.print(f"[bold blue]Could not load documentation: {str(e)}[/bold blue]")

    def option_generate_pdf_report(self):
        """Option 7: Generate PDF Report."""
        console.print("[bold blue]Generating PDF Research Report...[/bold blue]")
        build_pdf_report()
        console.print("[bold blue]PDF Report generated: Stock_Market_Risk_Prediction_Report.pdf[/bold blue]")

    def run_menu(self):
        """Main application execution loop."""
        print_project_header()

        while True:
            console.print("\n[bold blue]==================== MAIN MENU ====================[/bold blue]")
            console.print("[1] Run Tests")
            console.print("[2] Check Risk by Company Name")
            console.print("[3] Input Custom Risk Values (Sharpe, Drawdown, IV, HV)")
            console.print("[4] Run Genetic Algorithm (GA only)")
            console.print("[5] Run GA + GP Hybrid Pipeline")
            console.print("[6] View Fuzzy Logic Curves")
            console.print("[7] View Documentation (About)")
            console.print("[8] Generate PDF Research Report")
            console.print("[9] Exit")
            
            choice = Prompt.ask(
                "\nSelect option [1-9]",
                choices=["1","2","3","4","5","6","7","8","9"],
                default="1", console=console
            )

            if choice == "1":
                self.option_run_tests()
            elif choice == "2":
                self.option_analyze_ticker()
            elif choice == "3":
                self.option_manual_input()
            elif choice == "4":
                self.option_run_ga_optimization()
            elif choice == "5":
                self.option_run_hybrid_ga_gp()
            elif choice == "6":
                self.option_view_fuzzy_curves()
            elif choice == "7":
                self.option_view_about_docs()
            elif choice == "8":
                self.option_generate_pdf_report()
            elif choice == "9":
                console.print("[bold blue]Exiting program. Goodbye![/bold blue]")
                sys.exit(0)


if __name__ == "__main__":
    app = StockRiskApplication()
    app.run_menu()
