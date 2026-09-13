"""
Automated Test Runner for Stock Market Risk Prediction System.
Runs all unit and integration test suites quietly, returning a clean, concise diagnostics report.
Restricted styling palette: Standard Default and Blue accents only.
"""

import io
import sys
import unittest
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.theme import Theme

custom_theme = Theme({
    "info": "blue",
    "warning": "blue",
    "danger": "blue",
    "repr.number": "none",
    "repr.str": "none",
    "repr.path": "none",
    "table.header": "bold blue",
    "table.title": "bold blue"
})

console = Console(theme=custom_theme, soft_wrap=True, highlight=False)


def run_all_tests() -> bool:
    """
    Discovers and runs all tests quietly without terminal clutter.
    Returns True if all tests passed, False otherwise.
    """
    console.print(Panel("[bold blue]AUTOMATED SYSTEM TEST SUITE[/bold blue]", border_style="blue"))

    tests_dir = Path(__file__).resolve().parent / "tests"
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(tests_dir), pattern="test_*.py")

    suppress_buffer = io.StringIO()
    runner = unittest.TextTestRunner(stream=suppress_buffer, verbosity=0)
    
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = suppress_buffer, suppress_buffer
        result = runner.run(suite)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    table = Table(title="Test Results Summary", show_header=True, header_style="bold blue", title_style="bold blue", border_style="dim")
    table.add_column("Item", style="blue")
    table.add_column("Count", justify="right", style="bold white")
    table.add_column("Status", justify="center", style="bold blue")

    table.add_row("Total Tests Run", str(result.testsRun), "DONE")
    table.add_row("Passed Tests", str(result.testsRun - len(result.failures) - len(result.errors)), "PASS")
    table.add_row("Failures", str(len(result.failures)), "ZERO" if len(result.failures) == 0 else "FAILURES")
    table.add_row("Errors", str(len(result.errors)), "ZERO" if len(result.errors) == 0 else "ERRORS")

    console.print(table)

    if result.wasSuccessful():
        console.print(Panel("[bold blue]ALL TESTS PASSED SUCCESSFULLY![/bold blue]", border_style="blue"))
        return True
    else:
        console.print(Panel("[bold white]TESTS FAILED - SEE LOG BELOW[/bold white]", border_style="blue"))
        if result.failures or result.errors:
            console.print(f"[bold blue]Error Log:[/bold blue]\n{suppress_buffer.getvalue()}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
