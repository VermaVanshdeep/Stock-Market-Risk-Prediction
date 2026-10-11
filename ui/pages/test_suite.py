"""
Automated test suite page: run the full unit + integration suite and surface the
results in a readable, structured form.
"""

from typing import Dict, List

import pandas as pd
import streamlit as st

from ui import services
from ui.components import callout, page_header, render_cards, section_title


def _details_table(items: List[Dict[str, str]]) -> None:
    if not items:
        return
    st.dataframe(
        pd.DataFrame([{"Test": i["test"], "Outcome": "FAILED"} for i in items]),
        hide_index=True,
    )
    for item in items:
        with st.expander(item["test"]):
            st.code(item["traceback"], language="text")


def render() -> None:
    page_header(
        "Verification",
        "Automated Test Suite",
        "Executes every unit and integration test — metric maths, synthetic data "
        "generation, fuzzy forward passes, GA evolution and the end-to-end pipeline — "
        "without any terminal clutter.",
    )

    st.caption("Equivalent CLI command: `python test_runner.py`")

    cols = st.columns([1.6, 1, 1.6], gap="medium")
    with cols[0]:
        st.markdown("")
        run = st.button("Run full test suite", type="primary", width="stretch")
    with cols[1]:
        st.markdown("")
        clear = st.button("Clear results", width="stretch")
    with cols[2]:
        st.markdown("")
        st.caption(
            "Results stay on this page while you keep exploring other sections."
        )

    if clear:
        st.session_state.pop("test_report", None)
        st.rerun()

    if run:
        with st.spinner("Discovering and executing test modules…"):
            try:
                report = services.run_tests()
            except Exception as exc:
                st.error(f"Test runner failed to start — {exc}")
                report = None
        if report is not None:
            st.session_state["test_report"] = report

    report = st.session_state.get("test_report")
    if not report:
        callout(
            "No results yet — press <b>Run full test suite</b>. The suite is read-only "
            "and never mutates model or data state."
        )
        return

    total = int(report["tests_run"]) or 1
    passed = int(report["passed"])
    failures = int(report["failures"])
    errors = int(report["errors"])
    pass_rate = passed / total * 100.0

    st.divider()
    section_title("Results summary", "Latest suite execution on this machine.")

    if report["success"]:
        st.success(f"All {passed} checks passed — the pipeline is operational.")
    else:
        st.error(
            f"{failures} failure(s) and {errors} error(s) across {total} tests. "
            "Expand the details below for tracebacks."
        )

    st.progress(min(pass_rate / 100.0, 1.0), text=f"Pass rate {pass_rate:.1f}%")

    render_cards(
        [
            {"eyebrow": "Tests run", "value": str(total), "hint": "Discovered test methods executed."},
            {"eyebrow": "Passed", "value": str(passed), "hint": "Assertions satisfied on the first attempt."},
            {"eyebrow": "Failures", "value": str(failures), "hint": "Assertions that returned an unexpected value."},
            {"eyebrow": "Errors", "value": str(errors), "hint": "Tests that raised an unexpected exception."},
            {"eyebrow": "Pass rate", "value": f"{pass_rate:.1f}%", "hint": "Passed divided by total tests run."},
            {"eyebrow": "Status", "value": "PASS" if report["success"] else "FAIL",
             "hint": "Overall suite verdict."},
        ],
        columns=3,
    )

    st.divider()
    tab_fail, tab_log = st.tabs(["Details", "Raw runner log"])

    with tab_fail:
        if failures or errors:
            if report["failure_details"]:
                st.markdown("**Assertion failures**")
                _details_table(report["failure_details"])
            if report["error_details"]:
                st.markdown("**Execution errors**")
                _details_table(report["error_details"])
        else:
            st.info("No failures or errors to report.")

    with tab_log:
        log = report.get("log") or "(the runner suppresses output for successful runs)"
        st.code(log, language="text")
