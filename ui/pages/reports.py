"""
Report centre: generate, preview and download the PDF research report, plus a
quick view of the report already sitting on disk.
"""

import datetime as _dt

import streamlit as st

from ui import services
from ui.components import callout, page_header, render_cards, section_title


def _file_summary(path) -> dict:
    stat = path.stat()
    return {
        "size_kb": stat.st_size / 1024.0,
        "modified": _dt.datetime.fromtimestamp(stat.st_mtime),
    }


def render() -> None:
    page_header(
        "Export",
        "Research Report Centre",
        "Compile the full technical write-up — ANFIS and LSTM formulations, literature "
        "review, architecture table and verification results — into a print-ready PDF.",
    )

    existing = services.ROOT / services.PDF_FILENAME
    if existing.exists():
        info = _file_summary(existing)
        render_cards(
            [
                {
                    "eyebrow": "Report on disk",
                    "value": f"{info['size_kb']:.0f} KB",
                    "hint": f"Last generated {info['modified']:%d %b %Y, %H:%M}",
                },
                {
                    "eyebrow": "Format",
                    "value": "PDF · Letter",
                    "hint": "Numbered pages with running headers and footers.",
                },
                {
                    "eyebrow": "Palette",
                    "value": "Monochrome",
                    "hint": "Formal black-and-white print styling.",
                },
            ],
            columns=3,
        )

    st.write("")
    cols = st.columns([1.5, 1.5, 2], gap="medium")
    with cols[0]:
        generate = st.button("Generate report", type="primary", width="stretch")
    with cols[1]:
        save_disk = st.button("Save to project folder", width="stretch")

    if save_disk:
        try:
            with st.spinner("Building PDF on disk…"):
                path = services.build_pdf_on_disk()
            st.success(f"Saved to `{path}`")
        except ImportError:
            st.error("The `reportlab` package is required — install it with "
                     "`pip install reportlab`.")
        except Exception as exc:
            st.error(f"Could not generate the report — {exc}")

    if generate:
        try:
            with st.spinner("Compiling the research report…"):
                data = services.build_pdf_bytes()
            st.session_state["pdf_bytes"] = data
        except ImportError:
            st.error("The `reportlab` package is required — install it with "
                     "`pip install reportlab`.")
        except Exception as exc:
            st.error(f"Could not generate the report — {exc}")

    data = st.session_state.get("pdf_bytes")
    if data:
        st.success(f"Report compiled — {len(data) / 1024:.0f} KB ready for download.")
        st.download_button(
            "Download PDF report",
            data=data,
            file_name=services.PDF_FILENAME,
            mime="application/pdf",
            type="primary",
            width="stretch",
        )

    st.divider()
    section_title("What's inside", "Sections compiled into the document.")

    callout(
        "<b>1. Introduction</b> — market risk problem framing · "
        "<b>2. Literature analysis</b> — ANFIS (Jang, 1993) and LSTM "
        "(Hochreiter &amp; Schmidhuber, 1997) full reviews with equation boxes · "
        "<b>3. Research gaps</b> and the proposed hybrid approach · "
        "<b>4. System architecture</b> — module-by-module responsibility table · "
        "<b>5. Formulations</b> — Sharpe, MDD, Garman-Klass IV, HV, GA fitness · "
        "<b>6. Verification</b> — automated test results · "
        "<b>7. Conclusion &amp; future scope</b>."
    )

    st.caption(
        "Reports are generated locally — nothing is uploaded. The CLI equivalent is "
        "menu option 7 in `python main.py`."
    )
