"""
SafeGuard AI — Compliance Report Generator
Produces PDF and CSV safety reports from incident log data.
"""

import os
import pandas as pd
from datetime import datetime
from fpdf import FPDF


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def generate_csv(df: pd.DataFrame, output_path: str = None) -> str:
    os.makedirs("reports", exist_ok=True)
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/safety_report_{ts}.csv"
    df.to_csv(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class _SafetyPDF(FPDF):
    def header(self):
        # Dark navy bar
        self.set_fill_color(15, 20, 60)
        self.rect(0, 0, 210, 26, "F")

        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 212, 255)
        self.set_xy(10, 6)
        self.cell(0, 14, "SafeGuard AI  |  PPE Safety Compliance Report", ln=False)

        self.set_font("Helvetica", "", 8)
        self.set_text_color(180, 190, 220)
        self.set_xy(10, 18)
        self.cell(
            0, 6,
            f"Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
            ln=True,
        )
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 140, 160)
        self.cell(0, 10, f"Page {self.page_no()} / {{nb}}", align="C")


def generate_pdf(
    df: pd.DataFrame,
    stats: dict,
    output_path: str = None,
) -> str:
    os.makedirs("reports", exist_ok=True)
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/safety_report_{ts}.pdf"

    pdf = _SafetyPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ── Summary section ──────────────────────────────────────────────
    _section_title(pdf, "Summary Statistics")

    summary_data = [
        ("Total Incidents Logged",      stats.get("total_incidents", 0)),
        ("Total Workers Monitored",     stats.get("total_workers",   0)),
        ("Total Violations Detected",   stats.get("total_violations", 0)),
        ("Average Compliance Rate",    f"{stats.get('avg_compliance', 100.0):.1f} %"),
        ("Today's Incidents",           stats.get("today_incidents",  0)),
        ("Today's Violations",          stats.get("today_violations", 0)),
    ]

    for label, value in summary_data:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(50, 60, 100)
        pdf.cell(90, 8, label, border="B")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(20, 30, 80)
        pdf.cell(0, 8, str(value), border="B", ln=True)

    pdf.ln(8)

    # ── Compliance gauge bar ─────────────────────────────────────────
    compliance = stats.get("avg_compliance", 100.0)
    _draw_compliance_bar(pdf, compliance)
    pdf.ln(10)

    # ── Incident table ────────────────────────────────────────────────
    if not df.empty:
        _section_title(pdf, "Incident Log (latest 50 entries)")

        cols    = ["timestamp", "violation_type", "worker_count", "violation_count", "compliance_rate", "location"]
        headers = ["Timestamp",  "Violation Type", "Workers",      "Violations",      "Compliance %",    "Location"]
        widths  = [42,           48,                22,             22,                28,                28]

        # Table header
        pdf.set_fill_color(15, 20, 60)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)
        for h, w in zip(headers, widths):
            pdf.cell(w, 9, h, border=1, fill=True)
        pdf.ln()

        # Rows
        pdf.set_font("Helvetica", "", 8)
        for i, (_, row) in enumerate(df.head(50).iterrows()):
            fill = i % 2 == 0
            pdf.set_fill_color(235, 240, 255) if fill else pdf.set_fill_color(255, 255, 255)
            viol_rate = float(row.get("compliance_rate", 100))
            pdf.set_text_color(180, 0, 0) if viol_rate < 70 else pdf.set_text_color(30, 30, 30)

            for col, w in zip(cols, widths):
                val = str(row.get(col, ""))
                if col == "compliance_rate":
                    try:
                        val = f"{float(val):.1f} %"
                    except Exception:
                        pass
                pdf.cell(w, 7, val[:22], border=1, fill=True)
            pdf.ln()

    pdf.output(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section_title(pdf: FPDF, title: str):
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 20, 60)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_draw_color(0, 180, 255)
    pdf.set_line_width(0.6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(5)


def _draw_compliance_bar(pdf: FPDF, compliance: float):
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 80)
    pdf.cell(0, 8, f"Overall Compliance Rate: {compliance:.1f}%", ln=True)

    bar_x, bar_y, bar_w, bar_h = 10, pdf.get_y(), 190, 10
    # background
    pdf.set_fill_color(220, 220, 240)
    pdf.rect(bar_x, bar_y, bar_w, bar_h, "F")
    # fill
    fill_w = bar_w * compliance / 100
    if compliance >= 80:
        pdf.set_fill_color(0, 200, 100)
    elif compliance >= 60:
        pdf.set_fill_color(255, 180, 0)
    else:
        pdf.set_fill_color(220, 40, 40)
    pdf.rect(bar_x, bar_y, fill_w, bar_h, "F")

    pdf.ln(bar_h + 2)
