"""
Assembles the full equal-weight portfolio report: strategy recap, process
diagram (JPG), quantstats-style results, scienceplots-rendered equity/
drawdown/correlation charts (PNGs from generate_equal_weight_charts.py,
built in the base env since matplotlib crashes under adonys), and a
per-symbol performance table.

Light/paper theme (white background) to match the scienceplots aesthetic of
the embedded charts, unlike the dark-themed reportlab-native charts used
elsewhere this session -- deliberate choice to keep the whole report visually
coherent with images it can't itself override the styling of.

Usage:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/build_equal_weight_report.py
"""
import os
import pickle
import sys

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                 Spacer, Table, TableStyle)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PKL = os.path.join(RESEARCH_DIR, "data", "equal_weight_report_data.pkl")
CHARTS_DIR = os.path.join(RESEARCH_DIR, "reports", "equal_weight_full")
DIAGRAM_JPG = os.path.join(RESEARCH_DIR, "reports", "regression_gate_diagram.jpg")
OUT_PDF = os.path.join(RESEARCH_DIR, "reports", "equal_weight_full", "equal_weight_portfolio_report.pdf")

NAVY = HexColor("#1a2744")
ACCENT = HexColor("#2f6ca8")
MUTED = HexColor("#5a6472")
LIGHT_GRID = HexColor("#e4e7ec")
GREEN = HexColor("#1b7a3d")
RED = HexColor("#b3221a")

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=NAVY, fontSize=20, spaceAfter=4)
subtitle_style = ParagraphStyle("SubtitleX", parent=styles["Normal"], textColor=MUTED, fontSize=10.5, spaceAfter=18)
section_style = ParagraphStyle("SectionX", parent=styles["Heading1"], textColor=NAVY, fontSize=14.5, spaceBefore=16, spaceAfter=8)
subsection_style = ParagraphStyle("SubsectionX", parent=styles["Heading2"], textColor=ACCENT, fontSize=11.5, spaceBefore=10, spaceAfter=6)
body_style = ParagraphStyle("BodyX", parent=styles["Normal"], fontSize=9.7, leading=14, alignment=TA_JUSTIFY, spaceAfter=8)
caption_style = ParagraphStyle("CaptionX", parent=styles["Normal"], fontSize=8.3, textColor=MUTED, leading=11, spaceAfter=10, spaceBefore=2)


def table_style(header_bg=NAVY, n_cols=2):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#f4f6f9")]),
        ("GRID", (0, 0), (-1, -1), 0.4, LIGHT_GRID),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])


def fitted_image(path, max_w, max_h):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(path, width=w * scale, height=h * scale)


def main():
    with open(DATA_PKL, "rb") as f:
        d = pickle.load(f)
    m = d["metrics"]
    per_symbol_df = d["per_symbol_df"]

    doc = SimpleDocTemplate(OUT_PDF, pagesize=letter,
                             leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    story = []

    # ---- Title ----
    story.append(Paragraph("MetaFVG &mdash; Spearman Mean-Reversion Gate", title_style))
    story.append(Paragraph(
        "Equal-Weight Portfolio Report &mdash; 14 Instruments &mdash; M15/4h &mdash; Fixed 2% Notional Sizing",
        subtitle_style))

    # ---- Strategy recap ----
    story.append(Paragraph("Strategy Recap", section_style))
    story.append(Paragraph(
        "MetaFVG trades Fair Value Gaps detected on a higher timeframe (HTF, 4h resample) and confirmed with "
        "lower-timeframe (LTF, M15) momentum before entry. For each HTF FVG zone, price re-entering the gap "
        "with a Fibonacci-style fill percentage triggers a search for a strong LTF momentum candle pattern in "
        "the trigger direction; on confirmation, a trade is opened with a pivot-based stop loss and an "
        "ATR-scaled take profit.", body_style))
    story.append(Paragraph(
        "This report covers the <b>mean-reversion variant</b>: a rolling Spearman rank-correlation gate is "
        "computed on HTF closes, and whenever the gate's sign <i>disagrees</i> with the HTF zone's own direction "
        "(<tt>invert_direction=True</tt>), the trade is taken in the gate's direction instead of the zone's "
        "&mdash; both the LTF momentum confirmation and the trade parameters follow the disagreeing signal, not "
        "the original zone. Gate threshold: rolling window 20 bars, |&rho;| gate acceptance as calibrated in "
        "the FX Majors sweep.", body_style))
    story.append(Paragraph(
        "The 14-instrument universe (AUDUSD, EURUSD, GBPUSD, NZDUSD, USDCAD, USDCHF, USDJPY, US500, GER40, "
        "JP225, HK50, USDSGD, EURNOK, USDZAR) was deliberately assembled to be fundamentally diversified across "
        "central banks and macro drivers, not just statistically decorrelated &mdash; four equity indices from "
        "different central banks plus three FX minors add currency exposures with no recombination of the 7 "
        "FX Majors. This report blends all 14 with <b>equal weights (1/14 each)</b>; a companion study tested "
        "four non-equal-weight allocation schemes and found none improved on equal weighting's Sharpe (see "
        "<tt>weighting_schemes/comparison_summary.pdf</tt>).", body_style))

    story.append(PageBreak())

    # ---- Process diagram ----
    story.append(Paragraph("Process Diagram", section_style))
    story.append(Paragraph(
        "HTF FVG detection &rarr; LTF momentum confirmation &rarr; regression/Spearman/Lasso gate &rarr; "
        "trade parameter calculation &rarr; pivot SL / ATR TP.", caption_style))
    story.append(fitted_image(DIAGRAM_JPG, 6.6 * inch, 8.6 * inch))

    story.append(PageBreak())

    # ---- Portfolio results ----
    story.append(Paragraph("Portfolio Results", section_style))
    metrics_table_data = [
        ["Metric", "Value"],
        ["Total Return", f"{d['total_return_pct']:+.3f}%"],
        ["CAGR", f"{m['qs_cagr']*100:+.4f}%"],
        ["Sharpe (daily, ann.)", f"{m['qs_sharpe']:.3f}"],
        ["Sortino", f"{m['qs_sortino']:.3f}"],
        ["Calmar", f"{m['qs_calmar']:.3f}"],
        ["Max Drawdown", f"{m['qs_max_drawdown_pct']:.4f}%"],
        ["Annualized Volatility", f"{m['qs_volatility_pct']:.4f}%"],
        ["Win Rate (days)", f"{d['win_days_pct']:.2f}%"],
        ["Best Day", f"{d['best_day_pct']:+.4f}%"],
        ["Worst Day", f"{d['worst_day_pct']:+.4f}%"],
        ["Skew", f"{m['qs_skew']:.3f}"],
        ["Tail Ratio", f"{m['qs_tail_ratio']:.3f}"],
        ["Kelly Criterion", f"{m['qs_kelly']:.4f}"],
        ["VaR (95%)", f"{m['qs_var_pct']:.4f}%"],
        ["CVaR (95%)", f"{m['qs_cvar_pct']:.4f}%"],
        ["Recovery Factor", f"{m['qs_recovery_factor']:.3f}"],
    ]
    t = Table(metrics_table_data, colWidths=[2.6 * inch, 2.0 * inch], hAlign="LEFT")
    t.setStyle(table_style())
    story.append(t)

    story.append(Spacer(1, 14))
    story.append(Paragraph("Equity Curve", subsection_style))
    story.append(fitted_image(os.path.join(CHARTS_DIR, "equity_curve.png"), 6.6 * inch, 3.6 * inch))

    story.append(Paragraph("Underwater Plot", subsection_style))
    story.append(fitted_image(os.path.join(CHARTS_DIR, "drawdown.png"), 6.6 * inch, 2.9 * inch))

    story.append(PageBreak())

    # ---- Rolling correlation ----
    story.append(Paragraph("Rolling Correlation Between Assets", section_style))
    story.append(Paragraph(
        "Average pairwise Pearson correlation across the 14 instruments' daily strategy P&amp;L, computed on "
        "a rolling 180-calendar-day window stepped every 14 days (minimum 15 overlapping observations per "
        "pair). This measures realized co-movement of the <i>traded</i> P&amp;L streams, not the underlying "
        "instruments' raw price returns &mdash; the latter are structurally far more correlated (e.g. the 7 FX "
        "Majors alone run 0.3&ndash;0.9 in absolute price-return correlation via a common USD factor); sparse, "
        "largely non-overlapping FVG trade timing across instruments is what keeps the traded correlation this "
        "low, on top of the fundamental diversification built into instrument selection.", body_style))
    story.append(fitted_image(os.path.join(CHARTS_DIR, "rolling_correlation.png"), 6.6 * inch, 3.0 * inch))
    story.append(Spacer(1, 10))
    story.append(fitted_image(os.path.join(CHARTS_DIR, "correlation_heatmap.png"), 6.0 * inch, 5.4 * inch))

    story.append(PageBreak())

    # ---- Per-symbol performance ----
    story.append(Paragraph("Per-Symbol Performance", section_style))
    story.append(Paragraph(
        "Standalone performance of each instrument's own trade stream (fixed 2% notional sizing), sorted by "
        "Sharpe. Equal weighting means each instrument contributes 1/14 of portfolio capital regardless of its "
        "standalone Sharpe.", caption_style))
    sym_table_data = [["Symbol", "Closed Trades", "Win Rate", "Sharpe", "Max DD", "Total Return"]]
    for _, row in per_symbol_df.iterrows():
        sym_table_data.append([
            row["symbol"],
            f"{int(row['closed_trades'])}",
            f"{row['win_rate_pct']:.1f}%",
            f"{row['sharpe']:.3f}",
            f"{row['max_dd_pct']:.2f}%",
            f"{row['total_return_pct']:+.3f}%",
        ])
    t2 = Table(sym_table_data, colWidths=[0.9 * inch, 1.1 * inch, 0.9 * inch, 0.8 * inch, 0.9 * inch, 1.1 * inch], hAlign="LEFT")
    style2 = table_style()
    for i, row in per_symbol_df.iterrows():
        color = GREEN if row["sharpe"] >= 0 else RED
        style2.add("TEXTCOLOR", (3, i + 1), (3, i + 1), color)
    t2.setStyle(style2)
    story.append(t2)

    def on_page(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.75 * inch, 0.45 * inch, "MetaFVG -- Spearman mean-reversion gate -- equal-weight portfolio report")
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.45 * inch, f"Page {_doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"written: {OUT_PDF}")


if __name__ == "__main__":
    main()
