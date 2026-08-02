"""
Per-pair (univariate) + whole-portfolio "quantstats-style" tearsheets for FX
Majors, one subfolder per config (Baseline, Regression Gate, Regression Gate
+ Risk Sizing, Spearman Gate, Spearman Gate + Risk Sizing, Spearman Gate
(0.82), Spearman Gate (tight) -- whatever's cached in fx_majors_portfolio_
blend.pkl, all post lookahead-bias fix). Each config's subfolder gets one PDF
per pair plus one for the equal-weighted 7-pair blended portfolio, in
metalib/research/reports/quantstats_fx_majors/<config-slug>/.

Why not quantstats.reports.html() directly: it (and even the plot-free
qs.reports.metrics(mode="full")) crashes natively in this env (exit 127,
same signature as the documented BLAS/matplotlib crash class -- confirmed by
smoke test before writing this). The individual qs.stats.* functions already
used safely all session (metafvg_ab_sweep.py's quantstats_metrics) don't hit
it. So: pull metrics from that same safe subset, render charts as native
reportlab vector Drawings (house pattern used for every other PDF chart this
session, for the same underlying reason).

Usage:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/generate_quantstats_fx_majors_reports.py
"""
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metafvg_ab_sweep import quantstats_metrics  # noqa: E402

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(RESEARCH_DIR, "data")
OUT_DIR = os.path.join(RESEARCH_DIR, "reports", "quantstats_fx_majors")
BLEND_PKL = os.path.join(DATA_DIR, "fx_majors_portfolio_blend.pkl")

FX_MAJORS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]

STYLE = {
    "bg": "#0f1117", "panel": "#1a1d27", "green": "#00c853", "red": "#ff3d3d",
    "blue": "#4a9eff", "orange": "#ff9800", "purple": "#ab47bc",
    "text": "#e0e0e0", "muted": "#666677",
}
BG = HexColor(STYLE["bg"])
PANEL = HexColor(STYLE["panel"])
GREEN = HexColor(STYLE["green"])
RED = HexColor(STYLE["red"])
BLUE = HexColor(STYLE["blue"])
TEXT = HexColor(STYLE["text"])
MUTED = HexColor(STYLE["muted"])
ACCENT = HexColor("#2a2d3a")
W, H = A4

styles = getSampleStyleSheet()
title_style = ParagraphStyle("Title", parent=styles["Normal"], fontSize=24, leading=28, textColor=TEXT,
                              alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4)
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11, leading=14, textColor=MUTED,
                                 alignment=TA_CENTER, fontName="Helvetica", spaceAfter=12)
section_style = ParagraphStyle("Section", parent=styles["Normal"], fontSize=13, leading=16, textColor=BLUE,
                                fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)


def base_table_style(extra=None):
    s = [
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PANEL, ACCENT]),
        ("GRID", (0, 0), (-1, -1), 0.3, ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
    ]
    if extra:
        s.extend(extra)
    return TableStyle(s)


def line_chart_drawing(title, series, width=460, height=190, n_x_labels=8, colors_cycle=None):
    """Native reportlab vector chart -- no matplotlib, sidesteps this env's
    canvas.draw()/savefig() crash entirely (see module docstring)."""
    colors_cycle = colors_cycle or [HexColor(STYLE["green"]), HexColor(STYLE["blue"])]
    target_points = 80

    d = Drawing(width, height)
    d.add(String(2, height - 14, title, fillColor=TEXT, fontSize=10.5, fontName="Helvetica-Bold"))

    lc = HorizontalLineChart()
    lc.x = 42
    lc.y = 18
    lc.width = width - 60
    lc.height = height - 40

    plotted = []
    labels = None
    for label, s in series.items():
        s = s.dropna()
        if len(s) == 0:
            continue
        step = max(1, len(s) // target_points)
        ds = s.iloc[::step]
        plotted.append((label, ds))
        if labels is None or len(ds) > len(labels):
            labels = ds.index

    if not plotted:
        d.add(String(width / 2 - 60, height / 2, "no data", fillColor=MUTED, fontSize=9))
        return d

    n = len(labels)
    label_every = max(1, n // n_x_labels)
    cat_names = [labels[i].strftime("%b'%y") if i % label_every == 0 else "" for i in range(n)]

    lc.data = []
    for (label, ds), color in zip(plotted, colors_cycle):
        vals = list(ds.values)
        if len(vals) < n:
            vals = vals + [vals[-1]] * (n - len(vals))
        lc.data.append(vals[:n])

    all_vals = [v for _, ds in plotted for v in ds.values]
    vmin, vmax = min(all_vals), max(all_vals)
    # Padding must scale with the *actual* data range, not a fixed % of the value level --
    # an equity curve near 1.0 with <1% real variation would otherwise get a fixed ~1%
    # padding (abs(vmax)*0.01) that swamps the real range and squashes the curve into a
    # sliver in the middle of a mostly-empty axis. Only fall back to an absolute minimum
    # for the genuinely-degenerate flat-line case (data_range ~ 0).
    data_range = vmax - vmin
    pad = data_range * 0.08 if data_range > 1e-9 else max(abs(vmax) * 0.01, 1e-6)

    lc.categoryAxis.categoryNames = cat_names
    lc.categoryAxis.labels.fontSize = 6.5
    lc.categoryAxis.labels.fillColor = MUTED
    lc.categoryAxis.strokeColor = ACCENT
    lc.valueAxis.strokeColor = ACCENT
    lc.valueAxis.labels.fontSize = 6.5
    lc.valueAxis.labels.fillColor = MUTED
    lc.valueAxis.valueMin = vmin - pad
    lc.valueAxis.valueMax = vmax + pad
    lc.joinedLines = 1
    for i, _ in enumerate(plotted):
        lc.lines[i].strokeColor = colors_cycle[i % len(colors_cycle)]
        lc.lines[i].strokeWidth = 1.2

    d.add(lc)
    return d


def monthly_returns_table(rets: pd.Series):
    """Year x month % return grid, cell background color-coded by sign/magnitude
    -- a heatmap built from Table cell backgrounds, not a rasterized image."""
    monthly = (1 + rets).resample("ME").prod() - 1
    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    header = [""] + month_names
    rows = [header]
    cell_colors = []
    max_abs = float(np.nanmax(np.abs(pivot.values))) if np.isfinite(pivot.values).any() else 1.0
    max_abs = max(max_abs, 1e-6)

    for year, row in pivot.iterrows():
        line = [str(year)]
        for v in row.values:
            line.append("" if pd.isna(v) else f"{v*100:+.1f}%")
        rows.append(line)
        row_colors = []
        for v in row.values:
            if pd.isna(v):
                row_colors.append(PANEL)
            else:
                intensity = min(abs(v) / max_abs, 1.0)
                base = GREEN if v >= 0 else RED
                row_colors.append(_blend(PANEL, base, 0.15 + 0.45 * intensity))
        cell_colors.append(row_colors)

    tbl = Table(rows, colWidths=[34] + [30] * 12)
    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.3, ACCENT),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for r, row_colors in enumerate(cell_colors, start=1):
        for c, col in enumerate(row_colors, start=1):
            ts.append(("BACKGROUND", (c, r), (c, r), col))
    tbl.setStyle(TableStyle(ts))
    return tbl


def _blend(c1: HexColor, c2: HexColor, t: float) -> HexColor:
    r = c1.red + (c2.red - c1.red) * t
    g = c1.green + (c2.green - c1.green) * t
    b = c1.blue + (c2.blue - c1.blue) * t
    return HexColor((int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255))


def build_report(name: str, rets: pd.Series, out_path: str, subtitle: str, extra_flowables=None):
    equity = (1 + rets).cumprod()
    drawdown = equity / equity.cummax() - 1
    m = quantstats_metrics(equity)

    total_return_pct = (equity.iloc[-1] - 1) * 100
    win_days_pct = (rets > 0).mean() * 100
    avg_daily_pct = rets.mean() * 100
    n_days = len(rets)

    metrics_rows = [
        ["Metric", "Value"],
        ["Total Return", f"{total_return_pct:+.2f}%"],
        ["CAGR", f"{m.get('qs_cagr', float('nan'))*100:+.3f}%"],
        ["Sharpe (daily, ann.)", f"{m.get('qs_sharpe', float('nan')):.3f}"],
        ["Sortino", f"{m.get('qs_sortino', float('nan')):.3f}"],
        ["Calmar", f"{m.get('qs_calmar', float('nan')):.3f}"],
        ["Max Drawdown", f"{m.get('qs_max_drawdown_pct', float('nan')):.2f}%"],
        ["Volatility (ann.)", f"{m.get('qs_volatility_pct', float('nan')):.2f}%"],
        ["Ulcer Index", f"{m.get('qs_ulcer_index', float('nan')):.5f}"],
        ["Kelly Criterion", f"{m.get('qs_kelly', float('nan')):.3f}"],
        ["Daily VaR (95%)", f"{m.get('qs_var_pct', float('nan')):.3f}%"],
        ["Daily CVaR (95%)", f"{m.get('qs_cvar_pct', float('nan')):.3f}%"],
        ["Skew", f"{m.get('qs_skew', float('nan')):.3f}"],
        ["Recovery Factor", f"{m.get('qs_recovery_factor', float('nan')):.3f}"],
        ["Win Days %", f"{win_days_pct:.1f}%"],
        ["Avg Daily Return", f"{avg_daily_pct:+.4f}%"],
        ["N Trading Days", f"{n_days}"],
    ]
    metrics_tbl = Table(metrics_rows, colWidths=[220, 150])
    metrics_tbl.setStyle(base_table_style([("SPAN", (0, 0), (0, 0))]))

    story = []
    story.append(Paragraph(name, title_style))
    story.append(Paragraph(subtitle, subtitle_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=10))

    story.append(Paragraph("Key Metrics", section_style))
    story.append(metrics_tbl)

    story.append(Paragraph("Equity Curve", section_style))
    story.append(line_chart_drawing("Cumulative Return (start=1.0)", {"equity": equity},
                                     colors_cycle=[GREEN]))

    story.append(Paragraph("Drawdown", section_style))
    story.append(line_chart_drawing("Underwater Plot", {"drawdown": drawdown},
                                     colors_cycle=[RED]))

    story.append(Paragraph("Monthly Returns", section_style))
    story.append(monthly_returns_table(rets))

    if extra_flowables:
        story.extend(extra_flowables)

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(BG)
        canvas.rect(0, 0, W, H, fill=1, stroke=0)
        canvas.restoreState()

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                             topMargin=36, bottomMargin=36, leftMargin=40, rightMargin=40)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(BLEND_PKL, "rb") as f:
        d = pickle.load(f)
    for config_name in d["per_symbol_daily"].keys():
        per_symbol = d["per_symbol_daily"][config_name]
        slug = config_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        config_dir = os.path.join(OUT_DIR, slug)
        os.makedirs(config_dir, exist_ok=True)

        for symbol in FX_MAJORS:
            rets = per_symbol[symbol]
            out_path = os.path.join(config_dir, f"{symbol.lower()}.pdf")
            build_report(f"{symbol} -- {config_name}", rets, out_path,
                         subtitle=f"MetaFVG -- {config_name} -- M15/4h -- univariate (single pair)")
            print(f"written: {out_path}", flush=True)

        rets_df = pd.DataFrame(per_symbol)
        blended = rets_df.mean(axis=1, skipna=True).dropna()
        out_path = os.path.join(config_dir, "portfolio.pdf")
        build_report(f"FX Majors Portfolio -- {config_name}", blended, out_path,
                     subtitle=f"MetaFVG -- {config_name} -- M15/4h -- equal-weighted, 7 pairs "
                              f"(AUDUSD/EURUSD/GBPUSD/NZDUSD/USDCAD/USDCHF/USDJPY)")
        print(f"written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
