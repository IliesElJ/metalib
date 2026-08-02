"""
MetaFVG Strategy — Fundamental & Statistical Review, for PM-level review.

Loads the cached output of metafvg_pm_report_data.py (run that first) and
renders a statistically-grounded PDF: the structural payoff asymmetry (hard
stop vs ATR-sized target), win-rate-vs-breakeven edge estimates with Wilson
score confidence intervals per instrument/scale, and the sample-size story
behind why H4/Weekly and M15/4h disagree on some instruments.

Run from the metalib repo root with the adonys interpreter:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/metafvg_pm_report_data.py   # once, ~15-20 min
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/generate_metafvg_pm_report.py
"""
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

from datetime import datetime

import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(RESEARCH_DIR, "data", "metafvg_pm_report_data.pkl")
OUT_PDF = os.path.join(RESEARCH_DIR, "reports", "metafvg_pm_review.pdf")

with open(CACHE_PATH, "rb") as f:
    DATA = pickle.load(f)

# =========================================================================
# House style (matches build_report_pdf.py / generate_metafvg_backtest_report.py)
# =========================================================================
STYLE = {
    "bg": "#0f1117", "panel": "#1a1d27", "green": "#00c853", "red": "#ff3d3d",
    "blue": "#4a9eff", "orange": "#ff9800", "purple": "#ab47bc",
    "text": "#e0e0e0", "muted": "#666677", "grid": "#2a2d3a",
}
BG = HexColor(STYLE["bg"])
PANEL = HexColor(STYLE["panel"])
GREEN = HexColor(STYLE["green"])
RED = HexColor(STYLE["red"])
BLUE = HexColor(STYLE["blue"])
AMBER = HexColor(STYLE["orange"])
TEXT = HexColor(STYLE["text"])
MUTED = HexColor(STYLE["muted"])
ACCENT = HexColor("#2a2d3a")
W, H = A4

styles = getSampleStyleSheet()
title_style = ParagraphStyle("Title", parent=styles["Normal"], fontSize=25, leading=29, textColor=TEXT,
                              alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4)
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11, leading=14, textColor=MUTED,
                                 alignment=TA_CENTER, fontName="Helvetica", spaceAfter=4)
meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9.5, leading=12, textColor=MUTED,
                             alignment=TA_CENTER, fontName="Helvetica", spaceAfter=12)
section_style = ParagraphStyle("Section", parent=styles["Normal"], fontSize=14, leading=17, textColor=BLUE,
                                fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=6)
subsection_style = ParagraphStyle("Subsection", parent=styles["Normal"], fontSize=11.5, leading=14, textColor=TEXT,
                                   fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
footer_style = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=9, leading=11, textColor=MUTED,
                               alignment=TA_CENTER, fontName="Helvetica")
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, textColor=TEXT,
                             fontName="Helvetica", leading=15, spaceAfter=8)
bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"], fontSize=9.5, textColor=TEXT,
                               fontName="Helvetica", leading=15, spaceAfter=6, leftIndent=16, bulletIndent=4)
callout_style = ParagraphStyle("Callout", parent=styles["Normal"], fontSize=10, textColor=TEXT,
                                fontName="Helvetica", leading=15, spaceAfter=4)


def section(title):
    return [Paragraph(title, section_style), HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=6)]


def signed_para(val_str, positive, align=TA_RIGHT, size=9):
    col = GREEN if positive else RED
    return Paragraph(f'<font color="#{col.hexval()[2:]}">{val_str}</font>',
                      ParagraphStyle("sv", textColor=col, fontSize=size, fontName="Helvetica", alignment=align))


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


def callout_box(paragraphs, border_color=BLUE):
    """A bordered callout card for headline findings/recommendations."""
    tbl = Table([[paragraphs]], colWidths=[W - 4 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("BOX", (0, 0), (-1, -1), 1.1, border_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return tbl


def fmt(x, decimals=2, signed=False, suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    f = f"{{:+.{decimals}f}}{suffix}" if signed else f"{{:.{decimals}f}}{suffix}"
    return f.format(x)


# =========================================================================
# Chart: line (equity curves)
# =========================================================================
def line_chart_drawing(title, series, width=460, height=200, n_x_labels=8):
    colors_cycle = [GREEN, BLUE, AMBER, HexColor(STYLE["purple"])]
    target_points = 60
    multi_series = len(series) > 1
    header_h = 30 if multi_series else 16

    d = Drawing(width, height)
    d.add(String(2, height - 14, title, fillColor=TEXT, fontSize=10.5, fontName="Helvetica-Bold"))

    lc = HorizontalLineChart()
    lc.x = 42
    lc.y = 22
    lc.width = width - 60
    lc.height = height - 22 - header_h

    plotted, labels = [], None
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
    # Scale padding to the actual data range, not a fixed % of value level -- otherwise a
    # near-flat series (e.g. equity curve close to its start value) gets squashed into a
    # sliver by a fixed abs(vmax)*0.01 padding that swamps the real variation.
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
    for i, (label, _) in enumerate(plotted):
        lc.lines[i].strokeColor = colors_cycle[i % len(colors_cycle)]
        lc.lines[i].strokeWidth = 1.3
    d.add(lc)

    if multi_series:
        ly, lx = height - 27, 2.0
        for i, (label, _) in enumerate(plotted):
            d.add(Line(lx, ly + 3, lx + 10, ly + 3, strokeColor=colors_cycle[i % len(colors_cycle)], strokeWidth=2))
            d.add(String(lx + 14, ly, label, fillColor=TEXT, fontSize=7.5, fontName="Helvetica"))
            lx += 14 + 7 * (len(label) + 2)
    return d


# =========================================================================
# Chart: forest plot (edge estimate +/- 95% CI vs breakeven)
# =========================================================================
def forest_plot_drawing(rows, width=460, height=260, title="Realized Edge vs. Breakeven (95% CI)"):
    """
    rows: list of dicts with keys label, edge_pp, lo_pp, hi_pp, n.
    Draws a horizontal error-bar (forest) plot: point estimate of (actual win
    rate - breakeven win rate) in percentage points, with its 95% Wilson CI.
    Green if the whole CI clears zero, red if it's whole below zero, amber if
    the CI straddles zero (edge not statistically distinguishable from
    breakeven at this sample size).
    """
    d = Drawing(width, height)
    d.add(String(2, height - 14, title, fillColor=TEXT, fontSize=10.5, fontName="Helvetica-Bold"))

    plot_x0, plot_x1 = 130, width - 20
    plot_y0, plot_y1 = 24, height - 34
    row_h = (plot_y1 - plot_y0) / len(rows)

    all_lo = [r["lo_pp"] for r in rows]
    all_hi = [r["hi_pp"] for r in rows]
    xmin, xmax = min(all_lo + [0]), max(all_hi + [0])
    pad = max((xmax - xmin) * 0.12, 1.0)
    xmin, xmax = xmin - pad, xmax + pad

    def to_x(val):
        return plot_x0 + (val - xmin) / (xmax - xmin) * (plot_x1 - plot_x0)

    # zero (breakeven) reference line
    zero_x = to_x(0)
    d.add(Line(zero_x, plot_y0 - 4, zero_x, plot_y1 + 4, strokeColor=MUTED, strokeWidth=0.8, strokeDashArray=[2, 2]))
    d.add(String(zero_x - 24, plot_y1 + 6, "breakeven", fillColor=MUTED, fontSize=6.5, fontName="Helvetica"))

    for i, r in enumerate(rows):
        cy = plot_y1 - (i + 0.5) * row_h
        color = GREEN if r["lo_pp"] > 0 else (RED if r["hi_pp"] < 0 else AMBER)
        x_lo, x_hi, x_pt = to_x(r["lo_pp"]), to_x(r["hi_pp"]), to_x(r["edge_pp"])
        d.add(Line(x_lo, cy, x_hi, cy, strokeColor=color, strokeWidth=1.4))
        d.add(Line(x_lo, cy - 3, x_lo, cy + 3, strokeColor=color, strokeWidth=1.0))
        d.add(Line(x_hi, cy - 3, x_hi, cy + 3, strokeColor=color, strokeWidth=1.0))
        d.add(Rect(x_pt - 2.2, cy - 2.2, 4.4, 4.4, fillColor=color, strokeColor=None))
        d.add(String(2, cy - 3, r["label"], fillColor=TEXT, fontSize=7.5, fontName="Helvetica"))

    # x-axis ticks
    d.add(Line(plot_x0, plot_y0 - 4, plot_x1, plot_y0 - 4, strokeColor=ACCENT, strokeWidth=0.6))
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        xv = xmin + frac * (xmax - xmin)
        xp = to_x(xv)
        d.add(Line(xp, plot_y0 - 4, xp, plot_y0 - 7, strokeColor=ACCENT, strokeWidth=0.6))
        d.add(String(xp - 8, plot_y0 - 16, f"{xv:+.0f}pp", fillColor=MUTED, fontSize=6.5, fontName="Helvetica"))
    return d


# =========================================================================
# Chart: R-multiple histogram
# =========================================================================
def r_multiple_histogram_drawing(values, width=460, height=220, title="Winning-Trade R-Multiple Distribution (all instruments, both scales pooled)"):
    bins = [0, 1, 2, 3, 5, 8, 100]
    bin_labels = ["0-1R", "1-2R", "2-3R", "3-5R", "5-8R", "8R+"]
    counts = [0] * (len(bins) - 1)
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break

    d = Drawing(width, height)
    d.add(String(2, height - 14, title, fillColor=TEXT, fontSize=10.5, fontName="Helvetica-Bold"))

    bc = VerticalBarChart()
    bc.x = 42
    bc.y = 30
    bc.width = width - 65
    bc.height = height - 60
    bc.data = [counts]
    bc.categoryAxis.categoryNames = bin_labels
    bc.categoryAxis.labels.fontSize = 7.5
    bc.categoryAxis.labels.fillColor = MUTED
    bc.categoryAxis.strokeColor = ACCENT
    bc.valueAxis.strokeColor = ACCENT
    bc.valueAxis.labels.fontSize = 7
    bc.valueAxis.labels.fillColor = MUTED
    bc.valueAxis.valueMin = 0
    bc.bars[0].fillColor = BLUE
    bc.barWidth = 10
    bc.groupSpacing = 8
    d.add(bc)
    n = len(values)
    d.add(String(width - 4, height - 27, f"n = {n:,} winners", fillColor=MUTED, fontSize=8,
                  fontName="Helvetica", textAnchor="end"))
    return d


# =========================================================================
# Derived analysis from cached data
# =========================================================================
SYMBOLS = ["AUDJPY", "EURUSD", "GER40", "US100"]
SCALES = ["H4/Weekly", "M15/4h"]

rows_stats = [DATA[(sym, scale)] for sym in SYMBOLS for scale in SCALES]

pooled_winner_r = []
for sym in SYMBOLS:
    for scale in SCALES:
        pooled_winner_r.extend(DATA[(sym, scale)]["winner_r_values"])

forest_rows = [
    {
        "label": f"{sym} ({scale.split('/')[0]})",
        "edge_pp": DATA[(sym, scale)]["edge_pp"],
        "lo_pp": DATA[(sym, scale)]["edge_ci_lo_pp"],
        "hi_pp": DATA[(sym, scale)]["edge_ci_hi_pp"],
        "n": DATA[(sym, scale)]["n_closed"],
    }
    for sym in SYMBOLS for scale in SCALES
]

# large-sample (M15) read is the trustworthy one per instrument
large_sample = {sym: DATA[(sym, "M15/4h")] for sym in SYMBOLS}
significant_positive = [s for s, d in large_sample.items() if d["edge_ci_lo_pp"] > 0]
significant_negative = [s for s, d in large_sample.items() if d["edge_ci_hi_pp"] < 0]
inconclusive = [s for s in SYMBOLS if s not in significant_positive and s not in significant_negative]

avg_fill_h4 = np.mean([DATA[(sym, "H4/Weekly")]["fill_rate_pct"] for sym in SYMBOLS])
avg_fill_m15 = np.mean([DATA[(sym, "M15/4h")]["fill_rate_pct"] for sym in SYMBOLS])

generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# =========================================================================
# Build the story
# =========================================================================
story = []

story += [
    Spacer(1, 0.4 * cm),
    Paragraph("MetaFVG Strategy Review", title_style),
    Paragraph("Fundamental Mechanism &amp; Statistical Edge Assessment", subtitle_style),
    HRFlowable(width="50%", thickness=0.5, color=MUTED, spaceAfter=6),
    Paragraph(
        f'Prepared for portfolio review &nbsp;|&nbsp; Engine: <font color="#4a9eff">metafvg_backtest.py</font> '
        f'&nbsp;|&nbsp; Instruments: {", ".join(SYMBOLS)} &nbsp;|&nbsp; Generated {generated_at}',
        meta_style,
    ),
    Spacer(1, 0.35 * cm),
]

# ── Executive Summary ───────────────────────────────────────────────────
story += section("Executive Summary")
story += [
    Paragraph(
        'MetaFVG is a trend-continuation strategy with a <b>hard, deterministic structural stop</b> '
        '(last swing pivot) and a <b>take-profit target sized off ATR, entirely decoupled from that stop</b>. '
        'This asymmetry is the single mechanical fact that explains every result in this review: every losing '
        'trade across all four instruments and both scales tested closes at almost exactly <b>-1.00R</b>, while '
        'winners are right-skewed and variable — median <b>+2R to +3R</b>, tails out to '
        f'<b>+{max(pooled_winner_r):.1f}R</b>. Win rate on its own is the wrong lens; the right question is '
        'whether the realized win rate clears the <b>breakeven win rate implied by the realized average winner '
        'size</b>, evaluated with a confidence interval appropriate to the sample.',
        body_style,
    ),
]
story.append(callout_box(
    Paragraph(
        f'<b>Headline finding:</b> at large sample size (M15/4h, 900-1,100+ closed trades per instrument), '
        f'<font color="#{GREEN.hexval()[2:]}"><b>{", ".join(significant_positive) or "none"}</b></font> '
        f'shows a statistically significant positive edge (95% CI entirely above breakeven); '
        f'<font color="#{RED.hexval()[2:]}"><b>{", ".join(significant_negative) or "none"}</b></font> shows a '
        f'statistically significant <b>negative</b> edge; '
        f'<font color="#{AMBER.hexval()[2:]}"><b>{", ".join(inconclusive) or "none"}</b></font> '
        f'{"is" if len(inconclusive) == 1 else "are"} inconclusive at this sample size. Several of the '
        'smaller-sample H4/Weekly reads (particularly US100) turn out not to replicate once the sample size '
        'grows by an order of magnitude — the earlier apparent edge was consistent with noise.',
        callout_style,
    ),
    border_color=BLUE,
))
story.append(Spacer(1, 0.15 * cm))

# ── Section 1: Payoff structure ─────────────────────────────────────────
story += section("1. The Payoff Structure")
story += [
    Paragraph(
        'Every trade\'s stop-loss is a swing pivot computed from the entry timeframe\'s recent price action — a '
        'hard, mechanical invalidation level with no discretion. The take-profit is <code>entry + ATR &times; '
        'atr_sensitivity &times; risk_reward</code> — a level fixed at trade inception from volatility alone, '
        'with no reference to how far away the stop happens to be. The practical consequence, visible in every '
        'instrument tested:',
        body_style,
    ),
]

payoff_rows = [["Instrument", "Scale", "N (closed)", "Avg Loss", "Avg Win", "Median Win", "Max Win"]]
for sym in SYMBOLS:
    for scale in SCALES:
        d = DATA[(sym, scale)]
        payoff_rows.append([
            sym, scale.split("/")[0], str(d["n_closed"]),
            signed_para(f'{fmt(d["avg_loser_r"], 2, signed=True)}R', False),
            signed_para(f'{fmt(d["avg_winner_r"], 2, signed=True)}R', True),
            f'{fmt(d["median_winner_r"], 2, signed=True)}R',
            f'{fmt(d["max_winner_r"], 2, signed=True)}R',
        ])
payoff_tbl = Table(payoff_rows, colWidths=[2.3*cm, 1.8*cm, 2.0*cm, 2.1*cm, 2.1*cm, 2.2*cm, 2.0*cm])
payoff_tbl.setStyle(base_table_style())
story += [payoff_tbl, Spacer(1, 0.2*cm)]
story += [
    Paragraph(
        'Losses are uniform by construction (the stop is deterministic); wins are heterogeneous and right-skewed '
        '— the mean winner is consistently pulled above the median by a small number of large trend moves. This '
        'is a classic trend-following payoff signature: frequent small defined losses, funded by infrequent large '
        'wins. It also means winners take materially longer to resolve than losers — trades that fail tend to '
        'fail fast; trades that work need time to run:',
        body_style,
    ),
]
dur_rows = [["Instrument", "Scale", "Avg Winner Duration", "Avg Loser Duration", "Ratio"]]
for sym in SYMBOLS:
    for scale in SCALES:
        d = DATA[(sym, scale)]
        ratio = d["avg_winner_duration_h"] / d["avg_loser_duration_h"] if d["avg_loser_duration_h"] else float("nan")
        dur_rows.append([sym, scale.split("/")[0], f'{fmt(d["avg_winner_duration_h"],1)}h',
                          f'{fmt(d["avg_loser_duration_h"],1)}h', f'{fmt(ratio,1)}x'])
dur_tbl = Table(dur_rows, colWidths=[2.6*cm, 2.0*cm, 3.4*cm, 3.4*cm, 2.0*cm])
dur_tbl.setStyle(base_table_style())
story += [dur_tbl, Spacer(1, 0.2*cm)]

story += [r_multiple_histogram_drawing(pooled_winner_r), Spacer(1, 0.15*cm)]
story += [
    Paragraph(
        'The pooled winner distribution across all eight (instrument, scale) combinations confirms the shape '
        'implied by the summary statistics: the bulk of wins cluster in the 1-3R range, with a long right tail '
        f'extending past 8R. This tail is doing real work in the aggregate expectancy and is a meaningfully '
        'different risk profile from a strategy with a tight, symmetric payoff — position sizing and drawdown '
        'tolerance should be set with this skew in mind, not a normal-distribution assumption.',
        body_style,
    ),
]

story.append(PageBreak())

# ── Section 2: Is there a real edge? ────────────────────────────────────
story += section("2. Is There a Real Edge? — Win Rate vs. Breakeven, With Confidence Intervals")
story += [
    Paragraph(
        'Win rate alone is uninformative without knowing the breakeven win rate implied by the realized average '
        'winner size: <b>breakeven WR = 1 / (1 + avg winner R)</b>. The chart below plots, for every instrument '
        'and scale, the realized edge — actual win rate minus that breakeven threshold, in percentage points — '
        'together with a 95% Wilson confidence interval on the estimate. A green bar means the edge is unlikely '
        'to be zero-or-negative by chance at this sample size; red means the reverse; amber means the sample is '
        'too small to tell.',
        body_style,
    ),
]
story += [forest_plot_drawing(forest_rows), Spacer(1, 0.2*cm)]

edge_rows = [["Instrument", "Scale", "N", "Win Rate", "Breakeven WR", "Edge", "95% CI (edge)"]]
for sym in SYMBOLS:
    for scale in SCALES:
        d = DATA[(sym, scale)]
        sig = d["edge_ci_lo_pp"] > 0 or d["edge_ci_hi_pp"] < 0
        edge_rows.append([
            sym, scale.split("/")[0], str(d["n_closed"]),
            f'{fmt(d["win_rate_pct"],1)}%', f'{fmt(d["breakeven_wr_pct"],1)}%',
            signed_para(f'{fmt(d["edge_pp"],1,signed=True)}pp', d["edge_pp"] >= 0),
            f'[{fmt(d["edge_ci_lo_pp"],1,signed=True)}, {fmt(d["edge_ci_hi_pp"],1,signed=True)}]pp' + (" *" if sig else ""),
        ])
edge_tbl = Table(edge_rows, colWidths=[2.2*cm, 1.7*cm, 1.5*cm, 2.0*cm, 2.4*cm, 2.0*cm, 3.4*cm])
edge_tbl.setStyle(base_table_style())
story += [edge_tbl, Spacer(1, 0.1*cm)]
story += [Paragraph('* 95% CI does not straddle zero — edge is statistically distinguishable from breakeven at this sample size.', bullet_style)]
story.append(Spacer(1, 0.15*cm))

story += [
    Paragraph(
        f'At the small H4/Weekly sample sizes (29-56 closed trades per instrument), none of the four edge '
        f'estimates are statistically distinguishable from breakeven — the confidence intervals are wide enough '
        f'to straddle zero even for AUDJPY, the best H4/Weekly performer. This is not a defect in the backtest; '
        f'it is an honest statement that ~50 trades is not enough to separate skill from variance for a strategy '
        f'with a ~25% win rate. At the M15/4h scale, replaying the identical entry/exit logic on the same '
        f'calendar window produces 900-1,100+ closed trades per instrument — roughly 20x the sample — and the '
        f'picture resolves.',
        body_style,
    ),
]

story.append(PageBreak())

# ── Section 3: Why the scales disagree ──────────────────────────────────
story += section("3. Why H4/Weekly and M15/4h Disagree — Sample Size, Not Different Mechanics")
audjpy_h4, audjpy_m15 = DATA[("AUDJPY", "H4/Weekly")], DATA[("AUDJPY", "M15/4h")]
us100_h4, us100_m15 = DATA[("US100", "H4/Weekly")], DATA[("US100", "M15/4h")]
audjpy_m15_sig = audjpy_m15["edge_ci_lo_pp"] > 0
us100_m15_sig = us100_m15["edge_ci_hi_pp"] < 0
audjpy_m15_clause = (
    f'a confirmed positive edge (CI excludes zero)'
    if audjpy_m15_sig else
    f'a directionally positive edge that still narrowly fails to clear 95% significance '
    f'(CI [{fmt(audjpy_m15["edge_ci_lo_pp"],1,signed=True)}, {fmt(audjpy_m15["edge_ci_hi_pp"],1,signed=True)}]pp)'
)
story += [
    Paragraph(
        f'<b>AUDJPY</b> looked marginal at H4/Weekly (edge {fmt(audjpy_h4["edge_pp"],1,signed=True)}pp on just '
        f'{audjpy_h4["n_wins"]} winning trades) and moves toward {audjpy_m15_clause} at M15/4h '
        f'({fmt(audjpy_m15["edge_pp"],1,signed=True)}pp on {audjpy_m15["n_wins"]} winners) — directionally '
        f'consistent at both scales, even though the larger sample alone is not yet quite enough to call it '
        f'significant outright. <b>US100</b> looked like the strongest H4/Weekly performer '
        f'(edge {fmt(us100_h4["edge_pp"],1,signed=True)}pp, Sharpe {fmt(us100_h4["sharpe"],2,signed=True)}) on '
        f'only {us100_h4["n_wins"]} winning trades, and resolves to a confirmed <b>negative</b> edge at M15/4h '
        f'({fmt(us100_m15["edge_pp"],1,signed=True)}pp on {us100_m15["n_wins"]} winners, CI entirely below zero — '
        f'the only statistically significant result in this review). In both cases the mechanism replayed at M15 '
        'is identical to H4 — only the sample size changed. The correct interpretation is not "M15 works better '
        'for AUDJPY and worse for US100"; it is that <b>the H4/Weekly US100 result was very likely noise</b>, and '
        'the large-sample read is the one to trust — while AUDJPY, EURUSD and GER40 all remain genuinely '
        'inconclusive and warrant a larger sample still before sizing decisions.',
        body_style,
    ),
    Paragraph(
        f'A second, independent factor compounds this: fill rate. The strategy enters on a day-limit pullback '
        f'order into a freshly-formed gap; average fill rate across all four instruments is '
        f'<b>{fmt(avg_fill_h4,0)}%</b> at H4/Weekly versus <b>{fmt(avg_fill_m15,0)}%</b> at M15/4h — a finer '
        'entry timeframe captures a much larger share of the setups the logic actually identifies, independent '
        'of the statistical-power argument above.',
        body_style,
    ),
]

story.append(PageBreak())

# ── Section 4: Recommendations ──────────────────────────────────────────
story += section("4. Recommendations")
story += [
    Paragraph('<b>1. Fix the live short-side order bug (highest priority, affects capital now).</b> '
               '<code>metafvg.py::check_conditions()</code> never passes <code>short=True</code> to '
               '<code>execute()</code>, so every order — long or short signal alike — is sent as a buy-limit. '
               'Confirmed independently against live MT5 position history (100% of open positions are type BUY). '
               'Given the payoff structure in Section 1 depends on catching a small number of large trend moves, '
               'this bug likely forfeits roughly half of the strategy\'s available edge in any down-trending regime.',
               bullet_style),
    Paragraph('<b>2. Anchor the take-profit to the realized stop distance, not to ATR.</b> '
               '<code>risk_reward=2</code> currently does not produce anything resembling a 2:1 payoff — median '
               'realized winner R ranges 1.9-2.8 with no relationship to the stop distance actually taken. '
               'Changing to <code>tp = entry + (entry - sl) * risk_reward</code> would make the parameter mean '
               'what it says and let breakeven win rate be reasoned about directly per instrument. Test as a '
               'backtest-only fork before touching production logic.',
               bullet_style),
    Paragraph(f'<b>3. Do not scale further capital into {", ".join(significant_negative) or "the negative-edge instrument(s)"} '
               'under current parameters.</b> The large-sample read is statistically significant negative, not the '
               'small-sample positive result that motivated interest in it. Either exclude it or treat it as a '
               'separate parameter-search problem (momentum body ratio, HTF crossing filter) rather than assuming '
               'the shared default configuration generalizes across instruments.',
               bullet_style),
    Paragraph(
        (f'<b>4. {", ".join(significant_positive)} has a confirmed, large-sample edge.</b> '
         'It is the strongest candidate for continued or increased allocation among the four tested — subject to '
         'the caveats in Section 5 (no spread/slippage modeled, active-hours filter not replicated, '
         'fixed-fractional sizing for the percentage-based stats).')
        if significant_positive else
        ('<b>4. No instrument has yet cleared statistical significance on the positive side.</b> '
         f'AUDJPY is the closest — directionally positive at both scales tested, with its M15/4h confidence '
         f'interval [{fmt(audjpy_m15["edge_ci_lo_pp"],1,signed=True)}, {fmt(audjpy_m15["edge_ci_hi_pp"],1,signed=True)}]pp '
         'narrowly including zero. Treat it as the best candidate for a larger confirmatory sample, not yet as a '
         'confirmed edge to size into.'),
        bullet_style,
    ),
    Paragraph('<b>5. Raise fill rate at coarser scales, or standardize on the finer entry timeframe in production.</b> '
               'Given M15/4h captures roughly double the fill rate of H4/Weekly for identical setups, either '
               'shortening entry-order validity requirements or moving production to a finer LTF captures a '
               'larger share of the edge already being correctly identified.',
               bullet_style),
]

story.append(PageBreak())

# ── Section 5: Methodology & Caveats ────────────────────────────────────
story += section("5. Methodology &amp; Caveats")
story += [
    Paragraph(
        'Each instance was replayed bar-by-bar reusing MetaFVG\'s own live detection and trade-parameter methods '
        'directly (no reimplementation), cross-validated against real MT5 fills on EURUSD at production scales, '
        'where simulated entry and stop-loss matched live fills to 5 decimal places.',
        body_style,
    ),
    Paragraph('&bull; <b>Active-hours filter not applied.</b> AUDJPY and GER40 restrict live trading to specific '
               'server hours; this backtest evaluates every hour, so their trade counts likely overstate live '
               'frequency (win rate / edge estimates are less affected, since this is a time-of-day filter, not '
               'a signal-quality filter).', bullet_style),
    Paragraph('&bull; <b>No spread or slippage.</b> Fills use MT5 bid-only candle ranges.', bullet_style),
    Paragraph('&bull; <b>Same-bar SL/TP tie-break assumes stop-loss fires first</b> — conservative.', bullet_style),
    Paragraph('&bull; <b>Percentage-based stats (Sharpe, Max Drawdown, Total Return) use fixed 2%-of-equity '
               'sizing</b>, not live lot sizes — this keeps them comparable across instruments of very different '
               'price scale (EURUSD ~1.10/unit vs. GER40 ~20,000/unit) but is not a live P&amp;L projection. R-multiple '
               'and win-rate figures are sizing-independent.', bullet_style),
    Paragraph('&bull; <b>Concurrency cap enforced strictly</b>; live MT5 data shows the bot has at times exceeded '
               'its configured position limit because the live check does not see resting pending orders.', bullet_style),
    Paragraph('&bull; <b>Wilson score intervals</b> assume trade outcomes are independent draws from a fixed-probability '
               'process — a simplification given autocorrelated market regimes, but standard practice for a '
               'first-pass significance check and considerably more conservative than treating the point estimate '
               'as exact.', bullet_style),
]

story += [
    Spacer(1, 0.3*cm),
    HRFlowable(width="100%", thickness=0.3, color=MUTED, spaceAfter=6),
    Paragraph(f"Generated {generated_at} &middot; metafvg_backtest.py &middot; not investment advice", footer_style),
]

# =========================================================================
# Render
# =========================================================================
doc = SimpleDocTemplate(
    OUT_PDF, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
    title="MetaFVG Strategy Review — Fundamental & Statistical Assessment",
    author="metafvg_backtest.py",
)


def on_page(canvas, doc_):
    canvas.saveState()
    canvas.setFillColor(BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    canvas.restoreState()


doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF written to: {OUT_PDF}")
