"""
Build trading_report.pdf from trading_report.md data using ReportLab.
Run from the metalib root directory.
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.colors import HexColor

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = HexColor("#0f1117")
PANEL   = HexColor("#1a1d27")
GREEN   = HexColor("#00c853")
RED     = HexColor("#ff3d3d")
BLUE    = HexColor("#4a9eff")
TEXT    = HexColor("#e0e0e0")
MUTED   = HexColor("#666677")
ACCENT  = HexColor("#2a2d3a")

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR    = os.path.join(REPORT_DIR, "report_imgs")
OUT_PDF    = os.path.join(REPORT_DIR, "trading_report.pdf")

W, H = A4

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "Title", parent=styles["Normal"],
    fontSize=26, textColor=TEXT, alignment=TA_CENTER,
    fontName="Helvetica-Bold", spaceAfter=4,
)
subtitle_style = ParagraphStyle(
    "Subtitle", parent=styles["Normal"],
    fontSize=11, textColor=MUTED, alignment=TA_CENTER,
    fontName="Helvetica", spaceAfter=4,
)
meta_style = ParagraphStyle(
    "Meta", parent=styles["Normal"],
    fontSize=10, textColor=MUTED, alignment=TA_CENTER,
    fontName="Helvetica", spaceAfter=12,
)
section_style = ParagraphStyle(
    "Section", parent=styles["Normal"],
    fontSize=14, textColor=BLUE,
    fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=6,
)
footer_style = ParagraphStyle(
    "Footer", parent=styles["Normal"],
    fontSize=9, textColor=MUTED, alignment=TA_CENTER,
    fontName="Helvetica",
)
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=9.5, textColor=TEXT, fontName="Helvetica",
    leading=15, spaceAfter=8,
)
bullet_style = ParagraphStyle(
    "Bullet", parent=styles["Normal"],
    fontSize=9.5, textColor=TEXT, fontName="Helvetica",
    leading=15, spaceAfter=5, leftIndent=16, bulletIndent=4,
)
subsection_style = ParagraphStyle(
    "Subsection", parent=styles["Normal"],
    fontSize=11, textColor=BLUE, fontName="Helvetica-Bold",
    spaceBefore=10, spaceAfter=4,
)

def section(title):
    return [
        Paragraph(title, section_style),
        HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=6),
    ]

def pval(v, positive=True):
    """Color a value string green/red."""
    col = GREEN.hexval() if positive else RED.hexval()
    return f'<font color="#{col[2:]}">{v}</font>'

# ── Table style helper ────────────────────────────────────────────────────────
def base_table_style(extra=None):
    s = [
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("TEXTCOLOR",  (0, 0), (-1, -1), TEXT),
        ("FONTNAME",   (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PANEL, ACCENT]),
        ("GRID",       (0, 0), (-1, -1), 0.3, ACCENT),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("ALIGN",      (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN",      (0, 0), (0, -1), "LEFT"),
    ]
    if extra:
        s.extend(extra)
    return TableStyle(s)

def img(name, width_frac=1.0):
    path = os.path.join(IMG_DIR, name)
    avail = W - 4 * cm
    return Image(path, width=avail * width_frac,
                 height=avail * width_frac * 0.4,
                 kind="proportional")

# ── Build story ───────────────────────────────────────────────────────────────
story = []

# Title
story += [
    Spacer(1, 0.5 * cm),
    Paragraph("Trading Report", title_style),
    Paragraph("Mar 29 to Apr 12, 2026", subtitle_style),
    HRFlowable(width="50%", thickness=0.5, color=MUTED, spaceAfter=6),
    Paragraph(
        'Account <b><font color="#e0e0e0">Ilies El Jaouhari</font></b> &nbsp;|&nbsp; '
        'Server: <font color="#4a9eff">FPTradingLLC-Demo</font> &nbsp;|&nbsp; Currency: USD',
        meta_style,
    ),
    Spacer(1, 0.4 * cm),
]

# ── Executive Summary ─────────────────────────────────────────────────────────
story += section("Executive Summary")
story += [
    Paragraph(
        'The portfolio closed the two-week period with a net loss of '
        '<font color="#ff3d3d"><b>$80.78</b></font> across 209 trades, driven primarily by '
        '<i>metafvg</i> running on EURUSD. The Sharpe ratio of <b>-2.29</b> and profit factor '
        'of <b>0.60</b> indicate that the current configuration is not generating risk-adjusted alpha. '
        'The first two trading days (Mar 30–31) were modestly profitable, but a sustained drawdown began '
        'on Apr 1 and accelerated into the Apr 3 session (<font color="#ff3d3d">$-36.39</font>), which '
        'coincided with heightened macro volatility driven by US tariff announcements. A partial recovery '
        'on Apr 8–9 was reversed by the final session loss on Apr 10.',
        body_style,
    ),
    Paragraph(
        'The key structural issue is <i>metafvg</i>\'s win rate of <b>12.5%</b>. With an RRR of 3.58, '
        'the strategy would need at least a <b>22%</b> win rate to break even — meaning the current regime '
        'is approximately 10 percentage points below viability. <i>metamlp</i> compounds the loss with an '
        'RRR below 1 (0.95), a configuration that cannot be profitable regardless of win rate unless corrected.',
        body_style,
    ),
    Spacer(1, 0.2 * cm),
]

# ── Account Snapshot ──────────────────────────────────────────────────────────
story += section("Account Snapshot")

snap_data = [
    ["Metric",     "Value"],
    ["Balance",    "$87,185.94"],
    ["Equity",     "$86,477.13"],
    ["Open P&L",   Paragraph(pval("$-708.81", False), ParagraphStyle("v", textColor=RED, fontSize=9, fontName="Helvetica", alignment=TA_RIGHT))],
    ["Margin Used","$93.59"],
]
snap_tbl = Table(snap_data, colWidths=[8 * cm, 5 * cm])
snap_tbl.setStyle(base_table_style([
    ("FONTNAME", (1, 1), (1, 3), "Helvetica-Bold"),
]))
story += [snap_tbl, Spacer(1, 0.3 * cm)]

# ── Equity Curve ──────────────────────────────────────────────────────────────
story += section("Equity Curve")
story += [img("equity_curve.png"), Spacer(1, 0.15 * cm)]
story += [
    Paragraph(
        'The equity curve shows two distinct phases: a brief period of stability in the final days of March, '
        'followed by a near-monotonic decline through the first two weeks of April. There is no meaningful '
        'mean-reversion in the drawdown, which suggests the losses are regime-driven rather than random noise. '
        'The curve flattened slightly mid-week (Apr 7–9) before resuming lower.',
        body_style,
    ),
]

# ── Daily P&L ─────────────────────────────────────────────────────────────────
story += section("Daily P&L")
story += [img("daily_pnl.png"), Spacer(1, 0.2 * cm)]

def signed_para(val_str, positive):
    col = GREEN if positive else RED
    return Paragraph(
        f'<font color="#{col.hexval()[2:]}">{val_str}</font>',
        ParagraphStyle("sv", textColor=col, fontSize=9, fontName="Helvetica", alignment=TA_RIGHT),
    )

daily_rows = [
    ("Mar 30", "+$1.19",   True,  "+$1.19",   True),
    ("Mar 31", "+$3.65",   True,  "+$4.84",   True),
    ("Apr 01", "$-14.95",  False, "$-10.11",  False),
    ("Apr 02", "$-19.06",  False, "$-29.17",  False),
    ("Apr 03", "$-36.39",  False, "$-65.56",  False),
    ("Apr 06", "$-11.87",  False, "$-77.43",  False),
    ("Apr 07", "$-1.91",   False, "$-79.34",  False),
    ("Apr 08", "+$6.65",   True,  "$-72.69",  False),
    ("Apr 09", "+$2.67",   True,  "$-70.02",  False),
    ("Apr 10", "$-10.76",  False, "$-80.78",  False),
]

daily_data = [["Date", "Daily P&L", "Cumulative"]]
for date, dp, dp_pos, cum, cum_pos in daily_rows:
    daily_data.append([date, signed_para(dp, dp_pos), signed_para(cum, cum_pos)])

daily_tbl = Table(daily_data, colWidths=[4 * cm, 5 * cm, 5 * cm])
daily_tbl.setStyle(base_table_style())
story += [daily_tbl, Spacer(1, 0.15 * cm)]
story += [
    Paragraph(
        'The sharpest single-day loss occurred on <b>Apr 3</b> (<font color="#ff3d3d">$-36.39</font>), '
        'representing nearly half the total two-week drawdown in a single session. The two green days '
        '(Apr 8–9, totalling +$9.32) were insufficient to offset the broader trend. No single day '
        'produced a meaningful upside outlier, whereas the downside saw two outsized events (Apr 3 and Apr 2).',
        body_style,
    ),
]

# ── Overall Performance ───────────────────────────────────────────────────────
story += section("Overall Performance")

perf_data = [
    ["Metric",                    "Value"],
    ["Closed Trades",             "209"],
    ["Total P&L",                 signed_para("$-80.78", False)],
    ["Win Rate",                  "18.2%"],
    ["Profit Factor",             "0.60"],
    ["Sharpe Ratio",              "-2.29"],
    ["RRR (avg win / avg loss)",  "2.71"],
    ["Max Drawdown",              "-0.11%"],
]
perf_tbl = Table(perf_data, colWidths=[9 * cm, 5 * cm])
perf_tbl.setStyle(base_table_style())
story += [perf_tbl, Spacer(1, 0.15 * cm)]
story += [
    Paragraph(
        'Despite a high average win-to-loss ratio (<b>RRR = 2.71</b>), the portfolio\'s 18.2% win rate '
        'falls well short of the minimum required for the aggregate RRR to generate positive expectancy '
        '(~27%). This divergence between RRR quality and win rate frequency is the defining problem of '
        'this period: the system finds large winners but not often enough to overcome the volume of small losses.',
        body_style,
    ),
    Spacer(1, 0.15 * cm),
]

# ── Strategy Breakdown ────────────────────────────────────────────────────────
story += section("Strategy Breakdown")
story += [img("strategy_breakdown.png"), Spacer(1, 0.2 * cm)]
story += [img("win_rate.png", width_frac=0.7), Spacer(1, 0.2 * cm)]

strat_data = [
    ["Strategy", "Trades", "Total P&L", "Win Rate", "PF", "RRR", "Max DD"],
    ["metafvg",  "168",    signed_para("-52.11", False),  "12.5%",  "0.51", "3.58", "-0.1%"],
    ["metago",   "2",      signed_para("+10.67", True),   "100.0%", "inf",  "inf",  "0.0%"],
    ["metamlp",  "39",     signed_para("-39.34", False),  "38.5%",  "0.59", "0.95", "-0.1%"],
]
col_w = [3.5*cm, 2*cm, 3*cm, 2.5*cm, 2*cm, 2*cm, 2.2*cm]
strat_tbl = Table(strat_data, colWidths=col_w)
strat_tbl.setStyle(base_table_style())
story += [strat_tbl, Spacer(1, 0.15 * cm)]
story += [
    Paragraph(
        '<b>metafvg</b> (168 trades, 80% of total volume) is the primary driver of losses. Its 12.5% win rate '
        'on EURUSD is structurally insufficient for its 3.58 RRR — the breakeven threshold sits around 22%. '
        'The strategy is likely entering too frequently in a trending or noisy regime unfavorable to FVG fills. '
        'Reducing position frequency or adding a regime filter (e.g. ATR-based volatility gate or trend direction '
        'filter) would be the first lever to pull.',
        body_style,
    ),
    Paragraph(
        '<b>metamlp</b> (39 trades) presents a different problem: its RRR of 0.95 means average losses exceed '
        'average wins. An MLP mean-reversion strategy with sub-1 RRR suggests the take-profit targets are set '
        'too tight relative to stop losses, or that the model\'s predicted direction is correct but the sizing '
        'of moves is underestimated. Widening TP or tightening SL should be explored in backtesting.',
        body_style,
    ),
    Paragraph(
        '<b>metago</b> (2 trades, +$10.67, 100% win rate) is the only profitable strategy this period, '
        'but the sample is far too small to draw conclusions. It should be monitored closely as trade count grows.',
        body_style,
    ),
    Spacer(1, 0.15 * cm),
]

# ── Symbol P&L ────────────────────────────────────────────────────────────────
story += section("Symbol P&L")
story += [img("symbol_pnl.png", width_frac=0.8), Spacer(1, 0.2 * cm)]

sym_rows = [
    ("AUDJPY", "1",   "+$8.27",  True,  "100.0%", "+$8.27",  True),
    ("USDJPY", "1",   "+$2.40",  True,  "100.0%", "+$2.40",  True),
    ("EURCHF", "7",   "+$1.69",  True,  "42.9%",  "+$0.24",  True),
    ("CADCHF", "19",  "$-17.09", False, "47.4%",  "$-0.90",  False),
    ("NZDCHF", "13",  "$-23.94", False, "23.1%",  "$-1.84",  False),
    ("EURUSD", "168", "$-52.11", False, "12.5%",  "$-0.31",  False),
]
sym_data = [["Symbol", "Trades", "Total P&L", "Win Rate", "Avg P&L"]]
for sym, trades, pnl, pnl_pos, wr, avg, avg_pos in sym_rows:
    sym_data.append([sym, trades, signed_para(pnl, pnl_pos), wr, signed_para(avg, avg_pos)])

sym_tbl = Table(sym_data, colWidths=[3.5*cm, 2.5*cm, 4*cm, 3*cm, 4*cm])
sym_tbl.setStyle(base_table_style())
story += [sym_tbl, Spacer(1, 0.15 * cm)]
story += [
    Paragraph(
        'EURUSD accounts for <b>$-52.11</b> (64% of total losses) with 168 trades at a 12.5% win rate — '
        'this is entirely the <i>metafvg</i> exposure. NZDCHF is the second largest detractor ($-23.94, '
        '23.1% win rate), driven by <i>metamlp</i>. CADCHF ($-17.09, 47.4% win rate) is marginally below '
        'breakeven and likely improvable with better TP/SL calibration. The three profitable symbols '
        '(AUDJPY, USDJPY, EURCHF) are small in trade count and likely represent <i>metago</i> and isolated '
        '<i>metamlp</i> wins.',
        body_style,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph("Key Actions", subsection_style),
    Paragraph(
        '<b>metafvg / EURUSD</b> — Pause or reduce lot size until win rate can be improved above 22%. '
        'Investigate entry conditions; consider adding a higher-timeframe trend filter to avoid fading '
        'strong directional moves.',
        bullet_style,
    ),
    Paragraph(
        '<b>metamlp</b> — Recalibrate TP/SL ratio. Backtest with RRR &ge; 1.5 to ensure positive '
        'expectancy at observed win rates (~38%).',
        bullet_style,
    ),
    Paragraph(
        '<b>metago</b> — Let it run. Accumulate more trades before drawing conclusions, but flag as the '
        'most promising strategy this period.',
        bullet_style,
    ),
    Paragraph(
        '<b>Macro awareness</b> — The Apr 3 spike loss aligns with a known high-volatility macro event. '
        'Consider implementing a news/event blackout around scheduled high-impact releases.',
        bullet_style,
    ),
    Spacer(1, 0.4 * cm),
]

# Footer
story += [
    HRFlowable(width="100%", thickness=0.3, color=MUTED, spaceAfter=6),
    Paragraph("Generated 2026-04-12 14:50 UTC", footer_style),
]

# ── Render ────────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUT_PDF,
    pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
    title="Trading Report — Mar 29 to Apr 12, 2026",
    author="Ilies El Jaouhari",
)

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    canvas.restoreState()

doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF written to: {OUT_PDF}")
