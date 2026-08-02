"""
MetaOB Trend-Filter Validation Report - bucketed by asset class.

Loads metaob_ab_sweep_data.pkl (run metaob_ab_sweep.py first) and renders a
PDF comparing three configurations across the full asset universe:
  - No Trend Filter:   raw OB + pivot-breakout edge, no trend gate at all
  - Old Sharpe Filter:  the filter that was actually live in production -
                        rolling Sharpe of M15 returns over a ~2-day window,
                        thresholded against a daily-recalibrated quantile
  - New Trend Filter:   the fix - rolling OLS t-stat of D1 closes over a
                        ~200-trading-day window

This is the empirical check for whether the trend-filter fix (which corrects
trades firing against the genuine long-term trend - confirmed by a live-trade
audit against D1 SMA200) actually improves risk-adjusted performance, or only
changes which trades fire.

Run from the metalib repo root with the adonys interpreter:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/metaob_ab_sweep.py          # once, ~20-40 min
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/generate_metaob_ab_report.py
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
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from metafvg_ab_universe import UNIVERSE, symbol_to_bucket

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(RESEARCH_DIR, "data", "metaob_ab_sweep_data.pkl")
OUT_PDF = os.path.join(RESEARCH_DIR, "reports", "metaob_ab_review.pdf")

with open(CACHE_PATH, "rb") as f:
    DATA = pickle.load(f)

SYM2BUCKET = symbol_to_bucket()
CONFIGS = ["No Trend Filter", "Old Sharpe Filter", "New Trend Filter"]
BUCKETS = list(UNIVERSE.keys())

# =========================================================================
# House style (matches generate_metafvg_ab_report.py)
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
PURPLE = HexColor(STYLE["purple"])
TEXT = HexColor(STYLE["text"])
MUTED = HexColor(STYLE["muted"])
ACCENT = HexColor("#2a2d3a")
W, H = A4
CONFIG_COLORS = [MUTED, RED, BLUE]  # No Filter=neutral, Old(buggy)=red, New(fix)=blue

styles = getSampleStyleSheet()
title_style = ParagraphStyle("Title", parent=styles["Normal"], fontSize=24, leading=28, textColor=TEXT,
                              alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4)
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11, leading=14, textColor=MUTED,
                                 alignment=TA_CENTER, fontName="Helvetica", spaceAfter=4)
meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9.5, leading=13, textColor=MUTED,
                             alignment=TA_CENTER, fontName="Helvetica", spaceAfter=12)
section_style = ParagraphStyle("Section", parent=styles["Normal"], fontSize=14, leading=17, textColor=BLUE,
                                fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=6)
subsection_style = ParagraphStyle("Subsection", parent=styles["Normal"], fontSize=11.5, leading=14, textColor=TEXT,
                                   fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
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
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PANEL, ACCENT]),
        ("GRID", (0, 0), (-1, -1), 0.3, ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
    ]
    if extra:
        s.extend(extra)
    return TableStyle(s)


def callout_box(paragraphs, border_color=BLUE):
    tbl = Table([[paragraphs]], colWidths=[W - 4 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("BOX", (0, 0), (-1, -1), 1.1, border_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return tbl


def fmt(x, decimals=2, signed=False, suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    f = f"{{:+.{decimals}f}}{suffix}" if signed else f"{{:.{decimals}f}}{suffix}"
    return f.format(x)


# =========================================================================
# Grouped bar chart: metric by bucket, one series per config
# =========================================================================
def grouped_bar_chart(title, bucket_values, width=460, height=250):
    """bucket_values: dict[bucket] -> dict[config] -> float"""
    d = Drawing(width, height)
    d.add(String(2, height - 14, title, fillColor=TEXT, fontSize=10.5, fontName="Helvetica-Bold"))

    bc = VerticalBarChart()
    bc.x = 40
    bc.y = 38
    bc.width = width - 55
    bc.height = height - 78
    bc.data = [[bucket_values[b].get(cfg, 0.0) for b in BUCKETS] for cfg in CONFIGS]
    bc.categoryAxis.categoryNames = [b.replace(" ", "\n") for b in BUCKETS]
    bc.categoryAxis.labels.fontSize = 6.3
    bc.categoryAxis.labels.fillColor = MUTED
    bc.categoryAxis.labels.dy = -(bc.y - 4)
    bc.categoryAxis.strokeColor = ACCENT
    bc.valueAxis.strokeColor = ACCENT
    bc.valueAxis.labels.fontSize = 7
    bc.valueAxis.labels.fillColor = MUTED
    bc.groupSpacing = 12
    bc.barSpacing = 1
    for i, color in enumerate(CONFIG_COLORS[: len(CONFIGS)]):
        bc.bars[i].fillColor = color
    d.add(bc)

    lx, ly = 2.0, height - 27
    for cfg, color in zip(CONFIGS, CONFIG_COLORS):
        d.add(Line(lx, ly + 3, lx + 10, ly + 3, strokeColor=color, strokeWidth=2.5))
        d.add(String(lx + 14, ly, cfg, fillColor=TEXT, fontSize=7.3, fontName="Helvetica"))
        lx += 14 + 6.3 * (len(cfg) + 2)
    return d


# =========================================================================
# Aggregate the cached sweep data
# =========================================================================
def get(sym, cfg, key, default=np.nan):
    r = DATA.get((sym, cfg))
    return r.get(key, default) if r else default


bucket_config_agg = {}
for bucket, syms in UNIVERSE.items():
    bucket_config_agg[bucket] = {}
    for cfg in CONFIGS:
        sharpes = [get(s, cfg, "sharpe") for s in syms if (s, cfg) in DATA]
        edges = [get(s, cfg, "edge_pp") for s in syms if (s, cfg) in DATA]
        win_rates = [get(s, cfg, "win_rate_pct") for s in syms if (s, cfg) in DATA]
        n_trades = [get(s, cfg, "n_closed", 0) for s in syms if (s, cfg) in DATA]
        sharpes = [x for x in sharpes if np.isfinite(x)]
        edges = [x for x in edges if np.isfinite(x)]
        win_rates = [x for x in win_rates if np.isfinite(x)]
        bucket_config_agg[bucket][cfg] = {
            "avg_sharpe": np.mean(sharpes) if sharpes else np.nan,
            "avg_edge_pp": np.mean(edges) if edges else np.nan,
            "avg_win_rate_pct": np.mean(win_rates) if win_rates else np.nan,
            "total_closed_trades": int(np.sum(n_trades)),
            "n_symbols": len([s for s in syms if (s, cfg) in DATA]),
        }

overall_config_agg = {}
for cfg in CONFIGS:
    all_syms = list(SYM2BUCKET.keys())
    sharpes = [get(s, cfg, "sharpe") for s in all_syms if (s, cfg) in DATA]
    edges = [get(s, cfg, "edge_pp") for s in all_syms if (s, cfg) in DATA]
    win_rates = [get(s, cfg, "win_rate_pct") for s in all_syms if (s, cfg) in DATA]
    pfs = [get(s, cfg, "profit_factor") for s in all_syms if (s, cfg) in DATA]
    n_trades = [get(s, cfg, "n_closed", 0) for s in all_syms if (s, cfg) in DATA]
    qs_sharpes = [get(s, cfg, "qs_sharpe") for s in all_syms if (s, cfg) in DATA]
    sharpes_c = [x for x in sharpes if np.isfinite(x)]
    edges_c = [x for x in edges if np.isfinite(x)]
    win_rates_c = [x for x in win_rates if np.isfinite(x)]
    pfs_c = [x for x in pfs if np.isfinite(x)]
    qs_sharpes_c = [x for x in qs_sharpes if np.isfinite(x)]
    overall_config_agg[cfg] = {
        "avg_sharpe": np.mean(sharpes_c) if sharpes_c else np.nan,
        "avg_qs_sharpe": np.mean(qs_sharpes_c) if qs_sharpes_c else np.nan,
        "avg_edge_pp": np.mean(edges_c) if edges_c else np.nan,
        "avg_win_rate_pct": np.mean(win_rates_c) if win_rates_c else np.nan,
        "avg_profit_factor": np.mean(pfs_c) if pfs_c else np.nan,
        "total_closed_trades": int(np.sum(n_trades)),
        "avg_closed_trades_per_symbol": float(np.mean(n_trades)) if n_trades else np.nan,
        "pct_symbols_positive_sharpe": (np.mean([x > 0 for x in sharpes_c]) * 100) if sharpes_c else np.nan,
    }

# Head-to-head: New Trend Filter vs Old Sharpe Filter, per symbol
h2h_new_wins, h2h_old_wins, h2h_ties = 0, 0, 0
h2h_rows = []
for sym in SYM2BUCKET:
    if (sym, "New Trend Filter") not in DATA or (sym, "Old Sharpe Filter") not in DATA:
        continue
    new_sh = get(sym, "New Trend Filter", "sharpe")
    old_sh = get(sym, "Old Sharpe Filter", "sharpe")
    if np.isnan(new_sh) or np.isnan(old_sh):
        continue
    delta = new_sh - old_sh
    if delta > 0.05:
        h2h_new_wins += 1
    elif delta < -0.05:
        h2h_old_wins += 1
    else:
        h2h_ties += 1
    h2h_rows.append((sym, SYM2BUCKET[sym], old_sh, new_sh, delta))

best_overall = max(overall_config_agg.items(), key=lambda kv: (kv[1]["avg_sharpe"] if not np.isnan(kv[1]["avg_sharpe"]) else -999))
bucket_winner = {}
for bucket in BUCKETS:
    valid = {cfg: v["avg_sharpe"] for cfg, v in bucket_config_agg[bucket].items() if not np.isnan(v["avg_sharpe"])}
    bucket_winner[bucket] = max(valid.items(), key=lambda kv: kv[1]) if valid else (None, np.nan)

generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
n_symbols_tested = len(set(s for (s, c) in DATA.keys()))

# =========================================================================
# Build the story
# =========================================================================
story = []
story += [
    Spacer(1, 0.4 * cm),
    Paragraph("MetaOB Trend-Filter Validation", title_style),
    Paragraph("No Filter vs. the Live (Buggy) Sharpe Filter vs. the D1-Regression Fix, by Asset Class", subtitle_style),
    HRFlowable(width="50%", thickness=0.5, color=MUTED, spaceAfter=6),
    Paragraph(
        f'{n_symbols_tested} instruments across {len(BUCKETS)} asset-class buckets &nbsp;|&nbsp; '
        f'M15 entries, D1 trend &nbsp;|&nbsp; Generated {generated_at}',
        meta_style,
    ),
    Spacer(1, 0.35 * cm),
]

story += section("Executive Summary")
story += [
    Paragraph(
        'A live-trade audit found MetaOB entries firing against the genuine long-term (D1) trend: the '
        '<b>Old Sharpe Filter</b> that was actually running in production classified trend using a rolling Sharpe '
        'ratio of M15 returns over a window that, despite being named <i>sma_long_hours</i>, was only ~190-220 M15 '
        'bars (~2 days), thresholded against a quantile recalibrated daily from the trailing 66 days. The '
        '<b>New Trend Filter</b> fix replaces this with a rolling OLS t-stat of D1 closes over a 200-bar '
        '(~200-trading-day) window - a genuine long-term trend read. <b>No Trend Filter</b> (the raw OB + '
        'pivot-breakout signal, no trend gate at all) is included as the baseline both other configs are gating.',
        body_style,
    ),
]
new_vs_old = overall_config_agg["New Trend Filter"]["avg_sharpe"] - overall_config_agg["Old Sharpe Filter"]["avg_sharpe"]
story.append(callout_box(
    Paragraph(
        f'<b>Headline finding:</b> <font color="#{CONFIG_COLORS[CONFIGS.index(best_overall[0])].hexval()[2:]}">'
        f'<b>{best_overall[0]}</b></font> has the highest average Sharpe across the full universe '
        f'({fmt(best_overall[1]["avg_sharpe"],2,signed=True)}). Head-to-head against the filter that was actually '
        f'live, the New Trend Filter beats Old Sharpe Filter on <b>{h2h_new_wins}/{h2h_new_wins+h2h_old_wins+h2h_ties} '
        f'symbols</b> ({h2h_old_wins} favor the old filter, {h2h_ties} roughly tied), for an average Sharpe delta of '
        f'{fmt(new_vs_old,2,signed=True)}. The fix corrects a real trend-alignment bug (confirmed independently '
        f'against D1 SMA200 on live trade history) but that is not automatically the same claim as "higher Sharpe '
        f'everywhere" - see the per-symbol head-to-head in Section 3.',
        callout_style,
    ),
    border_color=BLUE if new_vs_old >= 0 else RED,
))
story.append(Spacer(1, 0.15 * cm))

overall_rows = [["Config", "Sharpe (vbt)", "Sharpe (qs)", "Avg Edge", "Win Rate", "Avg PF", "Trades/Sym", "Sharpe+ %"]]
for cfg in CONFIGS:
    v = overall_config_agg[cfg]
    overall_rows.append([
        cfg,
        signed_para(fmt(v["avg_sharpe"], 2, signed=True), v["avg_sharpe"] >= 0 if not np.isnan(v["avg_sharpe"]) else True),
        signed_para(fmt(v["avg_qs_sharpe"], 2, signed=True), v["avg_qs_sharpe"] >= 0 if not np.isnan(v["avg_qs_sharpe"]) else True),
        signed_para(fmt(v["avg_edge_pp"], 1, signed=True, suffix="pp"), v["avg_edge_pp"] >= 0 if not np.isnan(v["avg_edge_pp"]) else True),
        fmt(v["avg_win_rate_pct"], 1, suffix="%"),
        fmt(v["avg_profit_factor"], 2),
        fmt(v["avg_closed_trades_per_symbol"], 1),
        fmt(v["pct_symbols_positive_sharpe"], 0, suffix="%"),
    ])
overall_tbl = Table(overall_rows, colWidths=[2.9*cm, 2.1*cm, 2.0*cm, 1.9*cm, 1.9*cm, 1.6*cm, 1.8*cm, 2.0*cm])
overall_tbl.setStyle(base_table_style())
story += [overall_tbl, Spacer(1, 0.2*cm)]

story.append(PageBreak())

# ── Section 2: bucket breakdown ─────────────────────────────────────────
story += section("2. Performance by Asset-Class Bucket")
sharpe_by_bucket = {b: {cfg: bucket_config_agg[b][cfg]["avg_sharpe"] for cfg in CONFIGS} for b in BUCKETS}
story += [grouped_bar_chart("Average Sharpe Ratio by Bucket and Configuration", sharpe_by_bucket), Spacer(1, 0.15*cm)]

bucket_rows = [["Bucket", "Winner", "Winner Sharpe", "No-Filter Sharpe", "N Symbols", "Closed Trades (all cfgs)"]]
for b in BUCKETS:
    winner_cfg, winner_sharpe = bucket_winner[b]
    baseline_sharpe = bucket_config_agg[b]["No Trend Filter"]["avg_sharpe"]
    n_sym = bucket_config_agg[b]["No Trend Filter"]["n_symbols"]
    total_trades = sum(bucket_config_agg[b][c]["total_closed_trades"] for c in CONFIGS)
    bucket_rows.append([
        b, winner_cfg or "n/a",
        signed_para(fmt(winner_sharpe, 2, signed=True), winner_sharpe >= 0 if not np.isnan(winner_sharpe) else True),
        signed_para(fmt(baseline_sharpe, 2, signed=True), baseline_sharpe >= 0 if not np.isnan(baseline_sharpe) else True),
        str(n_sym), str(total_trades),
    ])
bucket_tbl = Table(bucket_rows, colWidths=[2.6*cm, 3.0*cm, 2.6*cm, 2.6*cm, 2.2*cm, 3.7*cm])
bucket_tbl.setStyle(base_table_style())
story += [bucket_tbl, Spacer(1, 0.2*cm)]

story.append(PageBreak())

# ── Section 3: per-symbol head-to-head, New vs Old ──────────────────────
story += section("3. Per-Symbol Head-to-Head: New Trend Filter vs. Old Sharpe Filter")
story += [
    Paragraph(
        f'Sorted by Sharpe delta (New - Old). {h2h_new_wins} symbols favor the fix, {h2h_old_wins} favor the '
        f'filter that was actually live, {h2h_ties} are roughly tied (|&Delta;| &le; 0.05).',
        body_style,
    ),
]
h2h_rows.sort(key=lambda r: r[4], reverse=True)
h2h_table_rows = [["Symbol", "Bucket", "Old Sharpe", "New Sharpe", "Δ (New-Old)"]]
for sym, bucket, old_sh, new_sh, delta in h2h_rows:
    h2h_table_rows.append([
        sym, bucket,
        fmt(old_sh, 2, signed=True),
        fmt(new_sh, 2, signed=True),
        signed_para(fmt(delta, 2, signed=True), delta >= 0),
    ])
h2h_tbl = Table(h2h_table_rows, colWidths=[2.4*cm, 3.0*cm, 2.6*cm, 2.6*cm, 2.6*cm], repeatRows=1)
h2h_tbl.setStyle(base_table_style())
story += [h2h_tbl, Spacer(1, 0.2*cm)]

story.append(PageBreak())

# ── Section 4: methodology & caveats ─────────────────────────────────────
story += section("4. Methodology &amp; Caveats")
story += [
    Paragraph('&bull; <b>Entry/exit model.</b> MetaOB places market orders with no concurrency cap live (the '
               'are_positions_with_tag_open guard in signals() is commented out) - this backtest mirrors that: a '
               'position opens on every bar the OB+pivot+trend condition holds, sized at a fixed 2% of current '
               'equity per trade (percent sizing keeps stats comparable across instruments at very different price '
               'scales - see build_vbt_portfolio in metafvg_backtest.py, reused unchanged here). No spread, '
               'slippage, or commission is modeled.', bullet_style),
    Paragraph('&bull; <b>Same OB/pivot/ATR parameters across the whole universe</b> (pivot_window=60, '
               'breakout_lookback=3, atr_period=14, sl_atr_mult=5, tp_atr_mult=10) - not per-symbol-tuned like the '
               'live prod config (metaob.yaml). This isolates the trend-filter effect from per-symbol parameter '
               'tuning, but means absolute Sharpe/PF here should not be read as "what prod actually earns" for any '
               'one symbol.', bullet_style),
    Paragraph('&bull; <b>New Trend Filter uses only fully-closed D1 bars</b> - a day\'s regression output becomes '
               'available starting the following calendar day. Live MetaOB.compute_trend_t() can also see the '
               'still-forming current day\'s D1 bar, so this backtest is slightly more conservative (up to one day '
               'slower to react) than live.', bullet_style),
    Paragraph('&bull; <b>Old Sharpe Filter</b> replicates the original production logic faithfully, including its '
               'daily threshold recalibration (mirrors the old fit(), rescheduled daily at 00:01 UTC by '
               'metaworker.py) from a trailing 66-day quantile of the rolling-Sharpe series - not a single '
               'fixed threshold for the whole backtest.', bullet_style),
    Paragraph('&bull; <b>2-year window, M15 entries.</b> Both configs share the same fetched OHLC per symbol, so '
               'differences are attributable to the trend filter alone, not to different market windows.', bullet_style),
    Paragraph('&bull; <b>quantstats metrics</b> (Sharpe, Sortino, Calmar, CAGR, max drawdown) are computed on the '
               'vectorbt equity curve resampled to daily, per quantstats\' own annualization convention - reported '
               'per (symbol, config) in the underlying cache (metaob_ab_sweep_data.pkl) even though only vbt Sharpe '
               'and qs Sharpe are tabulated here for space.', bullet_style),
    Paragraph('&bull; <b>Sample size.</b> Per-symbol trade counts vary widely by config (the trend filters are, by '
               'design, selective) - see the "Trades/Sym" column in Section 1. Individual symbol Sharpe figures with '
               'few closed trades should be read with proportionally more skepticism.', bullet_style),
]

story += [
    Spacer(1, 0.3*cm),
    HRFlowable(width="100%", thickness=0.3, color=MUTED, spaceAfter=6),
    Paragraph(f"Generated {generated_at} &middot; metaob_ab_sweep.py &middot; not investment advice", footer_style),
]

# =========================================================================
# Render
# =========================================================================
doc = SimpleDocTemplate(
    OUT_PDF, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
    title="MetaOB Trend-Filter Validation",
    author="metaob_ab_sweep.py",
)


def on_page(canvas, doc_):
    canvas.saveState()
    canvas.setFillColor(BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    canvas.restoreState()


doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF written to: {OUT_PDF}")
