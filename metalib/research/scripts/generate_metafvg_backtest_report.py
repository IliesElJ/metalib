"""
MetaFVG Multi-Asset Backtest Report — runs the upscaled (H4 -> Weekly) backtest
engine for every instance configured in metalib/config/prod/metafvg.yaml, then
renders a CTA-fund-style PDF tearsheet, reusing this repo's existing dark-theme
report house style (see build_report_pdf.py / generate_report.py).

Run from the metalib repo root with the adonys interpreter:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/generate_metafvg_backtest_report.py
"""
import os
import warnings

warnings.filterwarnings("ignore")

from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from metalib.metafvg_backtest import (
    BacktestParams,
    HTF_RESAMPLE_OPTIONS,
    LTF_TIMEFRAME_OPTIONS,
    PROD_HTF_LOOKBACK_BARS,
    run_backtest,
)

# =========================================================================
# Config / paths
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(RESEARCH_DIR))
REPORTS_DIR = os.path.join(RESEARCH_DIR, "reports")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "metalib", "config", "prod", "metafvg.yaml")

# LTF/HTF pair for this run — override via env vars, e.g.:
#   METAFVG_LTF=M15 METAFVG_HTF="4h" python generate_metafvg_backtest_report.py
# LTF_LABEL must be a key of LTF_TIMEFRAME_OPTIONS (M1/M5/M15/M30/H1/H4/D1);
# HTF_LABEL must be a key of HTF_RESAMPLE_OPTIONS (1h/4h/12h/1 Day/1 Week).
LTF_LABEL = os.environ.get("METAFVG_LTF", "H4")
HTF_LABEL = os.environ.get("METAFVG_HTF", "1 Week")
RUN_LTF_TIMEFRAME = LTF_TIMEFRAME_OPTIONS[LTF_LABEL]
RUN_HTF_RESAMPLE_RULE = HTF_RESAMPLE_OPTIONS[HTF_LABEL]
# 42 bars (7 days worth of 4h bars) is calibrated for HTF=4h, matching
# MetaFVG.DEFAULT_LOOKBACK_DAYS; reuse for any HTF choice as a reasonable default
# rolling-zone-detection window rather than the H4->Weekly report's 52 (~1yr).
RUN_HTF_LOOKBACK_BARS = PROD_HTF_LOOKBACK_BARS if HTF_LABEL != "1 Week" else 52

_slug = f"{LTF_LABEL}_{HTF_LABEL}".lower().replace(" ", "").replace("'", "")
OUT_PDF = os.path.join(REPORTS_DIR, f"metafvg_backtest_report_{_slug}.pdf")

BACKTEST_END = datetime.utcnow()
# 5 years back: long enough for a meaningful sample, short enough to stay clear of
# the pre-inception synthetic/spliced data some brokers backfill for older
# instruments (e.g. this server's EURUSD H4 history nominally starts in 1971,
# decades before the euro existed — not usable for a credible backtest window).
# GER40/US100 have less live history than this anyway and are naturally capped by
# their own inception once fetched.
BACKTEST_START = BACKTEST_END - pd.Timedelta(days=365 * 5)

# =========================================================================
# House style (matches build_report_pdf.py / generate_report.py)
# =========================================================================
STYLE = {
    "bg": "#0f1117",
    "panel": "#1a1d27",
    "green": "#00c853",
    "red": "#ff3d3d",
    "blue": "#4a9eff",
    "orange": "#ff9800",
    "purple": "#ab47bc",
    "text": "#e0e0e0",
    "muted": "#666677",
    "grid": "#2a2d3a",
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
# NB: ParagraphStyle inherits `leading` from its parent verbatim -- it does not
# auto-scale with an overridden `fontSize`. Every style below that bumps fontSize
# well past styles["Normal"]'s (10pt/leading 12) must set its own `leading`
# explicitly, or the next paragraph's baseline lands inside this one's glyphs.
title_style = ParagraphStyle("Title", parent=styles["Normal"], fontSize=26, leading=30, textColor=TEXT,
                              alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4)
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11, leading=14, textColor=MUTED,
                                 alignment=TA_CENTER, fontName="Helvetica", spaceAfter=4)
meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=10, leading=13, textColor=MUTED,
                             alignment=TA_CENTER, fontName="Helvetica", spaceAfter=12)
section_style = ParagraphStyle("Section", parent=styles["Normal"], fontSize=14, leading=17, textColor=BLUE,
                                fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=6)
subsection_style = ParagraphStyle("Subsection", parent=styles["Normal"], fontSize=12, leading=15, textColor=BLUE,
                                   fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
footer_style = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=9, leading=11, textColor=MUTED,
                               alignment=TA_CENTER, fontName="Helvetica")
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, textColor=TEXT,
                             fontName="Helvetica", leading=15, spaceAfter=8)
bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"], fontSize=9.5, textColor=TEXT,
                               fontName="Helvetica", leading=15, spaceAfter=5, leftIndent=16, bulletIndent=4)
caveat_style = ParagraphStyle("Caveat", parent=styles["Normal"], fontSize=8.5, textColor=MUTED,
                               fontName="Helvetica-Oblique", leading=13, spaceAfter=5, leftIndent=10)


def section(title):
    return [Paragraph(title, section_style), HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=6)]


def pcolor(v, positive=True):
    col = (GREEN if positive else RED).hexval()
    return f'<font color="#{col[2:]}">{v}</font>'


def signed_para(val_str, positive, align=TA_RIGHT):
    col = GREEN if positive else RED
    return Paragraph(f'<font color="#{col.hexval()[2:]}">{val_str}</font>',
                      ParagraphStyle("sv", textColor=col, fontSize=9, fontName="Helvetica", alignment=align))


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


def line_chart_drawing(title, series, width=460, height=210, n_x_labels=8):
    """
    Build a dark-themed line chart as a native reportlab vector Drawing (no
    matplotlib/PNG rasterization involved — this environment's matplotlib Agg
    savefig() crashes natively regardless of rc content or style, isolated and
    confirmed independent of this repo's code; reportlab's own charting renders
    straight into the PDF and sidesteps it entirely).

    series: dict[label] -> pandas Series (DatetimeIndex -> float). All series
    are downsampled to a common small point count for a readable static chart.
    """
    colors_cycle = [HexColor(STYLE["green"]), HexColor(STYLE["blue"]), HexColor(STYLE["orange"]), HexColor(STYLE["purple"])]
    target_points = 60
    multi_series = len(series) > 1
    # Reserve a dedicated legend row below the title when there's more than one
    # series, rather than overlaying legend swatches on top of the title text at
    # the same y — the title can run the full drawing width, so anything sharing
    # its row collides with it.
    header_h = 30 if multi_series else 16

    d = Drawing(width, height)
    d.add(String(2, height - 14, title, fillColor=TEXT, fontSize=10.5, fontName="Helvetica-Bold"))

    lc = HorizontalLineChart()
    lc.x = 42
    lc.y = 22
    lc.width = width - 60
    lc.height = height - 22 - header_h

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
        from reportlab.graphics.shapes import Line
        ly = height - 27
        lx = 2.0
        for i, (label, _) in enumerate(plotted):
            d.add(Line(lx, ly + 3, lx + 10, ly + 3,
                        strokeColor=colors_cycle[i % len(colors_cycle)], strokeWidth=2))
            d.add(String(lx + 14, ly, label, fillColor=TEXT, fontSize=7.5, fontName="Helvetica"))
            lx += 14 + 7 * (len(label) + 2)  # rough monospace-ish advance, good enough for short tickers

    return d


def fmt_pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}%"


def fmt_num(x, decimals=2, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    fmt = f"{{:+.{decimals}f}}" if signed else f"{{:.{decimals}f}}"
    return fmt.format(x)


# =========================================================================
# 1. Load config, run backtests
# =========================================================================
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

print(f"Loaded {len(config)} instances from {CONFIG_PATH}")

results = {}
for instance_name, entry in config.items():
    symbol = entry["symbols"][0]
    limit = int(entry["limit_number_position"])
    tag = entry["tag"]
    active_hours = entry.get("active_hours")
    has_hour_filter = isinstance(active_hours, list)

    print(f"\n=== {instance_name} ({tag}) — symbol={symbol}, limit={limit} ===")
    params = BacktestParams(
        symbol=symbol,
        start=BACKTEST_START,
        end=BACKTEST_END,
        limit_number_position=limit,
        ltf_timeframe=RUN_LTF_TIMEFRAME,
        htf_resample_rule=RUN_HTF_RESAMPLE_RULE,
        htf_lookback_bars=RUN_HTF_LOOKBACK_BARS,
    )
    try:
        result = run_backtest(params)
    except Exception as e:
        print(f"  FAILED: {e}")
        results[instance_name] = {"error": str(e), "symbol": symbol, "tag": tag, "limit": limit,
                                   "has_hour_filter": has_hour_filter}
        continue

    trades_df = result.trades_df
    closed = trades_df[trades_df["status"] == "closed"]
    stats = result.stats

    n_trades = len(trades_df)
    n_closed = len(closed)
    win_rate = (closed["pnl"] > 0).mean() * 100 if n_closed else float("nan")
    total_pnl_pts = closed["pnl"].sum() if n_closed else 0.0
    avg_r = closed["r_multiple"].mean() if n_closed else float("nan")
    wins = closed[closed["pnl"] > 0]["pnl"]
    losses = closed[closed["pnl"] < 0]["pnl"]
    profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("nan")

    results[instance_name] = {
        "symbol": symbol, "tag": tag, "limit": limit, "has_hour_filter": has_hour_filter,
        "result": result, "trades_df": trades_df, "closed": closed,
        "n_trades": n_trades, "n_closed": n_closed, "win_rate": win_rate,
        "total_pnl_pts": total_pnl_pts, "avg_r": avg_r, "profit_factor": profit_factor,
        "total_return_pct": stats.get("Total Return [%]", float("nan")),
        "sharpe": stats.get("Sharpe Ratio", float("nan")),
        "max_dd_pct": stats.get("Max Drawdown [%]", float("nan")),
        "start_covered": result.ltf_df.index.min(), "end_covered": result.ltf_df.index.max(),
        "n_ltf_bars": len(result.ltf_df), "n_htf_bars": len(result.htf_df),
    }
    print(f"  {n_trades} trades ({n_closed} closed) | win rate {win_rate:.1f}% | "
          f"Sharpe {results[instance_name]['sharpe']:.2f} | "
          f"coverage {result.ltf_df.index.min().date()} -> {result.ltf_df.index.max().date()}")

print("\nAll backtests complete. Building charts + PDF...")


# =========================================================================
# 2. Charts
# =========================================================================
equity_charts = {}
for instance_name, res in results.items():
    if "result" not in res:
        continue
    value = res["result"].portfolio.value()
    equity_charts[instance_name] = line_chart_drawing(
        f"{res['symbol']} — Equity Curve (vectorbt, 2% fixed-fractional)",
        {res["symbol"]: value},
        width=460, height=200,
    )

comparison_series = {
    res["symbol"]: (res["result"].portfolio.value() / res["result"].portfolio.value().iloc[0] * 100)
    for res in results.values() if "result" in res
}
comparison_chart = line_chart_drawing(
    "Normalized Equity Comparison (rebased to 100 at each series' own start)",
    comparison_series, width=460, height=230,
)


# =========================================================================
# 3. Aggregate figures for the executive summary
# =========================================================================
ok_results = {k: v for k, v in results.items() if "result" in v}
total_trades_all = sum(v["n_trades"] for v in ok_results.values())
total_closed_all = sum(v["n_closed"] for v in ok_results.values())
weighted_win_rate = (
    sum(v["win_rate"] * v["n_closed"] for v in ok_results.values() if not np.isnan(v["win_rate"])) / total_closed_all
    if total_closed_all else float("nan")
)
best_by_sharpe = max(ok_results.items(), key=lambda kv: (kv[1]["sharpe"] if not np.isnan(kv[1]["sharpe"]) else -999))
worst_by_sharpe = min(ok_results.items(), key=lambda kv: (kv[1]["sharpe"] if not np.isnan(kv[1]["sharpe"]) else 999))

generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

is_production_scales = (LTF_LABEL == "M1" and HTF_LABEL == "4h")
is_default_upscaled = (LTF_LABEL == "H4" and HTF_LABEL == "1 Week")
if is_production_scales:
    scale_relation = "matching MetaFVG's live production scales exactly"
elif is_default_upscaled:
    scale_relation = "one level upscaled from production's M1 &rarr; 4h scales, for a longer usable history window"
else:
    scale_relation = "between production's M1 &rarr; 4h scales and the fully-upscaled H4 &rarr; Weekly setup"

# =========================================================================
# 4. Build the story
# =========================================================================
story = []

story += [
    Spacer(1, 0.5 * cm),
    Paragraph("MetaFVG Strategy Review", title_style),
    Paragraph("Historical Backtest — All Configured Instances", subtitle_style),
    HRFlowable(width="50%", thickness=0.5, color=MUTED, spaceAfter=6),
    Paragraph(
        f'Engine: <font color="#4a9eff">metafvg_backtest.py</font> &nbsp;|&nbsp; '
        f'Scales: <b>{LTF_LABEL} &rarr; {HTF_LABEL}</b> '
        f'&nbsp;|&nbsp; Config: <font color="#4a9eff">config/prod/metafvg.yaml</font>',
        meta_style,
    ),
    Spacer(1, 0.4 * cm),
]

# ── Executive Summary ───────────────────────────────────────────────────
story += section("Executive Summary")

sharpe_best_val = best_by_sharpe[1]["sharpe"]
sharpe_worst_val = worst_by_sharpe[1]["sharpe"]
story += [
    Paragraph(
        f'This review covers all <b>{len(config)}</b> live-configured MetaFVG instances '
        f'({", ".join(v["symbol"] for v in results.values())}), backtested with '
        f'<b>{LTF_LABEL}</b> as the entry/exit timeframe and <b>{HTF_LABEL}</b> for zone detection — '
        f'{scale_relation}. MT5\'s terminal cache limits how far back finer timeframes are actually '
        f'available (M1 to roughly 3 months on this server); each instance\'s actual data coverage is shown '
        f'in its own section below rather than assumed. Across '
        f'<b>{total_trades_all}</b> generated signals (<b>{total_closed_all}</b> closed), the '
        f'volume-weighted win rate is <b>{fmt_num(weighted_win_rate, 1)}%</b>.',
        body_style,
    ),
    Paragraph(
        f'<b>{results[best_by_sharpe[0]]["symbol"]}</b> is the strongest performer on a risk-adjusted basis '
        f'(Sharpe {fmt_num(sharpe_best_val, 2, signed=True)}), while '
        f'<b>{results[worst_by_sharpe[0]]["symbol"]}</b> is the weakest '
        f'(Sharpe {fmt_num(sharpe_worst_val, 2, signed=True)}). None of the four configurations should be read '
        f'as a standalone allocation decision without the caveats in Section 2 — most notably, none of these '
        f'backtests replicate the live bot\'s active-hours trading-session filter (see per-instrument notes), '
        f'and win/loss classification assumes no bid/ask spread or slippage.',
        body_style,
    ),
    Spacer(1, 0.2 * cm),
]

# ── Comparison table ─────────────────────────────────────────────────────
story += section("Portfolio Comparison")
story += [comparison_chart, Spacer(1, 0.2 * cm)]

cmp_header = ["Instance", "Symbol", "Trades", "Win Rate", "Sharpe", "Max DD", "Profit Factor", "Coverage"]
cmp_rows = [cmp_header]
for instance_name, v in results.items():
    if "result" not in v:
        cmp_rows.append([instance_name, v["symbol"], "FAILED", "-", "-", "-", "-", "-"])
        continue
    cmp_rows.append([
        instance_name, v["symbol"], str(v["n_trades"]),
        f'{fmt_num(v["win_rate"], 1)}%',
        signed_para(fmt_num(v["sharpe"], 2, signed=True), v["sharpe"] >= 0 if not np.isnan(v["sharpe"]) else True),
        signed_para(f'{fmt_num(v["max_dd_pct"], 2)}%', False),
        fmt_num(v["profit_factor"], 2),
        f'{v["start_covered"].date()} to {v["end_covered"].date()}',
    ])
cmp_tbl = Table(cmp_rows, colWidths=[3.6*cm, 2*cm, 1.8*cm, 2*cm, 1.8*cm, 1.8*cm, 2.4*cm, 3.6*cm])
cmp_tbl.setStyle(base_table_style())
story += [cmp_tbl, Spacer(1, 0.2 * cm)]

# ── Methodology & Caveats ────────────────────────────────────────────────
story += section("Methodology & Caveats")
story += [
    Paragraph(
        f'Each instance was replayed bar-by-bar on {LTF_LABEL} candles, reusing MetaFVG\'s own live '
        f'detection and trade-parameter methods directly (no reimplementation) so simulated signals cannot '
        f'drift from production logic — cross-validated against real MT5 fills on EURUSD (production M1/4h '
        f'scales), where simulated entry and stop-loss matched live fills to 5 decimal places.',
        body_style,
    ),
    Paragraph("The following are known, deliberate simplifications of this backtest engine:", body_style),
    Paragraph(
        '&bull; <b>Active-hours filter not applied.</b> AUDJPY_H1 and GER40CASH_H1_GA restrict live trading '
        'to specific MT5-server hours; this backtest evaluates every hour, so trade counts and win rate for '
        'those two instances likely overstate what the live-hour-filtered bot actually captures. '
        'EURUSD_H1 and US100CASH_H1_GA trade unrestricted hours in production, so this caveat does not apply to them.',
        bullet_style,
    ),
    Paragraph(
        '&bull; <b>No spread or slippage.</b> Fills use MT5 bid-only candle ranges; entries/exits assume the '
        'quoted level is reachable exactly, which is optimistic versus live execution.',
        bullet_style,
    ),
    Paragraph(
        '&bull; <b>Same-bar SL/TP tie-break assumes stop-loss fires first</b> when a single bar\'s range '
        'contains both levels — a conservative assumption that can understate performance on wide bars.',
        bullet_style,
    ),
    Paragraph(
        '&bull; <b>Position sizing for the vectorbt-derived stats</b> (Sharpe, Max Drawdown, Total Return) '
        'uses a fixed 2% of equity per trade, not the symbol\'s configured live lot size — a raw 1-unit '
        'position is negligible against starting capital on EURUSD (~1.10/unit) but many multiples of it on '
        'GER40 (~20,000/unit), making unit-sized percentage stats meaningless for higher-priced instruments. '
        'Fixed-fractional sizing keeps Sharpe/Drawdown/Total Return comparable across symbols; it is not a '
        'projection of live P&amp;L at the configured lot sizes. Realized PnL and R-multiple figures are '
        'sizing-independent and directly comparable to live risk framing.',
        bullet_style,
    ),
    Paragraph(
        '&bull; <b>Concurrency cap enforced strictly.</b> Live MT5 data shows the bot has, at times, exceeded '
        'its configured `limit_number_position` because the live position-count check does not see resting '
        'pending orders — this backtest enforces the cap exactly as configured, so it is not directly comparable '
        'to live concurrent exposure during those episodes.',
        bullet_style,
    ),
    Spacer(1, 0.15 * cm),
]

story.append(PageBreak())

# ── Per-instrument sections ──────────────────────────────────────────────
instance_items = list(results.items())
for idx, (instance_name, v) in enumerate(instance_items):
    story += section(f"{instance_name}  ({v['symbol']})")

    if "result" not in v:
        story += [Paragraph(f'Backtest failed: {v["error"]}', body_style)]
        continue

    param_rows = [
        ["Parameter", "Value"],
        ["Tag", v["tag"]],
        ["Concurrent Position Limit", str(v["limit"])],
        ["Timeframes", f"{LTF_LABEL} entries / {HTF_LABEL} zones"],
        ["Data Coverage", f'{v["start_covered"].date()} to {v["end_covered"].date()}'],
        ["Bars", f'{v["n_ltf_bars"]:,} {LTF_LABEL} / {v["n_htf_bars"]:,} {HTF_LABEL}'],
        ["Active-Hours Filter (live)", "Yes — not modeled in this backtest" if v["has_hour_filter"] else "No restriction"],
    ]
    param_tbl = Table(param_rows, colWidths=[6*cm, 9*cm])
    param_tbl.setStyle(base_table_style())
    story += [param_tbl, Spacer(1, 0.2*cm)]

    story += [equity_charts[instance_name], Spacer(1, 0.15*cm)]

    perf_rows = [
        ["Metric", "Value"],
        ["Total Signals Generated", str(v["n_trades"])],
        ["Closed Trades", str(v["n_closed"])],
        ["Win Rate", f'{fmt_num(v["win_rate"], 1)}%'],
        ["Realized PnL (price points)", signed_para(fmt_num(v["total_pnl_pts"], 5, signed=True), v["total_pnl_pts"] >= 0)],
        ["Avg R-Multiple", signed_para(f'{fmt_num(v["avg_r"], 2, signed=True)}R', v["avg_r"] >= 0 if not np.isnan(v["avg_r"]) else True)],
        ["Profit Factor", fmt_num(v["profit_factor"], 2)],
        ["vectorbt Total Return", signed_para(fmt_pct(v["total_return_pct"]), v["total_return_pct"] >= 0 if not np.isnan(v["total_return_pct"]) else True)],
        ["Sharpe Ratio", signed_para(fmt_num(v["sharpe"], 2, signed=True), v["sharpe"] >= 0 if not np.isnan(v["sharpe"]) else True)],
        ["Max Drawdown", signed_para(f'{fmt_num(v["max_dd_pct"], 2)}%', False)],
    ]
    perf_tbl = Table(perf_rows, colWidths=[9*cm, 6*cm])
    perf_tbl.setStyle(base_table_style())
    story += [perf_tbl, Spacer(1, 0.15*cm)]

    # ── PM commentary (grounded in the actual computed numbers above) ──
    sharpe = v["sharpe"]
    win_rate = v["win_rate"]
    pf = v["profit_factor"]
    n_closed = v["n_closed"]

    sample_note = (
        f'With only {n_closed} closed trades, sample size limits statistical confidence in any of the above; '
        if n_closed < 30 else
        f'With {n_closed} closed trades, the sample is large enough for a preliminary read, though still short of '
        f'what would be required for a capital allocation decision; '
    )
    verdict = (
        "shows a positive risk-adjusted edge over the backtest window and merits continued live monitoring at current size"
        if (not np.isnan(sharpe) and sharpe > 0.3) else
        "shows a marginal or flat edge — insufficient to distinguish from noise at this sample size"
        if (not np.isnan(sharpe) and -0.3 <= sharpe <= 0.3) else
        "shows a negative risk-adjusted return over the backtest window and should be reviewed before further live sizing"
    )
    hour_note = (
        f' Recall this instance restricts live trading to specific server hours, which this backtest does not '
        f'replicate — the live signal count is expected to be materially lower than the {v["n_trades"]} shown here.'
        if v["has_hour_filter"] else ""
    )

    story += [
        Paragraph(
            f'{sample_note}on the numbers above, <b>{v["symbol"]}</b> {verdict} '
            f'(Sharpe {fmt_num(sharpe, 2, signed=True)}, win rate {fmt_num(win_rate, 1)}%, '
            f'profit factor {fmt_num(pf, 2)}).{hour_note}',
            body_style,
        ),
        Spacer(1, 0.25*cm),
    ]
    if idx < len(instance_items) - 1:
        story.append(PageBreak())

# Footer on the last page's content (before doc build handles page decoration)
story += [
    HRFlowable(width="100%", thickness=0.3, color=MUTED, spaceAfter=6),
    Paragraph(f"Generated {generated_at} &middot; metafvg_backtest.py &middot; not investment advice", footer_style),
]

# =========================================================================
# 5. Render
# =========================================================================
doc = SimpleDocTemplate(
    OUT_PDF, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
    title="MetaFVG Strategy Review — Historical Backtest",
    author="metafvg_backtest.py",
)


def on_page(canvas, doc_):
    canvas.saveState()
    canvas.setFillColor(BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    canvas.restoreState()


doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"\nPDF written to: {OUT_PDF}")
