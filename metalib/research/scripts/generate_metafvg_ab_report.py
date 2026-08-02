"""
MetaFVG A/B Trend-Filter Review — bucketed by asset class.

Loads metafvg_ab_sweep_data.pkl (run metafvg_ab_sweep.py first) and renders a
PDF comparing Baseline vs. ADX-gate / regression-gate / ATR-trailing-stop
variants across the full asset universe, bucketed by asset class, with a
fundamentals-grounded narrative for why buckets/configs over- or under-perform.

Run from the metalib repo root with the adonys interpreter:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/metafvg_ab_sweep.py          # once, ~30-60 min
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/generate_metafvg_ab_report.py
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
DATA_DIR = os.path.join(RESEARCH_DIR, "data")
REPORTS_DIR = os.path.join(RESEARCH_DIR, "reports")

# Must match the LTF/HTF the sweep was run with (metafvg_ab_sweep.py's own
# METAFVG_LTF/METAFVG_HTF env vars + its _SLUG derivation), so the report
# loads the matching cache file rather than silently falling back to defaults.
RUN_LTF_LABEL = os.environ.get("METAFVG_LTF", "H4")
RUN_HTF_LABEL = os.environ.get("METAFVG_HTF", "1 Week")
_SLUG = f"{RUN_LTF_LABEL}_{RUN_HTF_LABEL}".lower().replace(" ", "").replace("'", "")

CACHE_PATH = os.path.join(DATA_DIR, f"metafvg_ab_sweep_data_{_SLUG}.pkl")
OUT_PDF = os.path.join(REPORTS_DIR, f"metafvg_ab_review_{_SLUG}.pdf")

with open(CACHE_PATH, "rb") as f:
    DATA = pickle.load(f)

SYM2BUCKET = symbol_to_bucket()
# Dynamic: "Lasso Trend Gate" only has data at M15/4h (metafvg_ab_sweep.py
# skips it outright for H4/Weekly -- too few HTF bars for a walk-forward
# fit), so including it unconditionally would render an all-"n/a" column on
# the H4/Weekly report. Keep a config only if at least one symbol has data
# for it in whichever cache this run actually loaded.
_ALL_CONFIGS = ["Baseline", "ADX Gate", "Regression Gate", "ATR Trailing", "Lasso Trend Gate", "Lasso Trend Gate (tight)", "Decision Tree Gate", "Spearman Gate", "Theil-Sen Gate", "Spearman Gate (0.82)", "Spearman Gate (tight)", "Baseline + Risk Sizing", "Regression Gate + Risk Sizing"]
CONFIGS = [cfg for cfg in _ALL_CONFIGS if any((sym, cfg) in DATA for sym in symbol_to_bucket())]
BUCKETS = list(UNIVERSE.keys())

# =========================================================================
# House style
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
CONFIG_COLORS = [BLUE, AMBER, PURPLE, GREEN, HexColor("#00bcd4"), HexColor("#ffeb3b"), HexColor("#ff7043"), HexColor("#8d6e63"), HexColor("#ec407a"), HexColor("#7e57c2"), HexColor("#9ccc65"), HexColor("#5c6bc0"), HexColor("#26a69a")]  # 5th cyan, 6th yellow, 7th deep orange, 8th brown, 9th pink, 10th indigo-violet, 11th lime, 12th indigo, 13th teal -- all distinct from the semantic pos/neg colors used elsewhere on the page

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
    # Legend can outgrow one row once there are 5-6 configs with names as long
    # as "Lasso Trend Gate (tight)" -- wrap into as many rows as needed rather
    # than let entries run off the right edge of the drawing.
    entry_width = lambda cfg: 14 + 6.3 * (len(cfg) + 2)
    legend_rows, cur_row, cur_x = [], [], 2.0
    for cfg, color in zip(CONFIGS, CONFIG_COLORS):
        w = entry_width(cfg)
        if cur_row and cur_x + w > width - 2:
            legend_rows.append(cur_row)
            cur_row, cur_x = [], 2.0
        cur_row.append((cfg, color, cur_x))
        cur_x += w
    if cur_row:
        legend_rows.append(cur_row)
    legend_row_h = 11
    title_y_offset = 12   # title baseline = height - title_y_offset
    title_legend_gap = 14  # baseline-to-baseline clearance: 10.5pt title vs 7.3pt legend text
    legend_h = len(legend_rows) * legend_row_h
    header_h = title_y_offset + title_legend_gap + legend_h
    height = max(height, header_h + 130)  # keep the plot area usable even with a tall legend

    d = Drawing(width, height)
    d.add(String(2, height - title_y_offset, title, fillColor=TEXT, fontSize=10.5, fontName="Helvetica-Bold"))

    legend_top = height - title_y_offset - title_legend_gap
    for r, row in enumerate(legend_rows):
        ly = legend_top - r * legend_row_h
        for cfg, color, lx in row:
            d.add(Line(lx, ly + 3, lx + 10, ly + 3, strokeColor=color, strokeWidth=2.5))
            d.add(String(lx + 14, ly, cfg, fillColor=TEXT, fontSize=7.3, fontName="Helvetica"))

    bc = VerticalBarChart()
    bc.x = 40
    bc.y = 38
    bc.width = width - 55
    bc.height = height - header_h - 40
    bc.data = [[bucket_values[b].get(cfg, 0.0) for b in BUCKETS] for cfg in CONFIGS]
    bc.categoryAxis.categoryNames = [b.replace(" ", "\n") for b in BUCKETS]
    bc.categoryAxis.labels.fontSize = 6.3
    bc.categoryAxis.labels.fillColor = MUTED
    # Bars can go negative, and reportlab draws category labels at the zero
    # line by default -- push them below the plot area entirely so they don't
    # sit on top of (or collide with) any bar that dips below zero.
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
    return d


# =========================================================================
# Aggregate the cached sweep data
# =========================================================================
def get(sym, cfg, key, default=np.nan):
    r = DATA.get((sym, cfg))
    return r.get(key, default) if r else default


bucket_config_agg = {}  # bucket -> config -> {metric: value}
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
    Paragraph("MetaFVG A/B Trend-Filter Review", title_style),
    Paragraph("Baseline vs. ADX / Regression Trend Gates &amp; ATR Trailing Stop, by Asset Class", subtitle_style),
    HRFlowable(width="50%", thickness=0.5, color=MUTED, spaceAfter=6),
    Paragraph(
        f'{n_symbols_tested} instruments across {len(BUCKETS)} asset-class buckets &nbsp;|&nbsp; '
        f'{RUN_LTF_LABEL} &rarr; {RUN_HTF_LABEL} &nbsp;|&nbsp; Generated {generated_at}',
        meta_style,
    ),
    Spacer(1, 0.35 * cm),
]

story += section("Executive Summary")
story += [
    Paragraph(
        f'{len(CONFIGS)} configurations were backtested across {n_symbols_tested} instruments spanning FX majors/'
        f'minors/exotics, metals, energy, softs, equity indices and crypto: the unmodified <b>Baseline</b> signal, '
        f'an <b>ADX(14) trend-strength gate</b> (reject signals when HTF ADX is below 20), a <b>rolling-regression '
        f'trend gate</b> (reject signals unless a 20-bar HTF OLS fit has R&sup2; &ge; 0.5 and slope matching '
        f'direction), an <b>ATR trailing stop</b> replacing the fixed ATR take-profit'
        + (', and a <b>walk-forward, L1-regularized (Lasso) trend gate</b> adding two-window realized-volatility '
           'features and trend&times;volatility interactions on top of the rolling-regression gate (see Section 4 '
           'for the fitting methodology)'
           + (' — tested at two magnitude-gating thresholds, "Lasso Trend Gate" (looser, admits more signals) and '
              '"Lasso Trend Gate (tight)" (calibrated to a trade count matching Regression Gate\'s, for a direct '
              'apples-to-apples Sharpe comparison)' if "Lasso Trend Gate (tight)" in CONFIGS else '')
           if "Lasso Trend Gate" in CONFIGS else '') + '.',
        body_style,
    ),
]
story.append(callout_box(
    Paragraph(
        f'<b>Headline finding:</b> <font color="#{CONFIG_COLORS[CONFIGS.index(best_overall[0])].hexval()[2:]}">'
        f'<b>{best_overall[0]}</b></font> has the highest average Sharpe across the full universe '
        f'({fmt(best_overall[1]["avg_sharpe"],2,signed=True)}, '
        f'{fmt(best_overall[1]["pct_symbols_positive_sharpe"],0,suffix="%")} '
        f'of instruments Sharpe-positive). Performance is highly bucket-dependent, not uniform — see Section 2 '
        f'for which configuration wins in each asset class and why.',
        callout_style,
    ),
    border_color=BLUE,
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
overall_tbl = Table(overall_rows, colWidths=[2.7*cm, 2.1*cm, 2.0*cm, 1.9*cm, 1.9*cm, 1.6*cm, 1.8*cm, 2.0*cm])
overall_tbl.setStyle(base_table_style())
story += [overall_tbl, Spacer(1, 0.2*cm)]

story.append(PageBreak())

# ── Section 2: bucket breakdown ─────────────────────────────────────────
story += section("2. Performance by Asset-Class Bucket")
sharpe_by_bucket = {b: {cfg: bucket_config_agg[b][cfg]["avg_sharpe"] for cfg in CONFIGS} for b in BUCKETS}
story += [grouped_bar_chart("Average Sharpe Ratio by Bucket and Configuration", sharpe_by_bucket), Spacer(1, 0.15*cm)]

bucket_rows = [["Bucket", "Winner", "Winner Sharpe", "Baseline Sharpe", "N Symbols", "Closed Trades (all cfgs)"]]
for b in BUCKETS:
    winner_cfg, winner_sharpe = bucket_winner[b]
    baseline_sharpe = bucket_config_agg[b]["Baseline"]["avg_sharpe"]
    n_sym = bucket_config_agg[b]["Baseline"]["n_symbols"]
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

# ── Section 3: fundamental narrative per bucket ─────────────────────────
story += section("3. Fundamental Read, by Bucket")

BUCKET_NARRATIVE = {
    "FX Majors": 'Deep, continuously two-way flow with heavy mean-reversion pressure from rate-differential carry '
                 'and central-bank positioning; clean multi-week directional trends are comparatively rare, which '
                 'is exactly the regime an R&sup2;/ADX trend gate is designed to filter for.',
    "FX Minors": 'Similar liquidity character to majors but with more idiosyncratic cross-pair drift (funding-currency '
                 'exposure, e.g. JPY/CHF crosses) — trend persistence tends to be somewhat more regime-dependent.',
    "FX Exotics": 'Thinner liquidity and more event-driven (central bank intervention, capital controls) than majors — '
                   'moves can be sharper and more discontinuous, which can help or hurt a breakout-continuation '
                   'strategy depending on whether the discontinuity is the entry trigger or the stop-out.',
    "Metals": 'Gold/silver carry a structural macro-hedge bid (real yields, inflation expectations, USD) that can '
              'produce more persistent directional runs than FX; industrial metals (copper) trade more on '
              'growth/demand cycles.',
    "Energy": 'Supply-shock and inventory-driven, with strong trending behavior around genuine regime shifts '
              '(OPEC decisions, geopolitical supply risk) punctuated by range-bound consolidation — a trend gate '
              'should meaningfully change the signal mix here.',
    "Softs": 'Weather- and harvest-cycle driven; trends can be seasonal and slow-building, which favors an HTF '
             'trend-confirmation filter, but liquidity is thinner and gaps more common.',
    "Indices": 'Broad equity indices trend on macro/earnings regime and carry momentum/flow-driven persistence '
               '(systematic strategies chasing the same signal) — historically one of the more trend-following-'
               'friendly asset classes.',
    "Crypto": 'Retail-flow and momentum-dominated with limited fundamental anchor; trends can be strong but also '
              'reflexive and prone to sharp reversal, so trend-confirmation gates cut both ways.',
}

for b in BUCKETS:
    story += [Paragraph(b, subsection_style)]
    winner_cfg, winner_sharpe = bucket_winner[b]
    baseline = bucket_config_agg[b]["Baseline"]
    winner = bucket_config_agg[b].get(winner_cfg, {}) if winner_cfg else {}
    delta = (winner.get("avg_sharpe", np.nan) - baseline["avg_sharpe"]) if winner_cfg and winner_cfg != "Baseline" else 0.0
    verdict = (
        f'no configuration meaningfully beat baseline (best: {winner_cfg}, {fmt(delta,2,signed=True)} Sharpe vs. baseline)'
        if winner_cfg == "Baseline" or abs(delta) < 0.05 else
        f'<b>{winner_cfg}</b> improved average Sharpe by {fmt(delta,2,signed=True)} over baseline '
        f'({fmt(baseline["avg_sharpe"],2,signed=True)} &rarr; {fmt(winner.get("avg_sharpe"),2,signed=True)})'
    )
    story += [
        Paragraph(BUCKET_NARRATIVE.get(b, ""), body_style),
        Paragraph(f'<b>Result:</b> {verdict}, on {baseline["n_symbols"]} instruments and '
                   f'{sum(bucket_config_agg[b][c]["total_closed_trades"] for c in CONFIGS)} total closed trades '
                   f'across all {len(CONFIGS)} configurations tested.', body_style),
    ]

story.append(PageBreak())

# ── Section 4: caveats ───────────────────────────────────────────────────
story += section("4. Methodology &amp; Caveats")
story += [
    Paragraph(
        (f'&bull; <b>{RUN_LTF_LABEL} &rarr; {RUN_HTF_LABEL} only.</b> This sweep does not repeat the earlier PM '
         'review\'s large-sample M15/4h validation — with 47 instruments, an M15 sweep would take much longer. '
         'Bucket winners here should be treated as a first-pass screen, not a final verdict, especially for '
         'buckets with few closed trades.')
        if RUN_LTF_LABEL != "M15" else
        (f'&bull; <b>{RUN_LTF_LABEL} &rarr; {RUN_HTF_LABEL}.</b> This is the large-sample re-run of the '
         'H4 &rarr; Weekly sweep, at the same scale used for the earlier single-instrument PM review\'s '
         'validation — bucket/config winners here rest on substantially more closed trades per instrument than '
         'the H4 &rarr; Weekly pass and should be weighted more heavily where the two disagree.'),
        bullet_style,
    ),
    Paragraph(f'&bull; <b>ADX and regression gates use the same HTF ({RUN_HTF_LABEL}) series</b> as the '
               'zone-detection logic, recomputed at the same point-in-time cadence — no lookahead.', bullet_style),
    Paragraph(f'&bull; <b>Regression Gate trade-count caution.</b> Requiring R&sup2; &ge; 0.5 on a 20-bar HTF fit is '
               f'a strict filter — it cut average closed trades per symbol to '
               f'{fmt(overall_config_agg["Regression Gate"]["avg_closed_trades_per_symbol"],1)} vs. '
               f'{fmt(overall_config_agg["Baseline"]["avg_closed_trades_per_symbol"],1)} for Baseline (see the '
               '"Trades/Sym" column above), and several individual instruments closed fewer than 5 trades under '
               'this config. Its Sharpe/edge figures — both overall and per-bucket — should be read with '
               'materially more sample-size skepticism than the higher-trade-count configurations.', bullet_style),
    Paragraph('&bull; <b>ATR trailing stop</b> only ever tightens the stop in the trade\'s favor and removes the '
               'fixed take-profit entirely — it is a genuinely different exit distribution, not a superset of the '
               'baseline.', bullet_style),
] + ([
    Paragraph(
        '&bull; <b>Lasso Trend Gate methodology.</b> A walk-forward-fit, L1-regularized (Lasso) upgrade to '
        'Regression Gate: adds two-window realized-volatility features and a small curated set of trend '
        '&times; volatility interaction terms on top of the existing rolling-OLS slope/R&sup2;, fit via numba '
        'coordinate-descent (this environment\'s BLAS crashes on any matrix-matrix multiply, so sklearn is not '
        'usable here). Refit periodically on a trailing window of HTF bars; a training example is only used once '
        'its own forward-return outcome has fully realized before the fit\'s cutoff bar, and the gating magnitude '
        'threshold is a causal (past-only) rolling quantile of the prediction series\' own history — both '
        'deliberately conservative choices to avoid lookahead in a fitted-model gate, not just a fixed-formula one. '
        'Only run at M15/4h: the walk-forward fit needs far more HTF bars than the ~260 Weekly bars over 5 years '
        'provide, so this config is absent from any H4 &rarr; Weekly comparison.',
        bullet_style,
    ),
] + ([
    Paragraph(
        '&bull; <b>Why two Lasso variants.</b> "Lasso Trend Gate" uses the gate\'s default magnitude threshold '
        '(70th percentile of the prediction series\' own causal history), which admits substantially more signals '
        'than Regression Gate — an unmatched trade count makes a direct Sharpe comparison misleading, since a '
        'looser filter mechanically dilutes average trade quality regardless of whether the underlying model has '
        'any real edge. "Lasso Trend Gate (tight)" raises the threshold to the 90th percentile, calibrated on a '
        '4-symbol probe (EURUSD, BTCUSD, XAUUSD, US500) that landed at ~48 average closed trades/symbol on that '
        'probe — the closest achievable match to Regression Gate\'s own ~52.4. Realized across the full '
        '47-instrument universe the tight variant came in lower still, ~32.9 avg closed trades/symbol — the '
        '4-symbol probe skewed toward more liquid instruments than the full universe average, so even the '
        '"tight" comparison is not a perfect trade-count match, only a much closer one than the untuned default.',
        bullet_style,
    ),
] if "Lasso Trend Gate (tight)" in CONFIGS else []) if "Lasso Trend Gate" in CONFIGS else []) + [
    Paragraph('&bull; <b>quantstats metrics</b> (Sharpe, Sortino, Calmar, CAGR, Ulcer Index, Kelly, VaR/CVaR, tail '
               'ratio) are computed on the vectorbt equity curve resampled to daily, per quantstats\' own '
               'annualization convention — reported per (symbol, config) in the underlying cache '
               '(metafvg_ab_sweep_data.pkl) even though only the vbt/trade-level metrics are tabulated in this PDF '
               'for space.', bullet_style),
    Paragraph('&bull; <b>Same position-sizing, spread/slippage, and concurrency-cap caveats</b> as the earlier PM '
               'review apply here unchanged.', bullet_style),
]

story += [
    Spacer(1, 0.3*cm),
    HRFlowable(width="100%", thickness=0.3, color=MUTED, spaceAfter=6),
    Paragraph(f"Generated {generated_at} &middot; metafvg_ab_sweep.py &middot; not investment advice", footer_style),
]

# =========================================================================
# Render
# =========================================================================
doc = SimpleDocTemplate(
    OUT_PDF, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
    title="MetaFVG A/B Trend-Filter Review",
    author="metafvg_ab_sweep.py",
)


def on_page(canvas, doc_):
    canvas.saveState()
    canvas.setFillColor(BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    canvas.restoreState()


doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF written to: {OUT_PDF}")
