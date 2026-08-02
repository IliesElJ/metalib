"""
Per-instrument (univariate) + portfolio quantstats-style tearsheets for the
Spearman mean-reversion config across the diversified 14-instrument universe
(7 FX Majors + US500/GER40/JP225/HK50 + USDSGD/EURNOK/USDZAR), both fixed and
risk-normalized sizing. Reuses build_report from generate_quantstats_fx_
majors_reports.py -- same reportlab-native chart pipeline (quantstats' own
plotting crashes in this env). Univariate reports additionally get an
"Example Trades" section: one winning and one losing trade per instrument,
each rendered as an OHLC candlestick chart with the triggering HTF FVG zone
shaded and the rolling Spearman-rho series plotted underneath.

Usage:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/generate_meanrev_diversified_reports.py
"""
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime

import pandas as pd
from reportlab.platypus import Paragraph, Spacer

from generate_quantstats_fx_majors_reports import build_report, section_style, MUTED  # noqa: E402
from metalib.metafvg import MetaFVG  # noqa: E402
from metalib.metafvg_backtest import connect_mt5, fetch_ltf_htf_candles, HTF_RESAMPLE_OPTIONS, LTF_TIMEFRAME_OPTIONS  # noqa: E402
from metafvg_variants import VariantParams, simulate_fvg_trades_variant, _rolling_spearman  # noqa: E402
from _trade_chart_drawing import trade_example_drawing  # noqa: E402

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PKL = os.path.join(RESEARCH_DIR, "data", "meanrev_diversified_merged.pkl")
OUT_DIR = os.path.join(RESEARCH_DIR, "reports", "quantstats_meanrev_diversified")

SYMBOLS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
           "US500", "GER40", "JP225", "HK50", "USDSGD", "EURNOK", "USDZAR"]
SIZING_LABELS = {"fixed": "Fixed 2% Notional", "risk_sized": "Risk-Normalized Sizing"}


def _find_zone(strat, htf_df, ltf_df, htf_lookback_bars, created_time, htf_fill_pct, max_crossings):
    avail = htf_df[htf_df.index <= created_time]
    if len(avail) < 3:
        return None
    avail_w = avail.iloc[-htf_lookback_bars:]
    bull_p, bear_p = strat.detect_fvg_htf(avail_w)
    bull_res = strat._process_htf_fvg_patterns(bull_p, avail_w["low"], htf_fill_pct, "Bullish")
    bear_res = strat._process_htf_fvg_patterns(bear_p, avail_w["high"], 1 - htf_fill_pct, "Bearish")
    ref_price = ltf_df["close"].asof(created_time)
    for res in (bull_res, bear_res):
        containing = res.filtered_patterns.get_patterns_containing_price(ref_price)
        if containing:
            latest = max(containing, key=lambda p: p.timestamp)
            return (latest.gap_low, latest.gap_high)
    return None


def collect_example_trades(symbol: str, params: VariantParams):
    """Re-runs the backtest for `symbol` (trades aren't stored in the cached
    daily-returns pickle) to pick a representative winner and loser. Returns
    a dict of plain data (n_closed, examples=[(label, trade, zone, ltf_window,
    rho_window), ...]) -- NOT reportlab flowables. Flowables must be built
    fresh per report (see render_example_trades_flowables): reportlab
    Paragraph/Drawing objects pick up layout state during doc.build() and are
    not safe to reuse across two separate build() calls -- reusing them
    across the fixed/risk_sized report pairs caused a LayoutError on the
    second build even though the content was identical."""
    ltf_df, htf_df = fetch_ltf_htf_candles(symbol, params.start, params.end,
                                            ltf_timeframe=params.ltf_timeframe,
                                            htf_resample_rule=params.htf_resample_rule)
    trades, _, _ = simulate_fvg_trades_variant(ltf_df, htf_df, params)
    closed = [t for t in trades if t.status == "closed"]
    winners = [t for t in closed if t.pnl and t.pnl > 0]
    losers = [t for t in closed if t.pnl and t.pnl <= 0]
    if not winners and not losers:
        return None

    rho_series = pd.Series(_rolling_spearman(htf_df["close"].values, params.regression_window), index=htf_df.index)

    strat = MetaFVG(symbols=[symbol], timeframe=params.ltf_timeframe, size_position=1.0,
                     tag="chart", limit_number_position=params.limit_number_position)
    strat.debug = True
    strat._log = lambda *a, **k: None

    examples = []
    for label, trade in (("Winning", max(winners, key=lambda t: t.pnl) if winners else None),
                          ("Losing", min(losers, key=lambda t: t.pnl) if losers else None)):
        if trade is None:
            continue
        zone = _find_zone(strat, htf_df, ltf_df, params.htf_lookback_bars, trade.created_time,
                           params.htf_fill_pct, params.max_htf_number_crossings)
        window_start = trade.entry_time - pd.Timedelta(hours=10) if trade.entry_time else trade.created_time
        window_end = (trade.exit_time or trade.entry_time or trade.created_time) + pd.Timedelta(hours=4)
        ltf_window = ltf_df[(ltf_df.index >= window_start) & (ltf_df.index <= window_end)]
        rho_window = rho_series[(rho_series.index >= window_start - pd.Timedelta(days=2)) & (rho_series.index <= window_end)]
        if len(ltf_window) < 3:
            continue
        examples.append((label, trade, zone, ltf_window, rho_window))

    if not examples:
        return None
    return {"n_closed": len(closed), "examples": examples}


def render_example_trades_flowables(data, params: VariantParams):
    """Builds a fresh list of reportlab flowables from the plain data
    returned by collect_example_trades. Must be called separately for every
    build_report() invocation -- see the note on collect_example_trades."""
    if not data:
        return []
    flowables = [Paragraph("Example Trades", section_style),
                 Paragraph(f"One representative winner and one loser out of {data['n_closed']} closed trades "
                           f"for this instrument, showing the HTF FVG zone that triggered the setup "
                           f"(shaded) and the rolling Spearman ρ that gated it.",
                           _caption_style())]
    for label, trade, zone, ltf_window, rho_window in data["examples"]:
        title = f"{label} example (R={trade.r_multiple:.2f})" if trade.r_multiple is not None else f"{label} example"
        drawing = trade_example_drawing(ltf_window, trade, zone, rho_window, params.regression_r2_threshold,
                                         width=480, height=300, title=title)
        flowables.append(Spacer(1, 4))
        flowables.append(drawing)
    return flowables if len(flowables) > 2 else []


def _caption_style():
    from reportlab.lib.styles import ParagraphStyle
    return ParagraphStyle("Caption", fontSize=8.5, textColor=MUTED, fontName="Helvetica", leading=11, spaceAfter=6)


def main():
    connect_mt5()
    end = datetime.utcnow()
    start = end - pd.Timedelta(days=365 * 5)

    with open(DATA_PKL, "rb") as f:
        data = pickle.load(f)

    example_trades_data = {}
    example_trades_params = {}

    for sizing_key, sizing_label in SIZING_LABELS.items():
        out_dir = os.path.join(OUT_DIR, sizing_key)
        os.makedirs(out_dir, exist_ok=True)

        per_symbol_rets = {}
        for symbol in SYMBOLS:
            rets = data[symbol][sizing_key]["daily_rets"]
            per_symbol_rets[symbol] = rets

            if symbol not in example_trades_data:
                params = VariantParams(symbol=symbol, start=start, end=end, limit_number_position=2,
                                        trend_filter="spearman", regression_window=20, regression_r2_threshold=0.5,
                                        exit_style="fixed", invert_direction=True)
                example_trades_params[symbol] = params
                try:
                    example_trades_data[symbol] = collect_example_trades(symbol, params)
                except Exception as e:
                    print(f"  {symbol}: example-trade collection failed ({e}), skipping", flush=True)
                    example_trades_data[symbol] = None

            flowables = render_example_trades_flowables(example_trades_data[symbol], example_trades_params[symbol])
            out_path = os.path.join(out_dir, f"{symbol.lower()}.pdf")
            build_report(f"{symbol} -- Spearman Gate (mean-reversion)", rets, out_path,
                         subtitle=f"MetaFVG -- Spearman mean-reversion -- M15/4h -- {sizing_label}",
                         extra_flowables=flowables)
            print(f"written: {out_path}", flush=True)

        rets_df = pd.DataFrame(per_symbol_rets)
        blended = rets_df.mean(axis=1, skipna=True).dropna()
        out_path = os.path.join(out_dir, "portfolio.pdf")
        build_report(f"Diversified Portfolio -- Spearman Gate (mean-reversion)", blended, out_path,
                     subtitle=f"MetaFVG -- Spearman mean-reversion -- M15/4h -- {sizing_label} -- "
                              f"equal-weighted, 14 instruments (7 FX Majors + US500/GER40/JP225/HK50 "
                              f"+ USDSGD/EURNOK/USDZAR)")
        print(f"written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
