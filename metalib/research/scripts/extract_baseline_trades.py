"""
Re-runs the baseline config (window=20, threshold=0.5, atr_sensitivity=4.0,
invert_direction=True -- the same setup behind the report's headline
Sharpe=0.518 result) across all 14 instruments and dumps every closed trade's
symbol, entry/exit prices, SL, TP, and pnl to a single pickle, for the
research report's trade-list appendix. Only the baseline point (no grid), so
this is 14 simulate_fvg_trades_variant calls, not the 25/70-call sweeps.

Usage:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/extract_baseline_trades.py
"""
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metalib.metafvg_backtest import trades_to_dataframe  # noqa: E402
from metafvg_variants import VariantParams, simulate_fvg_trades_variant  # noqa: E402

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PKL = os.path.join(RESEARCH_DIR, "data", "sensitivity_ohlc_cache.pkl")
OUT_PKL = os.path.join(RESEARCH_DIR, "data", "baseline_trades.pkl")

SYMBOLS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
           "US500", "GER40", "JP225", "HK50", "USDSGD", "EURNOK", "USDZAR"]
BASELINE_WINDOW = 20
BASELINE_THRESHOLD = 0.5
BASELINE_ATR_SENSITIVITY = 4.0


def main():
    with open(CACHE_PKL, "rb") as f:
        cache = pickle.load(f)
    start, end = cache["start"], cache["end"]

    all_trades = []
    for symbol in SYMBOLS:
        ltf_df, htf_df = cache[symbol]["ltf_df"], cache[symbol]["htf_df"]
        params = VariantParams(symbol=symbol, start=start, end=end, limit_number_position=2,
                                trend_filter="spearman", regression_window=BASELINE_WINDOW,
                                regression_r2_threshold=BASELINE_THRESHOLD, exit_style="fixed",
                                invert_direction=True, atr_sensitivity=BASELINE_ATR_SENSITIVITY)
        trades, _, _ = simulate_fvg_trades_variant(ltf_df, htf_df, params)
        trades_df = trades_to_dataframe(trades)
        closed = trades_df[trades_df.status == "closed"].copy()
        closed.insert(0, "symbol", symbol)
        all_trades.append(closed)
        print(f"{symbol}: {len(closed)} closed trades", flush=True)

    combined = pd.concat(all_trades, ignore_index=True)
    combined = combined.sort_values("entry_time").reset_index(drop=True)

    with open(OUT_PKL, "wb") as f:
        pickle.dump(combined, f)
    print(f"written: {OUT_PKL} ({len(combined)} total closed trades)", flush=True)


if __name__ == "__main__":
    main()
