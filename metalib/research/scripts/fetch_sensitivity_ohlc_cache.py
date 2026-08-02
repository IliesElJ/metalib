"""
Fetches LTF+HTF OHLC once for all 14 instruments and caches it to disk, so
the Spearman-gate parameter sensitivity sweep (sweep_spearman_sensitivity.py)
can re-simulate trades at 25 different (window, threshold) combos without
re-fetching from MT5 25 times over -- simulate_fvg_trades_variant takes
already-fetched dataframes, so fetching is the one-time cost, simulation is
the part that varies per parameter combo.

Usage:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/fetch_sensitivity_ohlc_cache.py
"""
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metalib.metafvg_backtest import (connect_mt5, fetch_ltf_htf_candles,  # noqa: E402
                                       HTF_RESAMPLE_OPTIONS, LTF_TIMEFRAME_OPTIONS)

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PKL = os.path.join(RESEARCH_DIR, "data", "sensitivity_ohlc_cache.pkl")

SYMBOLS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
           "US500", "GER40", "JP225", "HK50", "USDSGD", "EURNOK", "USDZAR"]


def main():
    connect_mt5()
    end = pd.Timestamp.utcnow().tz_localize(None)
    start = end - pd.Timedelta(days=365 * 5)

    cache = {"start": start, "end": end}
    for symbol in SYMBOLS:
        ltf_df, htf_df = fetch_ltf_htf_candles(symbol, start, end,
                                                 ltf_timeframe=LTF_TIMEFRAME_OPTIONS["M15"],
                                                 htf_resample_rule=HTF_RESAMPLE_OPTIONS["4h"])
        cache[symbol] = {"ltf_df": ltf_df, "htf_df": htf_df}
        print(f"{symbol}: ltf={len(ltf_df)} bars, htf={len(htf_df)} bars", flush=True)

    with open(OUT_PKL, "wb") as f:
        pickle.dump(cache, f)
    print(f"written: {OUT_PKL}")


if __name__ == "__main__":
    main()
