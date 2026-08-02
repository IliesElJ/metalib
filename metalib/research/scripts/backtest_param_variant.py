"""
Runs a full equal-weight 14-instrument portfolio backtest at one specific
(window, threshold, atr_sensitivity) combo and saves the full blended daily-
returns/equity/drawdown series (not just the scalar Sharpe the sensitivity
sweeps kept) to a pickle, keyed by a slug -- used for the "Exploring
Outperforming Parameter Sets" section, which overlays equity/drawdown curves
for a few standout combos from the sensitivity sweep against the baseline.

Reads OHLC from the pre-fetched cache built by fetch_sensitivity_ohlc_cache.py.

Usage:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/backtest_param_variant.py <slug> <window> <threshold> <atr_sensitivity>
"""
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metalib.metafvg_backtest import build_vbt_portfolio, trades_to_dataframe  # noqa: E402
from metafvg_ab_sweep import quantstats_metrics  # noqa: E402
from metafvg_variants import VariantParams, simulate_fvg_trades_variant  # noqa: E402

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PKL = os.path.join(RESEARCH_DIR, "data", "sensitivity_ohlc_cache.pkl")
OUT_DIR = os.path.join(RESEARCH_DIR, "data", "param_variants")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
           "US500", "GER40", "JP225", "HK50", "USDSGD", "EURNOK", "USDZAR"]


def main():
    slug = sys.argv[1]
    window = int(sys.argv[2])
    threshold = float(sys.argv[3])
    atr_sensitivity = float(sys.argv[4])

    with open(CACHE_PKL, "rb") as f:
        cache = pickle.load(f)
    start, end = cache["start"], cache["end"]

    per_symbol_rets = {}
    for symbol in SYMBOLS:
        ltf_df, htf_df = cache[symbol]["ltf_df"], cache[symbol]["htf_df"]
        params = VariantParams(symbol=symbol, start=start, end=end, limit_number_position=2,
                                trend_filter="spearman", regression_window=window,
                                regression_r2_threshold=threshold, exit_style="fixed",
                                invert_direction=True, atr_sensitivity=atr_sensitivity)
        trades, _, _ = simulate_fvg_trades_variant(ltf_df, htf_df, params)
        trades_df = trades_to_dataframe(trades)
        closed = trades_df[trades_df.status == "closed"]
        if len(closed) == 0:
            per_symbol_rets[symbol] = pd.Series(dtype=float)
            continue
        portfolio = build_vbt_portfolio(ltf_df, trades, params.limit_number_position)
        value = portfolio.value()
        daily_rets = value.resample("D").last().ffill().dropna().pct_change().dropna()
        per_symbol_rets[symbol] = daily_rets

    rets_df = pd.DataFrame(per_symbol_rets)
    blended = rets_df.mean(axis=1, skipna=True).dropna()
    equity = (1 + blended).cumprod()
    drawdown = equity / equity.cummax() - 1
    metrics = quantstats_metrics(equity)

    result = {
        "slug": slug, "window": window, "threshold": threshold, "atr_sensitivity": atr_sensitivity,
        "blended_rets": blended, "equity": equity, "drawdown": drawdown, "metrics": metrics,
        "win_days_pct": (blended > 0).mean() * 100,
        "total_return_pct": (equity.iloc[-1] - 1) * 100,
    }
    out_path = os.path.join(OUT_DIR, f"{slug}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"{slug} (w={window}, tau={threshold}, atr={atr_sensitivity}): "
          f"sharpe={metrics.get('qs_sharpe', float('nan')):.4f}", flush=True)
    print(f"written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
