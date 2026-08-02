"""
Spearman-gate parameter sensitivity sweep: for one fixed regression_window
(passed as CLI arg, so 5 of these can run in parallel background processes
-- one per window value), loops over the threshold grid x all 14 instruments,
re-simulating trades and rebuilding the equal-weight portfolio at each
(window, threshold) combo. Reads OHLC from the pre-fetched cache built by
fetch_sensitivity_ohlc_cache.py so this only pays simulation cost, not MT5
fetch cost, per combo.

All other VariantParams are held at the values used to build the report's
headline equal-weight result (limit_number_position=2, exit_style="fixed",
invert_direction=True, trend_filter="spearman", everything else at
BacktestParams defaults) -- only regression_window and regression_r2_threshold
vary, matching the report's Spearman-gate-only sensitivity scope.

Usage:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/sweep_spearman_sensitivity.py <window>
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metalib.metafvg_backtest import build_vbt_portfolio, trades_to_dataframe  # noqa: E402
from metafvg_ab_sweep import quantstats_metrics  # noqa: E402
from metafvg_variants import VariantParams, simulate_fvg_trades_variant  # noqa: E402

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PKL = os.path.join(RESEARCH_DIR, "data", "sensitivity_ohlc_cache.pkl")
OUT_DIR = os.path.join(RESEARCH_DIR, "data", "sensitivity_sweep")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
           "US500", "GER40", "JP225", "HK50", "USDSGD", "EURNOK", "USDZAR"]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def main():
    window = int(sys.argv[1])

    with open(CACHE_PKL, "rb") as f:
        cache = pickle.load(f)
    start, end = cache["start"], cache["end"]

    results = {}
    for threshold in THRESHOLDS:
        per_symbol_rets = {}
        for symbol in SYMBOLS:
            ltf_df, htf_df = cache[symbol]["ltf_df"], cache[symbol]["htf_df"]
            params = VariantParams(symbol=symbol, start=start, end=end, limit_number_position=2,
                                    trend_filter="spearman", regression_window=window,
                                    regression_r2_threshold=threshold, exit_style="fixed",
                                    invert_direction=True)
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
        if len(blended) < 10 or blended.std() == 0:
            sharpe = float("nan")
        else:
            equity = (1 + blended).cumprod()
            m = quantstats_metrics(equity)
            sharpe = m.get("qs_sharpe", float("nan"))

        results[threshold] = {"sharpe": sharpe, "n_days": len(blended)}
        print(f"window={window} threshold={threshold}: sharpe={sharpe:.4f} n_days={len(blended)}", flush=True)

    out_path = os.path.join(OUT_DIR, f"window_{window}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
