"""
ATR-sensitivity sensitivity sweep, run as a 1D line (not a 2D heatmap):
`self.tp = entry + atr_value * atr_sensitivity * risk_reward * future_state`
(metalib/metafvg.py::_calculate_trade_parameters) means atr_sensitivity and
risk_reward only ever enter the trade as their product -- an
atr_sensitivity x risk_reward grid would just repeat identical trades along
every constant-product diagonal. So risk_reward is held fixed at its default
(2.0) and only atr_sensitivity varies; the Spearman gate (window, threshold)
is held at the report's baseline (20, 0.5).

Takes one atr_sensitivity value via CLI arg so 5 of these can run in
parallel background processes -- one per value. Reads OHLC from the
pre-fetched cache built by fetch_sensitivity_ohlc_cache.py.

Usage:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/sweep_atr_sensitivity.py <atr_sensitivity>
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
OUT_DIR = os.path.join(RESEARCH_DIR, "data", "sensitivity_sweep")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
           "US500", "GER40", "JP225", "HK50", "USDSGD", "EURNOK", "USDZAR"]
BASELINE_WINDOW = 20
BASELINE_THRESHOLD = 0.5


def main():
    atr_sensitivity = float(sys.argv[1])

    with open(CACHE_PKL, "rb") as f:
        cache = pickle.load(f)
    start, end = cache["start"], cache["end"]

    per_symbol_rets = {}
    for symbol in SYMBOLS:
        ltf_df, htf_df = cache[symbol]["ltf_df"], cache[symbol]["htf_df"]
        params = VariantParams(symbol=symbol, start=start, end=end, limit_number_position=2,
                                trend_filter="spearman", regression_window=BASELINE_WINDOW,
                                regression_r2_threshold=BASELINE_THRESHOLD, exit_style="fixed",
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
    if len(blended) < 10 or blended.std() == 0:
        sharpe = float("nan")
    else:
        equity = (1 + blended).cumprod()
        m = quantstats_metrics(equity)
        sharpe = m.get("qs_sharpe", float("nan"))

    result = {"sharpe": sharpe, "n_days": len(blended)}
    print(f"atr_sensitivity={atr_sensitivity}: sharpe={sharpe:.4f} n_days={len(blended)}", flush=True)

    out_path = os.path.join(OUT_DIR, f"atr_sensitivity_{atr_sensitivity}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
