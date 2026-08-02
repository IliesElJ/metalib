"""
Computes all numeric inputs for the equal-weight portfolio full report
(equity/drawdown series, quantstats metrics, static + rolling correlation,
per-symbol performance) and dumps them to a single pickle. Chart rendering
happens separately in the base conda env (scienceplots needs a matplotlib
that isn't crashed by this env's BLAS build -- see generate_equal_weight_
charts.py); PDF assembly happens separately too (build_equal_weight_report.py).

Usage:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/compute_equal_weight_report_data.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metafvg_ab_sweep import quantstats_metrics  # noqa: E402

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PKL = os.path.join(RESEARCH_DIR, "data", "meanrev_diversified_merged.pkl")
OUT_PKL = os.path.join(RESEARCH_DIR, "data", "equal_weight_report_data.pkl")

SYMBOLS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
           "US500", "GER40", "JP225", "HK50", "USDSGD", "EURNOK", "USDZAR"]

ROLL_WINDOW_DAYS = 180
ROLL_STEP_DAYS = 14
ROLL_MIN_PERIODS = 15


def rolling_avg_pairwise_corr(rets_df: pd.DataFrame) -> pd.Series:
    dates = rets_df.index
    start, end = dates.min(), dates.max()
    checkpoints = pd.date_range(start + pd.Timedelta(days=ROLL_WINDOW_DAYS), end, freq=f"{ROLL_STEP_DAYS}D")
    out = {}
    n = len(SYMBOLS)
    iu = np.triu_indices(n, k=1)
    for t in checkpoints:
        window = rets_df[(rets_df.index > t - pd.Timedelta(days=ROLL_WINDOW_DAYS)) & (rets_df.index <= t)]
        counts = window.notna().sum()
        valid_syms = counts[counts >= ROLL_MIN_PERIODS].index
        if len(valid_syms) < 3:
            continue
        corr = window[valid_syms].corr(min_periods=ROLL_MIN_PERIODS)
        vals = corr.values[np.triu_indices(len(valid_syms), k=1)]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        out[t] = float(np.mean(vals))
    return pd.Series(out).sort_index()


def main():
    with open(DATA_PKL, "rb") as f:
        data = pickle.load(f)

    rets_df = pd.DataFrame({s: data[s]["fixed"]["daily_rets"] for s in SYMBOLS})

    blended = rets_df.mean(axis=1, skipna=True).dropna()
    equity = (1 + blended).cumprod()
    drawdown = equity / equity.cummax() - 1
    metrics = quantstats_metrics(equity)

    corr_matrix = rets_df.corr()
    rolling_corr = rolling_avg_pairwise_corr(rets_df)

    per_symbol_rows = []
    for s in SYMBOLS:
        d = data[s]["fixed"]
        sym_equity = (1 + d["daily_rets"].dropna()).cumprod()
        total_return_pct = (sym_equity.iloc[-1] - 1) * 100 if len(sym_equity) else float("nan")
        per_symbol_rows.append({
            "symbol": s,
            "closed_trades": d["closed"],
            "win_rate_pct": d["wr"],
            "sharpe": d["sharpe"],
            "max_dd_pct": d["maxdd"] * 100,
            "total_return_pct": total_return_pct,
        })
    per_symbol_df = pd.DataFrame(per_symbol_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    out = {
        "equity": equity,
        "drawdown": drawdown,
        "blended_rets": blended,
        "metrics": metrics,
        "corr_matrix": corr_matrix,
        "rolling_corr": rolling_corr,
        "per_symbol_df": per_symbol_df,
        "symbols": SYMBOLS,
        "win_days_pct": (blended > 0).mean() * 100,
        "best_day_pct": blended.max() * 100,
        "worst_day_pct": blended.min() * 100,
        "total_return_pct": (equity.iloc[-1] - 1) * 100,
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f)
    print(f"written: {OUT_PKL}")
    print(f"metrics: {metrics}")
    print(f"rolling_corr points: {len(rolling_corr)}, range [{rolling_corr.index.min()}, {rolling_corr.index.max()}]")


if __name__ == "__main__":
    main()
