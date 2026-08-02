"""
Data-gathering pass for the PM-facing MetaFVG statistical report.

Runs H4/Weekly AND M15/4h backtests for all 4 configured instances, computes
win-rate-vs-breakeven edge estimates with Wilson score confidence intervals,
and caches everything to metafvg_pm_report_data.pkl so the (iterative, fast)
PDF-building script doesn't have to re-pay the ~2-3 min-per-instrument M15
backtest cost on every layout tweak.

Run with:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/metafvg_pm_report_data.py
"""
import math
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from metalib.metafvg_backtest import (
    BacktestParams,
    HTF_RESAMPLE_OPTIONS,
    LTF_TIMEFRAME_OPTIONS,
    PROD_HTF_LOOKBACK_BARS,
    run_backtest,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(RESEARCH_DIR))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "metalib", "config", "prod", "metafvg.yaml")
CACHE_PATH = os.path.join(RESEARCH_DIR, "data", "metafvg_pm_report_data.pkl")

BACKTEST_END = datetime.utcnow()
BACKTEST_START = BACKTEST_END - pd.Timedelta(days=365 * 5)

SCALE_RUNS = [
    ("H4/Weekly", LTF_TIMEFRAME_OPTIONS["H4"], HTF_RESAMPLE_OPTIONS["1 Week"], 52),
    ("M15/4h", LTF_TIMEFRAME_OPTIONS["M15"], HTF_RESAMPLE_OPTIONS["4h"], PROD_HTF_LOOKBACK_BARS),
]


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion -- better-behaved than the
    normal approximation at small n or p near 0/1, standard practice for win-rate
    confidence intervals."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return p, center - half, center + half


def analyze(trades_df: pd.DataFrame) -> dict:
    closed = trades_df[trades_df["status"] == "closed"]
    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] <= 0]
    n_closed = len(closed)
    n_wins = len(wins)

    avg_winner_r = float(wins["r_multiple"].mean()) if n_wins else float("nan")
    median_winner_r = float(wins["r_multiple"].median()) if n_wins else float("nan")
    max_winner_r = float(wins["r_multiple"].max()) if n_wins else float("nan")
    avg_loser_r = float(losses["r_multiple"].mean()) if len(losses) else float("nan")

    win_rate, ci_lo, ci_hi = wilson_ci(n_wins, n_closed)
    breakeven_wr = 1 / (1 + avg_winner_r) if avg_winner_r and not np.isnan(avg_winner_r) and avg_winner_r > 0 else float("nan")
    edge_pp = (win_rate - breakeven_wr) * 100 if not np.isnan(breakeven_wr) else float("nan")
    edge_ci_lo_pp = (ci_lo - breakeven_wr) * 100 if not np.isnan(breakeven_wr) else float("nan")
    edge_ci_hi_pp = (ci_hi - breakeven_wr) * 100 if not np.isnan(breakeven_wr) else float("nan")

    fill_rate = trades_df["entry_time"].notna().mean() * 100 if len(trades_df) else float("nan")

    dur = (closed["exit_time"] - closed["entry_time"]).dt.total_seconds() / 3600
    win_dur = float(dur[closed["pnl"] > 0].mean()) if n_wins else float("nan")
    loss_dur = float(dur[closed["pnl"] <= 0].mean()) if len(losses) else float("nan")

    return {
        "n_total": len(trades_df),
        "n_closed": n_closed,
        "n_wins": n_wins,
        "n_losses": len(losses),
        "win_rate_pct": win_rate * 100,
        "win_rate_ci_lo_pct": ci_lo * 100,
        "win_rate_ci_hi_pct": ci_hi * 100,
        "avg_winner_r": avg_winner_r,
        "median_winner_r": median_winner_r,
        "max_winner_r": max_winner_r,
        "avg_loser_r": avg_loser_r,
        "breakeven_wr_pct": breakeven_wr * 100 if not np.isnan(breakeven_wr) else float("nan"),
        "edge_pp": edge_pp,
        "edge_ci_lo_pp": edge_ci_lo_pp,
        "edge_ci_hi_pp": edge_ci_hi_pp,
        "fill_rate_pct": fill_rate,
        "avg_winner_duration_h": win_dur,
        "avg_loser_duration_h": loss_dur,
        "winner_r_values": wins["r_multiple"].tolist(),
        "profit_factor": float(wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) and losses["pnl"].sum() != 0 else float("nan"),
    }


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    results = {}
    for instance_name, entry in config.items():
        symbol = entry["symbols"][0]
        limit = int(entry["limit_number_position"])
        active_hours = entry.get("active_hours")
        has_hour_filter = isinstance(active_hours, list)

        for scale_label, ltf_tf, htf_rule, htf_lookback in SCALE_RUNS:
            print(f"=== {symbol} @ {scale_label} ===", flush=True)
            params = BacktestParams(
                symbol=symbol, start=BACKTEST_START, end=BACKTEST_END,
                limit_number_position=limit, ltf_timeframe=ltf_tf,
                htf_resample_rule=htf_rule, htf_lookback_bars=htf_lookback,
            )
            result = run_backtest(params)
            stats = analyze(result.trades_df)
            stats["symbol"] = symbol
            stats["instance_name"] = instance_name
            stats["scale_label"] = scale_label
            stats["has_hour_filter"] = has_hour_filter
            stats["sharpe"] = result.stats.get("Sharpe Ratio", float("nan"))
            stats["max_dd_pct"] = result.stats.get("Max Drawdown [%]", float("nan"))
            stats["total_return_pct"] = result.stats.get("Total Return [%]", float("nan"))
            stats["coverage_start"] = result.ltf_df.index.min()
            stats["coverage_end"] = result.ltf_df.index.max()
            stats["equity_curve"] = result.portfolio.value()
            results[(symbol, scale_label)] = stats
            print(f"  n_closed={stats['n_closed']} win_rate={stats['win_rate_pct']:.1f}% "
                  f"edge={stats['edge_pp']:.1f}pp Sharpe={stats['sharpe']:.2f}", flush=True)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"\nCached {len(results)} (symbol, scale) results to {CACHE_PATH}")


if __name__ == "__main__":
    main()
