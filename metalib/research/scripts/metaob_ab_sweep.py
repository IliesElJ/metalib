"""
A/B sweep: No Trend Filter vs Old (Sharpe) Trend Filter vs New (OLS regression)
Trend Filter, across the full asset universe (metafvg_ab_universe.py, reused
unchanged - MetaOB's OB+pivot signal isn't FVG-specific, no need for a
separate universe list). Caches every (symbol, config) result - trade-level
stats, vbt stats, and quantstats risk metrics - to a pickle for the report
builder.

Fetches OHLC once per symbol (both the M15 entry series and the D1 trend
series) and reuses it across all 3 configs, rather than re-fetching per config.

Run from the metalib repo root with the adonys interpreter:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/metaob_ab_sweep.py

Sharding for parallel workers (see metafvg_ab_sweep.py for the same pattern):
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/metaob_ab_sweep.py --shard 0/4
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/metaob_ab_sweep.py --shard 1/4
    ...
"""
import argparse
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

from datetime import datetime

import numpy as np
import pandas as pd
import quantstats as qs

from metalib.metaob_backtest import (
    BacktestParams,
    LTF_TIMEFRAME,
    TREND_TIMEFRAME,
    connect_mt5,
    fetch_candles,
    run_backtest_with_data,
)
from metafvg_ab_universe import all_symbols

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

BACKTEST_END = datetime.utcnow()
BACKTEST_START = BACKTEST_END - pd.Timedelta(days=365 * 2)

CONFIGS = {
    "No Trend Filter": dict(trend_filter="none"),
    "Old Sharpe Filter": dict(trend_filter="old_sharpe"),
    "New Trend Filter": dict(trend_filter="new"),
}


def analyze_trades(trades_df: pd.DataFrame) -> dict:
    closed = trades_df[trades_df["status"] == "closed"]
    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] <= 0]
    n_closed, n_wins = len(closed), len(wins)

    win_rate = n_wins / n_closed if n_closed else float("nan")
    avg_winner_r = float(wins["r_multiple"].mean()) if n_wins else float("nan")
    breakeven_wr = (1 / (1 + avg_winner_r)) if (avg_winner_r and not np.isnan(avg_winner_r) and avg_winner_r > 0) else float("nan")
    edge_pp = (win_rate - breakeven_wr) * 100 if not np.isnan(breakeven_wr) else float("nan")
    pf = (float(wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) and losses["pnl"].sum() != 0 else float("nan"))

    return {
        "n_total": len(trades_df),
        "n_closed": n_closed,
        "n_wins": n_wins,
        "win_rate_pct": win_rate * 100 if not np.isnan(win_rate) else float("nan"),
        "avg_winner_r": avg_winner_r,
        "breakeven_wr_pct": breakeven_wr * 100 if not np.isnan(breakeven_wr) else float("nan"),
        "edge_pp": edge_pp,
        "profit_factor": pf,
    }


def quantstats_metrics(value_series: pd.Series) -> dict:
    """quantstats assumes daily-periodicity returns for its annualization
    conventions; resample the (irregular bar-spacing) equity curve to daily first."""
    daily = value_series.resample("D").last().ffill().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) < 10 or rets.std() == 0:
        return {}
    try:
        return {
            "qs_sharpe": float(qs.stats.sharpe(rets)),
            "qs_sortino": float(qs.stats.sortino(rets)),
            "qs_calmar": float(qs.stats.calmar(rets)),
            "qs_cagr": float(qs.stats.cagr(rets)),
            "qs_max_drawdown_pct": float(qs.stats.max_drawdown(rets)) * 100,
        }
    except Exception as e:
        return {"qs_error": str(e)}


def run_symbols(symbols, cache_path):
    connect_mt5()
    results = {}

    for si, symbol in enumerate(symbols):
        print(f"=== [{si + 1}/{len(symbols)}] {symbol} ===", flush=True)
        try:
            ltf_df, trend_df = fetch_candles(
                symbol, BACKTEST_START, BACKTEST_END,
                ltf_timeframe=LTF_TIMEFRAME,
                trend_timeframe=TREND_TIMEFRAME,
                trend_window=200,  # must match BacktestParams.trend_window default used below
            )
        except Exception as e:
            print(f"  fetch failed: {e}", flush=True)
            continue
        if len(ltf_df) < 500:
            print(f"  insufficient data (ltf={len(ltf_df)}), skipping", flush=True)
            continue

        for config_name, kwargs in CONFIGS.items():
            params = BacktestParams(symbol=symbol, start=BACKTEST_START, end=BACKTEST_END, **kwargs)
            try:
                result = run_backtest_with_data(ltf_df, trend_df, params)
                vbt_stats = result.stats
            except Exception as e:
                print(f"  {config_name} FAILED: {e}", flush=True)
                continue

            trade_stats = analyze_trades(result.trades_df)
            qstats = quantstats_metrics(result.portfolio.value())

            key = (symbol, config_name)
            results[key] = {
                "symbol": symbol,
                "config": config_name,
                "sharpe": vbt_stats.get("Sharpe Ratio", float("nan")),
                "max_dd_pct": vbt_stats.get("Max Drawdown [%]", float("nan")),
                "total_return_pct": vbt_stats.get("Total Return [%]", float("nan")),
                "coverage_start": ltf_df.index.min(),
                "coverage_end": ltf_df.index.max(),
                "equity_curve": result.portfolio.value(),
                **trade_stats,
                **qstats,
            }
            print(f"  {config_name}: trades={trade_stats['n_total']} closed={trade_stats['n_closed']} "
                  f"win_rate={trade_stats['win_rate_pct']:.1f}% sharpe={results[key]['sharpe']:.2f}", flush=True)

    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nCached {len(results)} (symbol, config) results to {cache_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=str, default=None, help="e.g. '0/4' = this is worker 0 of 4")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    symbols = all_symbols()
    tag_suffix = f"_{args.tag}" if args.tag else ""
    if args.shard:
        idx_str, n_str = args.shard.split("/")
        idx, n = int(idx_str), int(n_str)
        symbols = symbols[idx::n]
        cache_path = os.path.join(REPORT_DIR, f"metaob_ab_sweep_data{tag_suffix}_shard{idx}.pkl")
        print(f"Shard {idx}/{n}: {len(symbols)} symbols -> {cache_path}", flush=True)
    else:
        cache_path = os.path.join(REPORT_DIR, f"metaob_ab_sweep_data{tag_suffix}.pkl")

    run_symbols(symbols, cache_path)


if __name__ == "__main__":
    main()
