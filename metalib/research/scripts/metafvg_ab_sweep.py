"""
A/B sweep: baseline MetaFVG vs. ADX-gate / regression-gate / ATR-trailing-stop
variants, across the full asset universe (metafvg_ab_universe.py). Caches
every (symbol, config) result -- trade-level stats, vbt stats, and quantstats
risk metrics -- to a pickle for the report builder.

Timeframes (env vars, default H4 -> 1 Week):
    METAFVG_LTF=M15 METAFVG_HTF="4h" PYTHONPATH=. "<adonys python>" metalib/research/scripts/metafvg_ab_sweep.py

Sharding for parallel workers -- run several of these at once, each on a
disjoint slice of the symbol list, then merge with metafvg_ab_sweep_merge.py:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/metafvg_ab_sweep.py --shard 0/8
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/metafvg_ab_sweep.py --shard 1/8
    ...
Each shard writes its own cache file (metafvg_ab_sweep_data_<tf>_shard<i>.pkl);
MT5's Python API supports multiple independent client connections to the same
running terminal, so parallel processes are safe here -- the simulation loop
itself is pure Python/numba (no shared state) once data is fetched.
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

from metalib.metafvg_backtest import (
    HTF_RESAMPLE_OPTIONS,
    LTF_TIMEFRAME_OPTIONS,
    PROD_HTF_LOOKBACK_BARS,
    build_vbt_portfolio,
    connect_mt5,
    fetch_ltf_htf_candles,
    trades_to_dataframe,
)
from metafvg_ab_universe import all_symbols
from metafvg_lasso_gate import compute_lasso_gate_series
from metafvg_tree_gate import compute_tree_gate_series
from metafvg_variants import VariantParams, build_vbt_portfolio_risk_sized, simulate_fvg_trades_variant

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

LTF_LABEL = os.environ.get("METAFVG_LTF", "H4")
HTF_LABEL = os.environ.get("METAFVG_HTF", "1 Week")
RUN_LTF_TIMEFRAME = LTF_TIMEFRAME_OPTIONS[LTF_LABEL]
RUN_HTF_RESAMPLE_RULE = HTF_RESAMPLE_OPTIONS[HTF_LABEL]
RUN_HTF_LOOKBACK_BARS = PROD_HTF_LOOKBACK_BARS if HTF_LABEL != "1 Week" else 52
_SLUG = f"{LTF_LABEL}_{HTF_LABEL}".lower().replace(" ", "").replace("'", "")

BACKTEST_END = datetime.utcnow()
BACKTEST_START = BACKTEST_END - pd.Timedelta(days=365 * 5)
CONCURRENCY_LIMIT = 2

CONFIGS = {
    "Baseline": dict(trend_filter="none", exit_style="fixed"),
    "ADX Gate": dict(trend_filter="adx", adx_period=14, adx_threshold=20.0, exit_style="fixed"),
    "Regression Gate": dict(trend_filter="regression", regression_window=20, regression_r2_threshold=0.5, exit_style="fixed"),
    "ATR Trailing": dict(trend_filter="none", exit_style="atr_trailing", trail_atr_mult=3.0),
    # Walk-forward, regularized (Lasso) upgrade to Regression Gate: adds
    # two-window realized-vol features + curated interactions, fit
    # out-of-sample via metafvg_lasso_gate.py. Needs a much larger HTF bar
    # count than Weekly bars provide (~260 over 5yr) for a meaningful
    # walk-forward fit -- run_symbols() below skips this config outright when
    # HTF_LABEL == "1 Week" rather than silently producing an all-empty result.
    # threshold_quantile=0.7 (default) admits the top 30% of |prediction|
    # magnitudes -- looser than Regression Gate's R^2>=0.5 filter (~97.6 vs.
    # ~52.4 avg closed trades/symbol), so its lower Sharpe isn't a clean
    # apples-to-apples comparison. "(tight)" below uses threshold_quantile=0.9,
    # calibrated on a 4-symbol probe to land at ~48 avg closed trades/symbol --
    # closest match to Regression Gate's own count -- to test whether the
    # signal quality is real once matched for trade frequency, or whether the
    # first result was just a looser-filter artifact.
    "Lasso Trend Gate": dict(trend_filter="lasso", exit_style="fixed"),
    "Lasso Trend Gate (tight)": dict(trend_filter="lasso", exit_style="fixed", threshold_quantile=0.9),
    # Third interpretable alternative: a *short* (max_depth=3) walk-forward
    # decision tree, same feature set as the Lasso gates (metafvg_tree_gate.py
    # reuses metafvg_lasso_gate.build_features directly -- isolates model
    # choice from feature choice). Classification framing (predict forward-
    # return sign), gated on predict_proba clearing a causal quantile
    # threshold. Calibrated on a 4-symbol probe (EURUSD/BTCUSD/XAUUSD/US500,
    # M15/4h) across q in {0.6..0.85}: unlike the Lasso gate, tightening the
    # tree's confidence threshold did NOT improve Sharpe -- avg Sharpe was
    # highest at the loosest quantile tested (q=0.6, ~0.40 avg, ~211 avg
    # trades/symbol) and degraded as trades were filtered down. The tree's
    # predict_proba distribution isn't shaped like Lasso's |predicted return|
    # magnitude, so a tight-quantile match to Regression Gate's trade count
    # isn't meaningful here -- q=0.6 is used as the best-performing setting
    # found, not a trade-count-matched one.
    "Decision Tree Gate": dict(trend_filter="tree", exit_style="fixed", threshold_quantile=0.6),
    # Two rank/robust-statistics alternatives to Regression Gate, isolating
    # its two weaknesses separately (same window=20, same 0.5 threshold, so
    # this is an apples-to-apples swap-in comparison, not a re-tuned one):
    #   - Spearman Gate: OLS slope+R^2 both replaced by rolling Spearman rank
    #     correlation (sign = direction, |rho| = cleanliness threshold). Tests
    #     whether a nonlinear-but-monotonic move (that OLS/R^2 would dock for
    #     curvature) is actually fine, or was rightly penalized.
    #   - Theil-Sen Gate: keeps the OLS R^2 cleanliness gate as-is, but swaps
    #     the direction call from OLS slope to the Theil-Sen median-pairwise
    #     slope, robust to a single outlier/tail bar flipping the read.
    "Spearman Gate": dict(trend_filter="spearman", regression_window=20, regression_r2_threshold=0.5, exit_style="fixed"),
    "Theil-Sen Gate": dict(trend_filter="theilsen", regression_window=20, regression_r2_threshold=0.5, exit_style="fixed"),
    # Spearman Gate at 0.5 was 4.7x looser than Regression Gate (245.6 vs 52.4
    # avg closed trades/symbol) -- not an apples-to-apples comparison. A
    # 4-symbol probe across rho thresholds 0.5-0.95 found avg Sharpe rising
    # sharply as the gate tightens, peaking around rho=0.82 (avg Sharpe 0.79,
    # ~83.5 avg closed trades -- still looser than Regression Gate) before
    # falling off by rho=0.87. rho=0.85 (~57.0 avg closed) is the closest
    # trade-count match to Regression Gate's 52.4, but scored lower on the
    # probe (0.38) -- the two picks disagree, and 4 symbols is too small a
    # probe to trust which is signal vs. noise (one symbol, EURUSD, swung
    # from +0.27 to -0.53 between 0.82 and 0.85 alone). Running both at full
    # 47-symbol scale rather than betting on the probe.
    "Spearman Gate (0.82)": dict(trend_filter="spearman", regression_window=20, regression_r2_threshold=0.82, exit_style="fixed"),
    "Spearman Gate (tight)": dict(trend_filter="spearman", regression_window=20, regression_r2_threshold=0.85, exit_style="fixed"),
    # Position sizing, not entry/exit: the validated baseline sizes every
    # trade at a fixed 2% of equity notional regardless of stop distance, and
    # stop distance empirically spans a 100x+ range even within one symbol
    # (EURUSD 0.008%-0.978% of price) -- so realized risk-per-trade is just as
    # inconsistent. build_vbt_portfolio_risk_sized instead sizes each trade so
    # a stop-out always costs the same ~fraction of equity (risk_fraction /
    # stop_distance_pct, capped), calibrated so a median-stop-distance trade
    # sizes out near the baseline's 2%. Same underlying trades as Baseline /
    # Regression Gate -- only the size array differs. A 5-symbol probe showed
    # this is the first change all campaign to move average Sharpe by more
    # than noise: Baseline 0.355->0.436, Regression Gate 0.467->0.611 (one
    # symbol, BTCUSD, crossed 1.0 Sharpe combined with Regression Gate), with
    # max drawdown shrinking in every single probed case, often 2-5x.
    "Baseline + Risk Sizing": dict(trend_filter="none", exit_style="fixed", risk_sized=True),
    "Regression Gate + Risk Sizing": dict(trend_filter="regression", regression_window=20, regression_r2_threshold=0.5, exit_style="fixed", risk_sized=True),
    "Spearman Gate + Risk Sizing": dict(trend_filter="spearman", regression_window=20, regression_r2_threshold=0.5, exit_style="fixed", risk_sized=True),
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
        "fill_rate_pct": trades_df["entry_time"].notna().mean() * 100 if len(trades_df) else float("nan"),
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
            "qs_volatility_pct": float(qs.stats.volatility(rets)) * 100,
            "qs_tail_ratio": float(qs.stats.tail_ratio(rets)),
            "qs_ulcer_index": float(qs.stats.ulcer_index(rets)),
            "qs_kelly": float(qs.stats.kelly_criterion(rets)),
            "qs_var_pct": float(qs.stats.var(rets)) * 100,
            "qs_cvar_pct": float(qs.stats.cvar(rets)) * 100,
            "qs_skew": float(qs.stats.skew(rets)),
            "qs_recovery_factor": float(qs.stats.recovery_factor(rets)),
        }
    except Exception as e:
        return {"qs_error": str(e)}


def run_symbols(symbols, cache_path):
    connect_mt5()
    results = {}

    for si, symbol in enumerate(symbols):
        print(f"=== [{si + 1}/{len(symbols)}] {symbol} ===", flush=True)
        try:
            ltf_df, htf_df = fetch_ltf_htf_candles(
                symbol, BACKTEST_START, BACKTEST_END,
                ltf_timeframe=RUN_LTF_TIMEFRAME, htf_resample_rule=RUN_HTF_RESAMPLE_RULE,
            )
        except Exception as e:
            print(f"  fetch failed: {e}", flush=True)
            continue
        if len(ltf_df) < 100 or len(htf_df) < 20:
            print(f"  insufficient data (ltf={len(ltf_df)}, htf={len(htf_df)}), skipping", flush=True)
            continue

        for config_name, kwargs in CONFIGS.items():
            trend_filter = kwargs.get("trend_filter")
            if trend_filter in ("lasso", "tree") and HTF_LABEL == "1 Week":
                # ~260 Weekly HTF bars over 5yr is far too few for a
                # walk-forward fit (default train_bars=450) -- skip outright
                # rather than silently record an all-empty result.
                print(f"  {config_name}: skipped (HTF={HTF_LABEL} too coarse for walk-forward fit)", flush=True)
                continue

            # threshold_quantile (fitted-gate-only) and risk_sized (portfolio-
            # construction-only) aren't VariantParams fields -- pop before
            # constructing params.
            risk_sized = kwargs.get("risk_sized", False)
            variant_kwargs = {k: v for k, v in kwargs.items() if k not in ("threshold_quantile", "risk_sized")}
            params = VariantParams(
                symbol=symbol, start=BACKTEST_START, end=BACKTEST_END,
                limit_number_position=CONCURRENCY_LIMIT,
                ltf_timeframe=RUN_LTF_TIMEFRAME, htf_resample_rule=RUN_HTF_RESAMPLE_RULE,
                htf_lookback_bars=RUN_HTF_LOOKBACK_BARS,
                **variant_kwargs,
            )
            try:
                extra = {}
                if trend_filter == "lasso":
                    lasso_pred, lasso_thr = compute_lasso_gate_series(
                        htf_df, threshold_quantile=kwargs.get("threshold_quantile", 0.7)
                    )
                    extra = dict(lasso_pred=lasso_pred, lasso_threshold=lasso_thr)
                elif trend_filter == "tree":
                    tree_pred, tree_thr = compute_tree_gate_series(
                        htf_df, threshold_quantile=kwargs.get("threshold_quantile", 0.7)
                    )
                    extra = dict(lasso_pred=tree_pred, lasso_threshold=tree_thr)
                trades, _, _ = simulate_fvg_trades_variant(ltf_df, htf_df, params, **extra)
                trades_df = trades_to_dataframe(trades)
                if risk_sized:
                    portfolio = build_vbt_portfolio_risk_sized(ltf_df, trades, params.limit_number_position)
                else:
                    portfolio = build_vbt_portfolio(ltf_df, trades, params.limit_number_position)
                vbt_stats = portfolio.stats()
            except Exception as e:
                print(f"  {config_name} FAILED: {e}", flush=True)
                continue

            trade_stats = analyze_trades(trades_df)
            qstats = quantstats_metrics(portfolio.value())

            key = (symbol, config_name)
            results[key] = {
                "symbol": symbol,
                "config": config_name,
                "sharpe": vbt_stats.get("Sharpe Ratio", float("nan")),
                "max_dd_pct": vbt_stats.get("Max Drawdown [%]", float("nan")),
                "total_return_pct": vbt_stats.get("Total Return [%]", float("nan")),
                "coverage_start": ltf_df.index.min(),
                "coverage_end": ltf_df.index.max(),
                "equity_curve": portfolio.value(),
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
    parser.add_argument("--shard", type=str, default=None, help="e.g. '0/8' = this is worker 0 of 8")
    parser.add_argument("--configs", type=str, default=None,
                         help="comma-separated subset of CONFIGS to run, e.g. 'ADX Gate' (for re-running just one after a fix)")
    parser.add_argument("--tag", type=str, default=None, help="extra suffix for the output cache filename")
    args = parser.parse_args()

    if args.configs:
        wanted = {c.strip() for c in args.configs.split(",")}
        global CONFIGS
        CONFIGS = {k: v for k, v in CONFIGS.items() if k in wanted}
        print(f"Restricted to configs: {list(CONFIGS.keys())}", flush=True)

    symbols = all_symbols()
    tag_suffix = f"_{args.tag}" if args.tag else ""
    if args.shard:
        idx_str, n_str = args.shard.split("/")
        idx, n = int(idx_str), int(n_str)
        symbols = symbols[idx::n]
        cache_path = os.path.join(REPORT_DIR, f"metafvg_ab_sweep_data_{_SLUG}{tag_suffix}_shard{idx}.pkl")
        print(f"Shard {idx}/{n}: {len(symbols)} symbols -> {cache_path}", flush=True)
    else:
        cache_path = os.path.join(REPORT_DIR, f"metafvg_ab_sweep_data_{_SLUG}{tag_suffix}.pkl")

    run_symbols(symbols, cache_path)


if __name__ == "__main__":
    main()
