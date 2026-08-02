"""
Per-symbol standalone Sharpe under the Combined Mix production parameters
(window=25, threshold=0.4, atr_sensitivity=2.0), for selecting the top-10
production universe. Deliberately NOT reusing the baseline-param ranking
(Table 2 in the research report, window=20/threshold=0.5/atr=4.0) --
parameter changes can shift which symbols perform best, and this ranking
feeds a real production deployment, so it should reflect the actual
parameters being deployed, not the parameters used for exploratory backtesting.

Usage:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/rank_symbols_combined_mix.py
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
OUT_PKL = os.path.join(RESEARCH_DIR, "data", "combined_mix_per_symbol.pkl")

SYMBOLS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
           "US500", "GER40", "JP225", "HK50", "USDSGD", "EURNOK", "USDZAR"]
WINDOW, THRESHOLD, ATR_SENSITIVITY = 25, 0.4, 2.0


def main():
    with open(CACHE_PKL, "rb") as f:
        cache = pickle.load(f)
    start, end = cache["start"], cache["end"]

    rows = []
    for symbol in SYMBOLS:
        ltf_df, htf_df = cache[symbol]["ltf_df"], cache[symbol]["htf_df"]
        params = VariantParams(symbol=symbol, start=start, end=end, limit_number_position=2,
                                trend_filter="spearman", regression_window=WINDOW,
                                regression_r2_threshold=THRESHOLD, exit_style="fixed",
                                invert_direction=True, atr_sensitivity=ATR_SENSITIVITY)
        trades, _, _ = simulate_fvg_trades_variant(ltf_df, htf_df, params)
        trades_df = trades_to_dataframe(trades)
        closed = trades_df[trades_df.status == "closed"]
        wr = (closed.pnl > 0).mean() * 100 if len(closed) else float("nan")
        if len(closed) == 0:
            rows.append({"symbol": symbol, "closed_trades": 0, "win_rate_pct": float("nan"),
                         "sharpe": float("nan"), "total_return_pct": float("nan")})
            print(f"{symbol}: 0 closed trades", flush=True)
            continue
        portfolio = build_vbt_portfolio(ltf_df, trades, params.limit_number_position)
        value = portfolio.value()
        daily_rets = value.resample("D").last().ffill().dropna().pct_change().dropna()
        equity = (1 + daily_rets).cumprod()
        m = quantstats_metrics(equity)
        sharpe = m.get("qs_sharpe", float("nan"))
        total_return_pct = (equity.iloc[-1] - 1) * 100 if len(equity) else float("nan")
        rows.append({"symbol": symbol, "closed_trades": len(closed), "win_rate_pct": wr,
                     "sharpe": sharpe, "total_return_pct": total_return_pct})
        print(f"{symbol}: closed={len(closed)} wr={wr:.1f}% sharpe={sharpe:.4f} "
              f"total_return={total_return_pct:.4f}%", flush=True)

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(df, f)
    print(f"\nwritten: {OUT_PKL}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
