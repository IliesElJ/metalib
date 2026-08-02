"""One-off: Spearman mean-reversion across a fundamentally-diversified 14-
instrument universe (7 FX Majors + 4 Indices across distinct central-bank
regimes + 3 FX pairs introducing genuinely new currency exposure), both fixed
and risk-normalized sizing, one symbol per process. Not a permanent module."""
import warnings; warnings.filterwarnings("ignore")
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime
import pandas as pd, numpy as np
from metalib.metafvg_backtest import fetch_ltf_htf_candles, connect_mt5, LTF_TIMEFRAME_OPTIONS, HTF_RESAMPLE_OPTIONS, build_vbt_portfolio, trades_to_dataframe
from metafvg_variants import VariantParams, simulate_fvg_trades_variant, build_vbt_portfolio_risk_sized

symbol = sys.argv[1]

connect_mt5()
end = datetime.utcnow()
start = end - pd.Timedelta(days=365 * 5)

ltf_df, htf_df = fetch_ltf_htf_candles(symbol, start, end, ltf_timeframe=LTF_TIMEFRAME_OPTIONS["M15"], htf_resample_rule=HTF_RESAMPLE_OPTIONS["4h"])

params = VariantParams(symbol=symbol, start=start, end=end, limit_number_position=2,
                        trend_filter="spearman", regression_window=20, regression_r2_threshold=0.5,
                        exit_style="fixed", invert_direction=True)
trades, _, _ = simulate_fvg_trades_variant(ltf_df, htf_df, params)
trades_df = trades_to_dataframe(trades)
closed = trades_df[trades_df.status == "closed"]
wr = (closed.pnl > 0).mean() * 100 if len(closed) else float("nan")

results = {}
for label, builder in [("fixed", build_vbt_portfolio), ("risk_sized", build_vbt_portfolio_risk_sized)]:
    portfolio = builder(ltf_df, trades, params.limit_number_position)
    stats = portfolio.stats()
    sharpe = stats.get("Sharpe Ratio", float("nan"))
    maxdd = stats.get("Max Drawdown [%]", float("nan"))
    value = portfolio.value()
    daily_rets = value.resample("D").last().ffill().dropna().pct_change().dropna()
    results[label] = dict(closed=len(closed), wr=wr, sharpe=sharpe, maxdd=maxdd, daily_rets=daily_rets)
    print(f"{symbol} [{label}]: closed={len(closed)} win_rate={wr:.1f}% sharpe={sharpe:.3f} maxdd={maxdd:.3f}%", flush=True)

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", f"meanrev_diversified_{symbol}.pkl")
with open(out_path, "wb") as f:
    pickle.dump(results, f)
print(f"SAVED {out_path}", flush=True)
