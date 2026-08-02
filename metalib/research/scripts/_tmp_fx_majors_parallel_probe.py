"""
Reusable template for ad hoc FX Majors probes, parallelized one OS process per
symbol (7 symbols = 7-way parallelism, same pattern as the 20-shard 47-symbol
campaigns, just at single-symbol granularity since there are only 7 pairs).
Edit CONFIGS below for whatever's being compared, then run via the shell loop
at the bottom of this docstring -- each process writes its own
metalib/research/data/fx_majors_probe_<symbol>.pkl, merge afterward.

    for s in AUDUSD EURUSD GBPUSD NZDUSD USDCAD USDCHF USDJPY; do
      PYTHONPATH=. "<adonys python>" metalib/research/scripts/_tmp_fx_majors_parallel_probe.py $s \
        > metalib/research/logs/fx_majors_probe_$s.log 2>&1 &
    done
    wait

Not a permanent module -- ad hoc probe scaffold, safe to delete/overwrite
between uses.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime
import pandas as pd, numpy as np
from metalib.metafvg_backtest import fetch_ltf_htf_candles, connect_mt5, LTF_TIMEFRAME_OPTIONS, HTF_RESAMPLE_OPTIONS, build_vbt_portfolio, trades_to_dataframe
from metafvg_lasso_gate import compute_lasso_gate_series
from metafvg_variants import VariantParams, simulate_fvg_trades_variant, build_vbt_portfolio_risk_sized

symbol = sys.argv[1]

connect_mt5()
end = datetime.utcnow()
start = end - pd.Timedelta(days=365 * 5)

# ---- Edit this per probe ----
CONFIGS = {
    "Regression Gate": dict(trend_filter="regression", regression_window=20, regression_r2_threshold=0.5, exit_style="fixed"),
}
# ------------------------------

ltf_df, htf_df = fetch_ltf_htf_candles(symbol, start, end, ltf_timeframe=LTF_TIMEFRAME_OPTIONS["M15"], htf_resample_rule=HTF_RESAMPLE_OPTIONS["4h"])

results = {}
for name, kwargs in CONFIGS.items():
    kwargs = dict(kwargs)
    risk_sized = kwargs.pop("risk_sized", False)
    trend_filter = kwargs.get("trend_filter")
    extra = {}
    if trend_filter == "lasso":
        pred, thr = compute_lasso_gate_series(htf_df, threshold_quantile=kwargs.pop("threshold_quantile", 0.7))
        extra = dict(lasso_pred=pred, lasso_threshold=thr)
    params = VariantParams(symbol=symbol, start=start, end=end, limit_number_position=2, **kwargs)
    trades, _, _ = simulate_fvg_trades_variant(ltf_df, htf_df, params, **extra)
    trades_df = trades_to_dataframe(trades)
    closed = trades_df[trades_df.status == "closed"]
    wr = (closed.pnl > 0).mean() * 100 if len(closed) else float("nan")
    portfolio = build_vbt_portfolio_risk_sized(ltf_df, trades, params.limit_number_position) if risk_sized else build_vbt_portfolio(ltf_df, trades, params.limit_number_position)
    stats = portfolio.stats()
    sharpe = stats.get("Sharpe Ratio", float("nan"))
    maxdd = stats.get("Max Drawdown [%]", float("nan"))
    value = portfolio.value()
    daily_rets = value.resample("D").last().ffill().dropna().pct_change().dropna()
    results[name] = dict(closed=len(closed), wr=wr, sharpe=sharpe, maxdd=maxdd, daily_rets=daily_rets)
    print(f"{symbol} {name}: closed={len(closed)} win_rate={wr:.1f}% sharpe={sharpe:.3f} maxdd={maxdd:.3f}%", flush=True)

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", f"fx_majors_probe_{symbol}.pkl")
with open(out_path, "wb") as f:
    pickle.dump(results, f)
print(f"SAVED {out_path}", flush=True)
