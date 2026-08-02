"""One-off: mean-reversion probe (invert_direction=True) across the regression
family, one symbol per process. Not a permanent module -- safe to delete."""
import warnings; warnings.filterwarnings("ignore")
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime
import pandas as pd, numpy as np
from metalib.metafvg_backtest import fetch_ltf_htf_candles, connect_mt5, LTF_TIMEFRAME_OPTIONS, HTF_RESAMPLE_OPTIONS, build_vbt_portfolio, trades_to_dataframe
from metafvg_lasso_gate import compute_lasso_gate_series
from metafvg_variants import VariantParams, simulate_fvg_trades_variant

symbol = sys.argv[1]

connect_mt5()
end = datetime.utcnow()
start = end - pd.Timedelta(days=365 * 5)

CONFIGS = {
    "Regression Gate (mean-rev, r2>=0.5)": dict(trend_filter="regression", regression_window=20, regression_r2_threshold=0.5, exit_style="fixed", invert_direction=True),
    "Regression Gate (mean-rev, r2>=0.3)": dict(trend_filter="regression", regression_window=20, regression_r2_threshold=0.3, exit_style="fixed", invert_direction=True),
    "Theil-Sen Gate (mean-rev, r2>=0.5)": dict(trend_filter="theilsen", regression_window=20, regression_r2_threshold=0.5, exit_style="fixed", invert_direction=True),
    "Spearman Gate (mean-rev, q=0.5)": dict(trend_filter="spearman", regression_window=20, regression_r2_threshold=0.5, exit_style="fixed", invert_direction=True),
    "Lasso Trend Gate (mean-rev)": dict(trend_filter="lasso", exit_style="fixed", invert_direction=True),
}

ltf_df, htf_df = fetch_ltf_htf_candles(symbol, start, end, ltf_timeframe=LTF_TIMEFRAME_OPTIONS["M15"], htf_resample_rule=HTF_RESAMPLE_OPTIONS["4h"])

results = {}
for name, kwargs in CONFIGS.items():
    kwargs = dict(kwargs)
    trend_filter = kwargs.get("trend_filter")
    extra = {}
    if trend_filter == "lasso":
        pred, thr = compute_lasso_gate_series(htf_df, threshold_quantile=0.7)
        extra = dict(lasso_pred=pred, lasso_threshold=thr)
    params = VariantParams(symbol=symbol, start=start, end=end, limit_number_position=2, **kwargs)
    trades, _, _ = simulate_fvg_trades_variant(ltf_df, htf_df, params, **extra)
    trades_df = trades_to_dataframe(trades)
    closed = trades_df[trades_df.status == "closed"]
    wr = (closed.pnl > 0).mean() * 100 if len(closed) else float("nan")
    portfolio = build_vbt_portfolio(ltf_df, trades, params.limit_number_position)
    stats = portfolio.stats()
    sharpe = stats.get("Sharpe Ratio", float("nan"))
    maxdd = stats.get("Max Drawdown [%]", float("nan"))
    results[name] = dict(closed=len(closed), wr=wr, sharpe=sharpe, maxdd=maxdd)
    print(f"{symbol} {name}: closed={len(closed)} win_rate={wr:.1f}% sharpe={sharpe:.3f} maxdd={maxdd:.3f}%", flush=True)

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", f"meanrev_probe_fixed_{symbol}.pkl")
with open(out_path, "wb") as f:
    pickle.dump(results, f)
print(f"SAVED {out_path}", flush=True)
