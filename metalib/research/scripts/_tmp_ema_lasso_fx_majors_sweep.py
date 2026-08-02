"""One-off: EMA-augmented Lasso gate across FX Majors, vs. Regression Gate.
Not a permanent module -- ad hoc probe, safe to delete after use."""
import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime
import pandas as pd, numpy as np
import quantstats as qs
from metalib.metafvg_backtest import fetch_ltf_htf_candles, connect_mt5, LTF_TIMEFRAME_OPTIONS, HTF_RESAMPLE_OPTIONS, build_vbt_portfolio, trades_to_dataframe
from metafvg_lasso_gate import compute_lasso_gate_series
from metafvg_variants import VariantParams, simulate_fvg_trades_variant

connect_mt5()
end = datetime.utcnow()
start = end - pd.Timedelta(days=365*5)

FX_MAJORS = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
quantiles = {'loose (q=0.7)': 0.7, 'tight (q=0.9)': 0.9}

results = {}
per_symbol_daily = {name: {} for name in quantiles}

def daily_returns(portfolio):
    value = portfolio.value()
    daily = value.resample('D').last().ffill().dropna()
    return daily.pct_change().dropna()

for symbol in FX_MAJORS:
    ltf_df, htf_df = fetch_ltf_htf_candles(symbol, start, end, ltf_timeframe=LTF_TIMEFRAME_OPTIONS['M15'], htf_resample_rule=HTF_RESAMPLE_OPTIONS['4h'])
    print(f'--- {symbol} ---', flush=True)
    for name, q in quantiles.items():
        pred, thr = compute_lasso_gate_series(htf_df, threshold_quantile=q)
        params = VariantParams(symbol=symbol, start=start, end=end, limit_number_position=2, trend_filter='lasso', exit_style='fixed')
        trades, _, _ = simulate_fvg_trades_variant(ltf_df, htf_df, params, lasso_pred=pred, lasso_threshold=thr)
        trades_df = trades_to_dataframe(trades)
        closed = trades_df[trades_df.status=='closed']
        wr = (closed.pnl>0).mean()*100 if len(closed) else float('nan')
        portfolio = build_vbt_portfolio(ltf_df, trades, params.limit_number_position)
        sharpe = portfolio.stats().get('Sharpe Ratio', float('nan'))
        maxdd = portfolio.stats().get('Max Drawdown [%]', float('nan'))
        results[(symbol, name)] = dict(closed=len(closed), wr=wr, sharpe=sharpe, maxdd=maxdd)
        rets = daily_returns(portfolio)
        per_symbol_daily[name][symbol] = rets
        print(f'  {name}: closed={len(closed)} win_rate={wr:.1f}% sharpe={sharpe:.3f} maxdd={maxdd:.3f}%', flush=True)

print()
print('=== PORTFOLIO BLEND (equal-weighted, 7 FX Majors) ===', flush=True)
for name in quantiles:
    rets_df = pd.DataFrame(per_symbol_daily[name])
    blended = rets_df.mean(axis=1, skipna=True).dropna()
    blended_sharpe = float(qs.stats.sharpe(blended)) if len(blended) >= 10 and blended.std() > 0 else float('nan')
    avg_sharpe = np.mean([results[(s, name)]['sharpe'] for s in FX_MAJORS])
    print(f'{name}: avg_individual_sharpe={avg_sharpe:.3f} blended_portfolio_sharpe={blended_sharpe:.3f}', flush=True)

import pickle
with open('metalib/research/data/fx_majors_ema_lasso_vadj_sweep.pkl', 'wb') as f:
    pickle.dump({'results': results, 'per_symbol_daily': per_symbol_daily}, f)
print('SAVED', flush=True)
