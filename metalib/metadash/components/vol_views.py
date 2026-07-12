"""
Vol Forecast view: MetaHAR walk-forward OOS predictions vs realized.
Data source: notebooks/metahar_wf/predictions_{SYM}_dummies_own.csv + volstore.db
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BORDER = 'rgba(38,34,29,0.10)'
TEXT_PRIMARY = '#26221D'
TEXT_SECONDARY = '#7A756C'
ACCENT = '#BF6A3D'
ACCENT_SOFT = '#F3E4D8'
GREEN = '#3D7A54'
GREEN_SOFT = '#E7F0EA'
RED = '#B5473A'
RED_SOFT = '#F5E4E1'
BLUE = '#2D6EA8'
BLUE_SOFT = '#E0EAF5'

CARD = {
    'background': '#fff', 'border': f'1px solid {BORDER}',
    'borderRadius': '12px', 'padding': '18px 20px',
}

_HERE = os.path.dirname(__file__)
# _HERE = .../metalib/metadash/components; go up 2 to reach metalib package root
_METALIB = os.path.abspath(os.path.join(_HERE, '..', '..'))
WF_DIR = os.path.join(_METALIB, 'notebooks', 'metahar_wf')
M15_DIR = os.path.join(_METALIB, 'data', 'mix_m15')
NOTEBOOKS_DIR = os.path.join(_METALIB, 'notebooks')

SYMBOLS = ['EURUSD', 'XAUUSD', 'BTCUSD', 'GBPUSD', 'US500']
PERIODS = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365, 'All': None}
TRAIN_DAYS = 66

# M15 bars per year by symbol (for annualizing realized vol to %)
_BARS_PER_YEAR = {
    'EURUSD': 252 * 24 * 4,   # ~24192 — 24h FX market
    'XAUUSD': 252 * 24 * 4,
    'GBPUSD': 252 * 24 * 4,
    'US500':  252 * 7 * 4,    # ~7056  — extended-hours futures
    'BTCUSD': 365 * 24 * 4,   # ~35040 — crypto 24/7
}

# ─── IS R² computation ────────────────────────────────────────────────────────

def compute_is_r2(symbol):
    """Fit Lasso on the last TRAIN_DAYS of M15 data, return IS R² (zero-forecast benchmark).
    Returns None on missing data or error. Imports numba code from the notebook."""
    m15_path = os.path.join(M15_DIR, f'{symbol}_m15.csv')
    if not os.path.exists(m15_path):
        return None
    try:
        if NOTEBOOKS_DIR not in sys.path:
            sys.path.insert(0, NOTEBOOKS_DIR)
        from metahar_wf_backtest import fit_lasso_cv

        df = pd.read_csv(m15_path, index_col='time', parse_dates=True)
        cutoff = df.index[-1] - pd.Timedelta(days=TRAIN_DAYS + 14)
        df_tr = df[df.index >= cutoff]
        closes = df_tr['close']
        closes.index = pd.to_datetime(closes.index)

        ret = np.log(closes).diff().dropna()
        sq = ret ** 2
        down = ret.where(ret < 0, 0.0) ** 2
        up = ret.where(ret > 0, 0.0) ** 2

        feats = pd.concat({
            'long_rv': np.log(ret.rolling(256).var(ddof=0)),
            'short_rv': np.log(ret.rolling(16).var(ddof=0)),
            'ewm_s': ret.ewm(halflife=16).mean(),
            'ewm_l': ret.ewm(halflife=256).mean(),
            'sq_s': sq.ewm(halflife=16).mean(),
            'sq_l': sq.ewm(halflife=256).mean(),
            'dn_s': down.ewm(halflife=16).mean(),
            'dn_l': down.ewm(halflife=256).mean(),
            'up_s': up.ewm(halflife=16).mean(),
            'up_l': up.ewm(halflife=256).mean(),
        }, axis=1)
        feats['const'] = 1.0
        resampled = (
            feats.resample('4h', closed='right', label='right')
            .last()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        # restrict to last TRAIN_DAYS
        resampled = resampled[resampled.index >= resampled.index[-1] - pd.Timedelta(days=TRAIN_DAYS)]
        x = resampled['short_rv']
        y = (x.shift(-1) - x).dropna()
        X = resampled.loc[y.index]
        if len(X) < 30:
            return None
        predict, _, _ = fit_lasso_cv(X, y)
        y_is = predict(X)
        ss_tot = float((y.values ** 2).sum())
        if ss_tot == 0:
            return None
        return float(1 - ((y.values - y_is) ** 2).sum() / ss_tot)
    except Exception as e:
        print(f'vol_views: IS R² failed for {symbol}: {e}')
        return None


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_wf_predictions(symbol):
    path = os.path.join(WF_DIR, f'predictions_{symbol}_dummies_own.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.rename(columns={'timestamp': 'ts', 'y': 'realized', 'pred': 'predicted'})
    df = df.sort_values('ts').reset_index(drop=True)
    return df


def _load_summary():
    path = os.path.join(WF_DIR, 'summary_dummies_own.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_live_predictions(symbol):
    """Recent rows from volstore, aligned so realized aligns with the bar it describes."""
    try:
        # PYTHONPATH includes the repo root (parent of metalib package)
        repo_root = os.path.abspath(os.path.join(_METALIB, '..'))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import metalib.volstore as vs
        df = vs.get_history(symbol=symbol)
        if df.empty:
            return None
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.sort_values('ts').reset_index(drop=True)
        # shift: prediction at row t applies to bar t+1; realized_prev_diff at t = realized for bar t
        df['predicted'] = df['prediction'].shift(1)
        df['realized'] = df['realized_prev_diff']
        return df[['ts', 'predicted', 'realized', 'tag']].dropna(subset=['predicted', 'realized'])
    except Exception as e:
        print(f'vol_views: volstore load failed: {e}')
        return None


def _filter_period(df, period):
    if period == 'All' or PERIODS.get(period) is None:
        return df
    cutoff = df['ts'].max() - pd.Timedelta(days=PERIODS[period])
    return df[df['ts'] >= cutoff].reset_index(drop=True)


def _compute_metrics(df):
    """OOS metrics from a filtered predictions DataFrame."""
    if df is None or df.empty or 'realized' not in df or 'predicted' not in df:
        return {}
    y = df['realized'].dropna().values
    p = df['predicted'].dropna().values
    mask = ~(np.isnan(y) | np.isnan(p))
    y, p = y[mask], p[mask]
    if len(y) < 5:
        return {}
    ss_tot = float((y ** 2).sum())
    oos_r2 = float(1 - ((y - p) ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan
    direction = float((np.sign(p) == np.sign(y)).mean())
    # Spearman IC: manual to avoid BLAS
    ry = pd.Series(y).rank().values
    rp = pd.Series(p).rank().values
    da, db = ry - ry.mean(), rp - rp.mean()
    denom = float(np.sqrt((da ** 2).sum() * (db ** 2).sum()))
    ic = float((da * db).sum() / denom) if denom > 0 else np.nan
    q75 = float(np.quantile(np.abs(p), 0.75))
    conv_mask = np.abs(p) >= q75
    hit_conv = float((np.sign(p[conv_mask]) == np.sign(y[conv_mask])).mean()) if conv_mask.sum() > 0 else np.nan
    return {
        'oos_r2': oos_r2,
        'direction': direction,
        'ic': ic,
        'hit_conv': hit_conv,
        'n': int(len(y)),
    }


# ─── Realized vol computation ─────────────────────────────────────────────────

def _compute_realized_vol_series(symbol, extra_days=30):
    """Load M15 data → 4h-bar annualized realized vol (%) Series indexed by timestamp.
    Returns None if M15 CSV is missing."""
    path = os.path.join(M15_DIR, f'{symbol}_m15.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col='time', parse_dates=True)
    ret = np.log(df['close']).diff().dropna()
    # rolling(16) variance over 16 M15 bars = realized variance per M15 bar in the 4h window
    rv_m15 = ret.rolling(16).var(ddof=0)
    rv_4h = rv_m15.resample('4h', closed='right', label='right').last().dropna()
    bpy = _BARS_PER_YEAR.get(symbol, 24192)
    vol_pct = np.sqrt(rv_4h.clip(lower=0) * bpy) * 100
    vol_pct.index.name = 'ts'
    return vol_pct


def _attach_vol_levels(df, symbol):
    """Join realized vol % onto predictions df; derive predicted vol % level."""
    rv = _compute_realized_vol_series(symbol)
    if rv is None or df is None:
        return df
    rv_df = rv.rename('realized_vol_pct').reset_index()
    rv_df['ts'] = pd.to_datetime(rv_df['ts'])
    out = df.merge(rv_df, on='ts', how='left')
    # pred(t) predicts log-rv change for bar t → predicted level at t =
    # exp(log(rv(t-1)) + pred(t)) = rv(t-1) * exp(pred(t))
    # in vol % terms: realized_vol_pct(t-1) * sqrt(exp(pred(t)))
    rv_prev = out['realized_vol_pct'].shift(1)
    out['predicted_vol_pct'] = rv_prev * np.sqrt(np.exp(out['predicted'].clip(-5, 5)))
    return out


# ─── Forward forecast data ────────────────────────────────────────────────────

_N_FORWARD = 8   # 4h bars ahead = 32h
_HAR_PHI   = 0.82  # vol persistence for mean-reversion step

def _build_forecast_data(symbol, df_wf=None):
    """Compute forward vol fan + spot price fan from current M15 state.

    Returns a dict or None if M15 CSV is missing.
    Residual sigma comes from the WF backtest (log-rv-change units).
    Multi-step uncertainty scales as √h (iid innovation assumption).
    """
    m15_path = os.path.join(M15_DIR, f'{symbol}_m15.csv')
    if not os.path.exists(m15_path):
        return None

    df_m15 = pd.read_csv(m15_path, index_col='time', parse_dates=True)
    bpy = _BARS_PER_YEAR.get(symbol, 24192)

    ret = np.log(df_m15['close']).diff().dropna()
    rv_m15 = ret.rolling(16).var(ddof=0)
    rv_4h = rv_m15.resample('4h', closed='right', label='right').last().dropna()
    vol_4h = np.sqrt(rv_4h.clip(lower=0) * bpy) * 100

    current_rv = float(rv_4h.iloc[-1])
    current_log_rv = float(np.log(current_rv)) if current_rv > 0 else -12.0
    current_vol = float(vol_4h.iloc[-1])
    current_price = float(df_m15['close'].iloc[-1])
    last_ts = rv_4h.index[-1]

    # Long-run mean log-rv (last 66 days) — target for HAR mean-reversion
    lr_cutoff = rv_4h.index[-1] - pd.Timedelta(days=66)
    long_run_log_rv = float(
        np.log(rv_4h[rv_4h.index >= lr_cutoff].clip(lower=1e-12)).replace(
            [np.inf, -np.inf], np.nan
        ).dropna().mean()
    )

    # Current prediction: volstore first, then last WF row, then 0
    current_pred = 0.0
    try:
        repo_root = os.path.abspath(os.path.join(_METALIB, '..'))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import metalib.volstore as vs
        latest = vs.get_latest(symbol, max_age_minutes=8 * 60)
        if latest:
            current_pred = float(latest['prediction'])
    except Exception:
        pass
    if current_pred == 0.0 and df_wf is not None and not df_wf.empty:
        current_pred = float(df_wf['predicted'].iloc[-1])

    # 1-step residual sigma from WF backtest (log-rv-change space)
    if df_wf is not None and len(df_wf) > 10:
        sigma1 = float((df_wf['realized'] - df_wf['predicted']).dropna().std())
    else:
        sigma1 = 0.50

    # Forward trajectory with HAR mean-reversion
    future_ts, future_log_rv, future_vol = [], [], []
    log_rv_h = current_log_rv + float(np.clip(current_pred, -2.5, 2.5))
    for h in range(1, _N_FORWARD + 1):
        future_ts.append(last_ts + pd.Timedelta(hours=4 * h))
        future_log_rv.append(log_rv_h)
        future_vol.append(float(np.sqrt(np.exp(log_rv_h) * bpy) * 100))
        log_rv_h = long_run_log_rv + _HAR_PHI * (log_rv_h - long_run_log_rv)

    # CI bands in vol % space: vol_center * sqrt(exp(±z * σ₁ * √h))
    z_levels = {50: 0.6745, 75: 1.1503, 95: 1.9600}
    vol_bands = {}
    for level, z in z_levels.items():
        lo, hi = [], []
        for h_idx, log_rv_h in enumerate(future_log_rv, 1):
            half = z * sigma1 * np.sqrt(h_idx)
            lo.append(float(np.sqrt(np.exp(log_rv_h - half) * bpy) * 100))
            hi.append(float(np.sqrt(np.exp(log_rv_h + half) * bpy) * 100))
        vol_bands[level] = {'lo': lo, 'hi': hi}

    # Price CI bands: log-normal GBM, σ² scales with predicted vol + h
    # T_h = h * 4h / (trading hours per year), σ = vol%/100 annualised
    trading_hours_per_year = bpy / 4  # bpy M15 bars × 15min/bar / 60 → hours → ÷4 for 4h
    price_bands = {}
    for level, z in z_levels.items():
        lo, hi = [], []
        for h_idx, vol_h in enumerate(future_vol, 1):
            T_h = (h_idx * 4) / trading_hours_per_year
            half = z * (vol_h / 100) * np.sqrt(T_h)
            lo.append(float(current_price * np.exp(-half)))
            hi.append(float(current_price * np.exp(+half)))
        price_bands[level] = {'lo': lo, 'hi': hi}

    # Historical context: last 24 × 4h bars for vol, last 48 × H1 for price
    hist_vol = vol_4h.tail(24)
    hist_h1 = df_m15.resample('1h', closed='right', label='right').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    ).dropna().tail(48)

    return {
        'symbol': symbol,
        'last_ts': last_ts,
        'current_price': current_price,
        'current_vol': current_vol,
        'hist_vol': hist_vol,
        'hist_h1': hist_h1,
        'future_ts': future_ts,
        'future_vol': future_vol,
        'vol_bands': vol_bands,
        'price_bands': price_bands,
    }


_CI_FILL = {95: 'rgba(191,106,61,0.10)', 75: 'rgba(191,106,61,0.22)', 50: 'rgba(191,106,61,0.40)'}
_CI_FILL_P = {95: 'rgba(45,110,168,0.10)', 75: 'rgba(45,110,168,0.20)', 50: 'rgba(45,110,168,0.34)'}


def _vol_fan_chart(fc):
    """Vol forecast fan: last 24 × 4h realized + forward fan."""
    fig = go.Figure()
    sym = fc['symbol']
    hist = fc['hist_vol']
    future_ts = fc['future_ts']
    future_vol = fc['future_vol']
    last_ts = fc['last_ts']

    # Fan bands, widest first for correct z-layering
    for level in [95, 75, 50]:
        lo, hi = fc['vol_bands'][level]['lo'], fc['vol_bands'][level]['hi']
        fig.add_trace(go.Scatter(
            x=future_ts + future_ts[::-1], y=hi + lo[::-1],
            fill='toself', fillcolor=_CI_FILL[level],
            line=dict(width=0), name=f'{level}%', legendgroup=f'ci{level}',
            hoverinfo='skip',
        ))

    # Historical realized vol (filled area)
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, name='Realized',
        fill='tozeroy', fillcolor='rgba(38,34,29,0.05)',
        line=dict(color='rgba(38,34,29,0.75)', width=2),
        hovertemplate='%{x|%b %d %H:%M}  <b>%{y:.1f}%</b><extra>Realized</extra>',
    ))

    # Connector from last realized → first forecast
    fig.add_trace(go.Scatter(
        x=[hist.index[-1], future_ts[0]], y=[hist.iloc[-1], future_vol[0]],
        mode='lines', line=dict(color=ACCENT, width=1.5, dash='dot'),
        showlegend=False, hoverinfo='skip',
    ))

    # Central forecast line
    fig.add_trace(go.Scatter(
        x=future_ts, y=future_vol, name='Forecast',
        mode='lines+markers',
        line=dict(color=ACCENT, width=2),
        marker=dict(size=5, color=ACCENT),
        hovertemplate='%{x|%b %d %H:%M}  <b>%{y:.1f}%</b><extra>Forecast</extra>',
    ))

    fig.add_vline(x=last_ts, line=dict(color='rgba(38,34,29,0.22)', width=1, dash='dot'))

    fig.update_layout(
        height=310, margin=dict(l=4, r=4, t=4, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(size=11, color=TEXT_PRIMARY),
        legend=dict(orientation='h', y=1.04, x=0, font=dict(size=10),
                    bgcolor='rgba(0,0,0,0)', traceorder='normal'),
        hovermode='x unified',
        yaxis=dict(
            ticksuffix='%', showgrid=True, gridcolor='rgba(38,34,29,0.07)',
            zeroline=False, rangemode='tozero',
        ),
        xaxis=dict(showgrid=True, gridcolor='rgba(38,34,29,0.07)', zeroline=False),
    )
    return fig


def _price_fan_chart(fc):
    """Spot price fan: last 48 × H1 candles + forward price ranges."""
    fig = go.Figure()
    sym = fc['symbol']
    hist = fc['hist_h1']
    future_ts = fc['future_ts']
    current_price = fc['current_price']
    last_ts = fc['last_ts']

    # Fan bands, widest first
    for level in [95, 75, 50]:
        lo, hi = fc['price_bands'][level]['lo'], fc['price_bands'][level]['hi']
        fig.add_trace(go.Scatter(
            x=future_ts + future_ts[::-1], y=hi + lo[::-1],
            fill='toself', fillcolor=_CI_FILL_P[level],
            line=dict(width=0), name=f'{level}%', legendgroup=f'ci{level}',
            hoverinfo='skip',
        ))

    # Price band edge lines (thin, to make band boundaries visible)
    for level in [50, 75, 95]:
        lo, hi = fc['price_bands'][level]['lo'], fc['price_bands'][level]['hi']
        alpha = {50: '0.7', 75: '0.5', 95: '0.35'}[level]
        clr = f'rgba(45,110,168,{alpha})'
        fig.add_trace(go.Scatter(
            x=future_ts, y=hi, mode='lines',
            line=dict(color=clr, width=1, dash='dot'),
            showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=future_ts, y=lo, mode='lines',
            line=dict(color=clr, width=1, dash='dot'),
            showlegend=False, hoverinfo='skip',
        ))

    # Historical H1 candlesticks
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['open'], high=hist['high'], low=hist['low'], close=hist['close'],
        name='Price',
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor='rgba(61,122,84,0.55)',
        decreasing_fillcolor='rgba(181,71,58,0.55)',
        line=dict(width=1),
        whiskerwidth=0,
    ))

    # Flat reference at current price extending into forecast window
    fig.add_trace(go.Scatter(
        x=[hist.index[-1]] + future_ts,
        y=[current_price] * (1 + len(future_ts)),
        mode='lines',
        line=dict(color='rgba(38,34,29,0.40)', width=1, dash='dot'),
        showlegend=False,
        hovertemplate='Current: <b>%{y}</b><extra></extra>',
    ))

    fig.add_vline(x=last_ts, line=dict(color='rgba(38,34,29,0.22)', width=1, dash='dot'))

    fig.update_layout(
        height=310, margin=dict(l=4, r=4, t=4, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(size=11, color=TEXT_PRIMARY),
        legend=dict(orientation='h', y=1.04, x=0, font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
        yaxis=dict(showgrid=True, gridcolor='rgba(38,34,29,0.07)', zeroline=False,
                   tickformat='.4g'),
        xaxis=dict(showgrid=True, gridcolor='rgba(38,34,29,0.07)', zeroline=False,
                   rangeslider=dict(visible=False)),
    )
    return fig


# ─── Chart builder ────────────────────────────────────────────────────────────

def _main_chart(df, symbol='EURUSD'):
    """Two-panel chart: realized + predicted vol % (top), direction accuracy (bottom)."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    ts = df['ts']
    real_vol = df.get('realized_vol_pct')
    pred_vol = df.get('predicted_vol_pct')
    # direction accuracy uses the log-change columns (model target space)
    realized_chg = df['realized']
    predicted_chg = df['predicted']

    has_vol = real_vol is not None and real_vol.notna().any()

    if has_vol:
        # Filled area under realized vol
        fig.add_trace(go.Scatter(
            x=ts, y=real_vol,
            name='Realized vol',
            line=dict(color='rgba(38,34,29,0.70)', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(38,34,29,0.06)',
            hovertemplate='%{x|%b %d %H:%M}  Realized: <b>%{y:.2f}%</b><extra></extra>',
        ), row=1, col=1)

        # Predicted vol — dashed orange line, no fill
        fig.add_trace(go.Scatter(
            x=ts, y=pred_vol,
            name='Predicted vol',
            line=dict(color=ACCENT, width=1.8, dash='dot'),
            hovertemplate='%{x|%b %d %H:%M}  Predicted: <b>%{y:.2f}%</b><extra></extra>',
        ), row=1, col=1)

        fig.update_yaxes(
            title_text='Annualized vol (%)',
            title_font=dict(size=10, color=TEXT_SECONDARY),
            showgrid=True, gridcolor='rgba(38,34,29,0.07)',
            zeroline=False, tickfont=dict(size=10),
            ticksuffix='%', rangemode='tozero',
            row=1, col=1,
        )
    else:
        # Fallback: raw log-change series
        fig.add_trace(go.Scatter(x=ts, y=realized_chg, name='Realized Δlog-rv',
                                  line=dict(color='rgba(38,34,29,0.65)', width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts, y=predicted_chg, name='Predicted Δlog-rv',
                                  line=dict(color=ACCENT, width=1.5)), row=1, col=1)
        fig.add_hline(y=0, line=dict(color='rgba(38,34,29,0.2)', width=1, dash='dot'), row=1, col=1)

    # Direction accuracy strip
    correct = pd.Series((np.sign(predicted_chg) == np.sign(realized_chg)).astype(float))
    colors = [GREEN if c else RED for c in correct]
    fig.add_trace(go.Bar(
        x=ts, y=[1.0] * len(ts),
        marker_color=colors, marker_line_width=0,
        showlegend=False, name='Direction',
        hovertemplate='%{x|%b %d %H:%M}: %{customdata}<extra></extra>',
        customdata=['✓' if c else '✗' for c in correct],
    ), row=2, col=1)

    roll = correct.rolling(30, min_periods=5).mean()
    fig.add_trace(go.Scatter(
        x=ts, y=roll, name='30-bar accuracy',
        line=dict(color=TEXT_PRIMARY, width=1.5),
        hovertemplate='Rolling accuracy: %{y:.0%}<extra></extra>',
    ), row=2, col=1)
    fig.add_hline(y=0.5, line=dict(color='rgba(38,34,29,0.3)', width=1, dash='dot'), row=2, col=1)

    fig.update_layout(
        margin=dict(l=4, r=8, t=8, b=0),
        height=500,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='inherit', size=11, color=TEXT_PRIMARY),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0,
            font=dict(size=11), bgcolor='rgba(0,0,0,0)',
        ),
        hovermode='x unified',
        bargap=0,
    )
    fig.update_xaxes(
        showgrid=True, gridcolor='rgba(38,34,29,0.06)',
        zeroline=False, tickfont=dict(size=10),
    )
    fig.update_yaxes(
        showgrid=False, tickformat='.0%',
        tickfont=dict(size=10), range=[0, 1.12],
        row=2, col=1,
    )

    # Remove weekend gaps (Saturday–Monday) for non-crypto symbols
    if symbol != 'BTCUSD':
        fig.update_xaxes(rangebreaks=[dict(bounds=['sat', 'mon'])])

    return fig


# ─── Layout ───────────────────────────────────────────────────────────────────

def _kpi(label, value, sub=None, color=None):
    return html.Div([
        html.Div(label, style={'fontSize': '11.5px', 'color': TEXT_SECONDARY, 'marginBottom': '8px'}),
        html.Div(value, style={
            'fontFamily': "'Source Serif 4', serif",
            'fontSize': '22px', 'fontWeight': '600', 'letterSpacing': '-0.01em',
            'color': color or TEXT_PRIMARY,
        }),
        html.Div(sub, style={'fontSize': '11px', 'color': TEXT_SECONDARY, 'marginTop': '5px'})
        if sub else None,
    ], style=CARD)


def _period_btn(period, selected):
    active = period == selected
    return html.Button(
        period,
        id={'type': 'vol-period-btn', 'index': period},
        n_clicks=0,
        style={
            'padding': '5px 12px', 'borderRadius': '6px', 'cursor': 'pointer',
            'fontSize': '12px', 'fontWeight': '600', 'border': 'none',
            'background': ACCENT if active else 'transparent',
            'color': '#fff' if active else TEXT_SECONDARY,
            'fontFamily': 'inherit',
        }
    )


def render_vol_forecast(symbol='EURUSD', period='3M', is_r2_cache=None):
    """Main view renderer for the Vol Forecast tab."""
    df_wf = _load_wf_predictions(symbol)
    summary = _load_summary()

    # Period filter, then attach realized vol % levels
    df = _filter_period(df_wf, period) if df_wf is not None else None
    df = _attach_vol_levels(df, symbol)

    # Metrics (computed on log-change columns, unchanged)
    m = _compute_metrics(df)

    # IS R²
    is_r2 = None
    if is_r2_cache and isinstance(is_r2_cache, dict):
        is_r2 = is_r2_cache.get(symbol)

    # Summary row for full-period WF metrics
    sym_summary = summary[summary['symbol'] == symbol].iloc[0] if (summary is not None and not summary.empty and symbol in summary['symbol'].values) else None

    # KPI cards
    oos_r2_val = m.get('oos_r2')
    dir_val = m.get('direction')
    ic_val = m.get('ic')
    hit_conv_val = m.get('hit_conv')
    n_bars = m.get('n', 0)

    def fmt_r2(v):
        return f'{v:.3f}' if v is not None and not np.isnan(v) else '—'

    def fmt_pct(v):
        return f'{v:.1%}' if v is not None and not np.isnan(v) else '—'

    def r2_color(v):
        if v is None or np.isnan(v):
            return TEXT_SECONDARY
        return GREEN if v > 0.3 else (ACCENT if v > 0 else RED)

    kpis = html.Div([
        _kpi(
            'IS R²',
            fmt_r2(is_r2),
            'Last 66-day window',
            r2_color(is_r2) if is_r2 is not None else TEXT_SECONDARY,
        ),
        _kpi(
            'OOS R² (period)',
            fmt_r2(oos_r2_val),
            f'{n_bars} 4h bars' if n_bars else 'Walk-forward',
            r2_color(oos_r2_val) if oos_r2_val is not None else TEXT_SECONDARY,
        ),
        _kpi(
            'Direction Accuracy',
            fmt_pct(dir_val),
            f'vs 50% random' if dir_val is not None else None,
            GREEN if (dir_val or 0) > 0.55 else (ACCENT if (dir_val or 0) > 0.5 else RED),
        ),
        _kpi(
            'Spearman IC',
            fmt_r2(ic_val),
            f'Conviction hit: {fmt_pct(hit_conv_val)}',
            GREEN if (ic_val or 0) > 0.3 else (ACCENT if (ic_val or 0) > 0 else RED),
        ),
    ], style={
        'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)',
        'gap': '12px', 'marginBottom': '16px',
    })

    # Chart
    chart = html.Div(
        dcc.Graph(
            figure=_main_chart(df, symbol) if df is not None and not df.empty else go.Figure(),
            config={'displayModeBar': False},
            style={'height': '500px'},
        ),
        style={**CARD, 'padding': '12px'},
    )

    # Symbol selector
    sym_dropdown = html.Div([
        dcc.Dropdown(
            id='vol-symbol-dropdown',
            options=[{'label': s, 'value': s} for s in SYMBOLS],
            value=symbol,
            clearable=False,
            style={
                'width': '140px', 'fontSize': '13px',
                'fontFamily': 'inherit', 'border': f'1px solid {BORDER}',
                'borderRadius': '8px',
            },
        ),
    ])

    period_row = html.Div(
        [_period_btn(p, period) for p in PERIODS],
        style={
            'display': 'flex', 'gap': '4px', 'background': '#F7F5F2',
            'padding': '4px', 'borderRadius': '8px',
        }
    )

    controls = html.Div(
        [sym_dropdown, period_row],
        style={'display': 'flex', 'alignItems': 'center', 'gap': '16px', 'marginBottom': '20px'}
    )

    title = html.Div([
        html.Div('Volatility Forecast', style={
            'fontFamily': "'Source Serif 4', serif",
            'fontSize': '22px', 'fontWeight': '600', 'letterSpacing': '-0.01em',
        }),
        html.Div('MetaHAR · next-4h log realized variance change', style={
            'fontSize': '12.5px', 'color': TEXT_SECONDARY, 'marginTop': '4px',
        }),
    ], style={'marginBottom': '20px'})

    # Forward forecast section
    fc_data = _build_forecast_data(symbol, df_wf)
    if fc_data is not None:
        section_label_style = {
            'fontSize': '11px', 'fontWeight': '600', 'letterSpacing': '0.06em',
            'textTransform': 'uppercase', 'color': TEXT_SECONDARY, 'marginBottom': '10px',
        }
        forecast_section = html.Div([
            html.Div('Forward Forecast · next 32h', style=section_label_style),
            html.Div([
                html.Div([
                    html.Div('Volatility fan · annualised %', style={
                        'fontSize': '11px', 'color': TEXT_SECONDARY, 'marginBottom': '6px',
                    }),
                    dcc.Graph(
                        figure=_vol_fan_chart(fc_data),
                        config={'displayModeBar': False},
                        style={'height': '310px'},
                    ),
                ], style={**CARD, 'padding': '14px 14px 10px', 'flex': '1', 'minWidth': '0'}),
                html.Div([
                    html.Div('Spot price range · GBM log-normal', style={
                        'fontSize': '11px', 'color': TEXT_SECONDARY, 'marginBottom': '6px',
                    }),
                    dcc.Graph(
                        figure=_price_fan_chart(fc_data),
                        config={'displayModeBar': False},
                        style={'height': '310px'},
                    ),
                ], style={**CARD, 'padding': '14px 14px 10px', 'flex': '1', 'minWidth': '0'}),
            ], style={'display': 'flex', 'gap': '14px'}),
        ], style={'marginTop': '16px'})
    else:
        forecast_section = html.Div()

    return html.Div(
        [title, controls, kpis, chart, forecast_section],
        style={'padding': '28px 32px', 'maxWidth': '1400px'},
    )
