"""
PM-friendly views: Overview, Trades, System.
Clean, minimal layouts targeting a non-quant audience.
"""

import base64
import numpy as np
import pandas as pd
from datetime import date, timedelta
from dash import html, dcc
import plotly.graph_objects as go

# ─── Design tokens ────────────────────────────────────────────────────────────
BORDER = 'rgba(38,34,29,0.10)'
TEXT_PRIMARY = '#26221D'
TEXT_SECONDARY = '#7A756C'
ACCENT = '#BF6A3D'
ACCENT_SOFT = '#F3E4D8'
GREEN = '#3D7A54'
GREEN_SOFT = '#E7F0EA'
RED = '#B5473A'
RED_SOFT = '#F5E4E1'

CARD = {'background': '#fff', 'border': f'1px solid {BORDER}', 'borderRadius': '12px', 'padding': '18px 20px'}

STRATEGY_NAMES = {
    'metaga': 'MetaGA', 'metamlp': 'Multi-Horizon MLP', 'metaob': 'Order Blocks',
    'metafvg': 'Fair Value Gaps', 'mtou': 'Mean-Reversion OU',
    'metago': 'MetaGO', 'metane': 'MetaNE', 'metamtou': 'Mean-Reversion OU',
    'metafvg_v2': 'FVG Mean-Reversion (v2)',
}
STRATEGY_ENGINE_KEYS = {'metaga', 'metamlp', 'metaob', 'metafvg', 'metamtou', 'metago', 'metane', 'metane', 'metafvg_v2'}
TIME_RANGE_DAYS = {'1W': 7, '1M': 30, '3M': 90, 'YTD': None, 'All': None}

STRATEGY_COLORS = {
    'metafvg': '#BF6A3D', 'metane': '#3D7A54', 'metaga': '#5B7FBA',
    'metamlp': '#7A5BA8', 'metago': '#B98A2E', 'metamtou': '#3D7A8A', 'metaob': '#B5473A',
    'metafvg_v2': '#1F7A6B',
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def serif(text, size=24, extra=None):
    style = {'fontFamily': "'Source Serif 4', serif", 'fontSize': f'{size}px',
              'fontWeight': '600', 'letterSpacing': '-0.01em'}
    if extra:
        style.update(extra)
    return html.Div(text, style=style)


def fmt(n, signed=False):
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return '—'
    sign = '+' if signed and n > 0 else ('-' if n < 0 else '')
    return f"{sign}${abs(n):,.2f}"


def pnl_color(n):
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return TEXT_SECONDARY
    return GREEN if n >= 0 else RED


def kpi_card(label, value, sub=None, value_color=None, bg='#fff'):
    return html.Div([
        html.Div(label, style={'fontSize': '12px', 'color': TEXT_SECONDARY, 'marginBottom': '8px'}),
        serif(value, 22, extra={'color': value_color} if value_color else None),
        (html.Div(sub, style={'fontSize': '12px', 'color': TEXT_SECONDARY, 'marginTop': '6px'})
         if sub and isinstance(sub, str) else sub) if sub else None,
    ], style={**CARD, 'background': bg})


def extract_strategy_type(tag):
    if not isinstance(tag, str):
        return 'unknown'
    return tag.replace('-', '_').split('_')[0]


def filter_by_range(df, timerange):
    now = pd.Timestamp.now()
    if timerange == '1W':
        cutoff = now - pd.Timedelta(days=7)
    elif timerange == '1M':
        cutoff = now - pd.Timedelta(days=30)
    elif timerange == '3M':
        cutoff = now - pd.Timedelta(days=90)
    elif timerange == 'YTD':
        cutoff = pd.Timestamp(now.year, 1, 1)
    else:
        return df
    return df[df['time_close'] >= cutoff]


def prep_deals(merged_deals):
    if merged_deals is None or merged_deals.empty:
        return pd.DataFrame()
    df = merged_deals.copy()
    df['total_profit'] = df['profit_open'] + df['profit_close']
    df['strategy_type'] = df['comment_open'].apply(extract_strategy_type)
    return df


def strategy_agg(df):
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby('strategy_type')
    agg = grp['total_profit'].agg(
        pnl='sum', trades='count',
        win_rate=lambda x: (x > 0).mean() * 100,
    ).reset_index()
    agg['avg_profit'] = agg['pnl'] / agg['trades']

    def adv(g):
        p = g['total_profit']
        if len(p) < 2:
            return pd.Series({'sharpe': 0.0, 'max_dd': 0.0, 'profit_factor': 0.0})
        cum = p.sort_index().cumsum()
        dd = (cum - cum.cummax()).min()
        gains = p[p > 0].sum()
        losses = abs(p[p < 0].sum())
        pf = gains / losses if losses > 0 else 99.0
        sh = p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0.0
        return pd.Series({'sharpe': sh, 'max_dd': dd, 'profit_factor': min(pf, 99)})

    adv_df = df.groupby('strategy_type').apply(adv).reset_index()
    result = agg.merge(adv_df, on='strategy_type').sort_values('pnl', ascending=False)
    return result


def sparkline_svg(values, width=120, height=32):
    if not values or len(values) < 2:
        return html.Div(style={'width': f'{width}px', 'height': f'{height}px'})
    total = sum(v for v in values if v is not None)
    color = GREEN if total >= 0 else RED
    fill = GREEN_SOFT if total >= 0 else RED_SOFT
    mn = min(min(values), 0)
    mx = max(max(values), 0)
    rng = (mx - mn) or 1
    lo, hi = mn - rng * 0.2, mx + rng * 0.2
    n = len(values)

    def sx(i): return round((i / (n - 1)) * width, 1)
    def sy(v): return round(height - ((v - lo) / (hi - lo)) * height, 1)

    pts = [(sx(i), sy(v)) for i, v in enumerate(values)]
    line_d = 'M ' + ' L '.join(f"{x} {y}" for x, y in pts)
    area_d = line_d + f' L {width} {height} L 0 {height} Z'
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
        f'<path d="{area_d}" fill="{fill}" stroke="none"/>'
        f'<path d="{line_d}" fill="none" stroke="{color}" stroke-width="2"/>'
        f'</svg>'
    )
    b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return html.Img(src=f'data:image/svg+xml;base64,{b64}',
                    style={'width': f'{width}px', 'height': f'{height}px', 'display': 'block'})


def pnl_figure(dates, cum_pnl, height=200):
    total = cum_pnl[-1] if cum_pnl else 0
    line_color = GREEN if total >= 0 else RED
    fig = go.Figure()
    if dates:
        hwm = list(np.maximum.accumulate(cum_pnl))

        # Y range: fit tightly to data, include 0 as reference but don't waste space
        all_vals = cum_pnl + hwm + [0]
        lo, hi = min(all_vals), max(all_vals)
        spread = (hi - lo) or abs(lo) or 1
        pad = spread * 0.12
        y_range = [lo - pad, hi + pad]

        # HWM dotted line
        fig.add_trace(go.Scatter(
            x=dates, y=hwm, mode='lines',
            line=dict(color='rgba(38,34,29,0.20)', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip',
        ))
        # Drawdown fill between HWM and curve
        fig.add_trace(go.Scatter(
            x=dates, y=cum_pnl, fill='tonexty',
            fillcolor='rgba(181,71,58,0.10)', line=dict(width=0),
            showlegend=False, hoverinfo='skip',
        ))
        # Main P&L line — fill to zero baseline (not axis edge)
        fill_color = 'rgba(61,122,84,0.08)' if total >= 0 else 'rgba(181,71,58,0.08)'
        fig.add_trace(go.Scatter(
            x=dates, y=cum_pnl,
            fill='tozeroy', fillcolor=fill_color,
            line=dict(color=line_color, width=2), mode='lines',
            hovertemplate='%{x|%b %d}<br>$%{y:,.2f}<extra></extra>',
        ))
        fig.add_hline(y=0, line=dict(color='rgba(38,34,29,0.15)', width=1))

    fig.update_layout(
        margin=dict(l=0, r=0, t=4, b=0), height=height,
        paper_bgcolor='white', plot_bgcolor='white', showlegend=False,
        xaxis=dict(showgrid=False, showline=False, zeroline=False,
                   tickfont=dict(size=10, color=TEXT_SECONDARY)),
        yaxis=dict(
            showgrid=True, gridcolor='rgba(38,34,29,0.05)', zeroline=False,
            showline=False, tickprefix='$', tickfont=dict(size=10, color=TEXT_SECONDARY),
            range=y_range if dates else None,
        ),
    )
    return fig


def range_bar(timerange):
    return html.Div(style={
        'display': 'flex', 'gap': '4px', 'background': '#fff',
        'border': f'1px solid {BORDER}', 'borderRadius': '9px', 'padding': '3px',
    }, children=[
        html.Button(r, id={'type': 'timerange-btn', 'index': r}, n_clicks=0, style={
            'padding': '6px 13px', 'borderRadius': '6px', 'fontSize': '12.5px',
            'fontWeight': '500', 'cursor': 'pointer', 'border': 'none', 'fontFamily': 'inherit',
            'background': ACCENT if r == timerange else 'transparent',
            'color': '#fff' if r == timerange else TEXT_SECONDARY,
        }) for r in ['1W', '1M', '3M', 'YTD', 'All']
    ])


def loading_placeholder():
    return html.Div([
        html.Div("Loading trading data…",
                 style={'color': TEXT_SECONDARY, 'fontSize': '14px', 'textAlign': 'center'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
              'minHeight': '300px', 'padding': '40px'})


# ─── OVERVIEW ─────────────────────────────────────────────────────────────────

def render_overview(merged_deals, account_info, timerange, selected_strategy):
    if selected_strategy:
        return render_detail(merged_deals, selected_strategy)
    return render_overview_list(merged_deals, account_info, timerange)


def render_overview_list(merged_deals, account_info, timerange):
    balance = account_info.get('balance', 0) if account_info else 0
    equity = account_info.get('equity', 0) if account_info else 0

    df = prep_deals(merged_deals)
    df_range = filter_by_range(df, timerange) if not df.empty else df

    closed_pnl = df_range['total_profit'].sum() if not df_range.empty else 0
    total_trades = len(df_range) if not df_range.empty else 0
    win_rate = (df_range['total_profit'] > 0).mean() * 100 if not df_range.empty else 0

    # Expectancy per trade
    if not df_range.empty:
        wins = df_range[df_range['total_profit'] > 0]['total_profit']
        losses = df_range[df_range['total_profit'] < 0]['total_profit']
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = abs(losses.mean()) if not losses.empty else 0
        wr = win_rate / 100
        expectancy = avg_win * wr - avg_loss * (1 - wr)
    else:
        expectancy = 0.0

    # Max drawdown from full history (not range-filtered)
    if not df.empty:
        cum_all = df.sort_values('time_close')['total_profit'].cumsum()
        max_dd = (cum_all - cum_all.cummax()).min()
    else:
        max_dd = 0.0

    # Live open P&L from MT5
    open_pnl = 0.0
    try:
        import MetaTrader5 as mt5
        positions = mt5.positions_get()
        if positions:
            open_pnl = sum(p.profit for p in positions)
    except Exception:
        pass

    # strategy_agg is computed later on df_range so table P&Ls match the KPI
    best_name, best_pnl = '—', 0

    # Chart
    if not df.empty:
        dfc = filter_by_range(df, timerange).sort_values('time_close')
        dates = dfc['time_close'].tolist()
        cum_pnl = dfc['total_profit'].cumsum().tolist()
    else:
        dates, cum_pnl = [], []

    alerts = _build_alerts(df)
    s_df = strategy_agg(df_range)
    if not s_df.empty:
        best = s_df.iloc[0]
        best_name = STRATEGY_NAMES.get(best['strategy_type'], best['strategy_type'])
        best_pnl = best['pnl']
    strat_rows = _strategy_rows(s_df, df_range)

    return html.Div([
        # Header
        html.Div([
            html.Div([
                serif("Overview", 26),
                html.Div("Real-time snapshot across all strategies",
                         style={'fontSize': '13px', 'color': TEXT_SECONDARY, 'marginTop': '2px'}),
            ]),
            range_bar(timerange),
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
                  'flexWrap': 'wrap', 'gap': '14px'}),

        # KPI rows — 2 rows of 3
        html.Div([
            html.Div([
                kpi_card("Balance", f"${balance:,.2f}", f"Equity ${equity:,.2f}"),
                kpi_card(f"Closed P&L · {timerange}", fmt(closed_pnl, signed=True),
                         f"{total_trades:,} trades", value_color=pnl_color(closed_pnl)),
                kpi_card("Expectancy / Trade", fmt(expectancy, signed=True),
                         f"{win_rate:.1f}% win rate · {total_trades} trades",
                         value_color=pnl_color(expectancy)),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3,1fr)', 'gap': '14px'}),
            html.Div([
                kpi_card("Max Drawdown", fmt(max_dd, signed=True),
                         "all-time, closed trades", value_color=RED if max_dd < -1 else TEXT_PRIMARY),
                kpi_card("Open P&L", fmt(open_pnl, signed=True),
                         "live positions", value_color=pnl_color(open_pnl)),
                kpi_card("Top Strategy", best_name,
                         html.Div(fmt(best_pnl, signed=True),
                                  style={'fontSize': '12.5px', 'fontWeight': '600',
                                         'color': pnl_color(best_pnl), 'marginTop': '6px'}),
                         bg=ACCENT_SOFT),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3,1fr)', 'gap': '14px'}),
        ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '14px'}),

        # Chart + Alerts
        html.Div([
            html.Div([
                html.Div("Account P&L trend",
                         style={'fontSize': '13px', 'fontWeight': '600', 'marginBottom': '14px'}),
                dcc.Graph(figure=pnl_figure(dates, cum_pnl, height=300),
                          config={'displayModeBar': False}, style={'height': '300px'}),
            ], style={**CARD}),
            html.Div([
                html.Div("Recent alerts",
                         style={'fontSize': '13px', 'fontWeight': '600', 'marginBottom': '13px'}),
                html.Div(alerts, style={'display': 'flex', 'flexDirection': 'column', 'gap': '13px',
                                        'overflowY': 'auto', 'maxHeight': '260px'}),
            ], style={**CARD}),
        ], style={'display': 'grid', 'gridTemplateColumns': '2fr 1fr', 'gap': '14px', 'alignItems': 'stretch'}),

        # Strategy table
        html.Div([
            html.Div([
                html.Div("Strategies", style={'fontSize': '13px', 'fontWeight': '600'}),
                html.Div(f"{len(s_df)} strategies" if not s_df.empty else "—",
                         style={'fontSize': '12px', 'color': TEXT_SECONDARY}),
            ], style={'padding': '16px 20px', 'borderBottom': f'1px solid {BORDER}',
                      'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'}),
            # Column headers
            html.Div([
                html.Div("Strategy"),
                html.Div("Trend"),
                html.Div("P&L", style={'textAlign': 'right'}),
                html.Div("Win %", style={'textAlign': 'right'}),
                html.Div("PF", style={'textAlign': 'right'}),
                html.Div("Trades", style={'textAlign': 'right'}),
                html.Div(),
            ], style={'display': 'grid', 'gridTemplateColumns': '1.4fr 80px 100px 70px 60px 70px 20px',
                      'gap': '14px', 'padding': '8px 20px', 'fontSize': '11px',
                      'color': TEXT_SECONDARY, 'fontWeight': '500'}),
            html.Div(strat_rows),
        ], style={'background': '#fff', 'border': f'1px solid {BORDER}',
                  'borderRadius': '12px', 'overflow': 'hidden'}),

    ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '24px', 'padding': '28px 40px 60px'})


def _build_alerts(df):
    items = []
    if df.empty:
        return [html.Div("No data available.", style={'fontSize': '12.5px', 'color': TEXT_SECONDARY})]

    # Strategy-level drawdown alerts
    for stype, grp in df.groupby('strategy_type'):
        cum = grp.sort_values('time_close')['total_profit'].cumsum()
        dd = (cum - cum.cummax()).min()
        strat_pnl = grp['total_profit'].sum()
        name = STRATEGY_NAMES.get(stype, stype)
        if dd < -500:
            items.append(_alert_dot(f"{name}: drawdown {fmt(dd, signed=True)}", "Risk", RED))
        elif strat_pnl < -200:
            items.append(_alert_dot(f"{name}: cumulative P&L {fmt(strat_pnl, signed=True)}", "Loss", RED))

    # Recent large individual losses (last 30 trades)
    recent = df.sort_values('time_close', ascending=False).head(30)
    for _, r in recent[recent['total_profit'] < -150].head(3).iterrows():
        sname = STRATEGY_NAMES.get(r['strategy_type'], r['strategy_type'])
        sym = r.get('symbol_open', '')
        items.append(_alert_dot(f"{sname} −${abs(r['total_profit']):,.0f} on {sym}",
                                r['time_close'].strftime('%b %d'), RED))

    # Recent big wins
    for _, r in recent[recent['total_profit'] > 250].head(2).iterrows():
        sname = STRATEGY_NAMES.get(r['strategy_type'], r['strategy_type'])
        sym = r.get('symbol_open', '')
        items.append(_alert_dot(f"{sname} +${r['total_profit']:,.0f} on {sym}",
                                r['time_close'].strftime('%b %d'), GREEN))

    if not items:
        items = [html.Div("No significant events.", style={'fontSize': '12.5px', 'color': TEXT_SECONDARY})]
    return items[:6]


def _alert_dot(text, ts, dot_color):
    return html.Div([
        html.Div(style={'width': '7px', 'height': '7px', 'borderRadius': '50%',
                        'background': dot_color, 'marginTop': '5px', 'flex': 'none'}),
        html.Div([
            html.Div(text, style={'fontSize': '12.5px', 'lineHeight': '1.4'}),
            html.Div(ts, style={'fontSize': '11px', 'color': TEXT_SECONDARY, 'marginTop': '2px'}),
        ]),
    ], style={'display': 'flex', 'gap': '9px', 'alignItems': 'flex-start'})


def _strategy_rows(s_df, df):
    if s_df.empty:
        return [html.Div("No strategy data available.",
                         style={'padding': '20px', 'color': TEXT_SECONDARY, 'fontSize': '13px'})]
    rows = []
    for _, row in s_df.iterrows():
        stype = row['strategy_type']
        name = STRATEGY_NAMES.get(stype, stype)
        pnl = row['pnl']

        spark_vals = []
        if not df.empty:
            s_df2 = df[df['strategy_type'] == stype].sort_values('time_close')
            vals = s_df2['total_profit'].cumsum().tolist()
            if len(vals) > 20:
                step = max(1, len(vals) // 20)
                vals = vals[::step][:20]
            spark_vals = vals

        dot = GREEN if pnl >= 0 else RED

        pf = row.get('profit_factor', 0)
        pf_color = GREEN if pf > 1.5 else (TEXT_SECONDARY if pf > 1.0 else RED)
        scolor = STRATEGY_COLORS.get(stype, TEXT_SECONDARY)

        rows.append(html.Div([
            html.Div([
                html.Div(style={'width': '3px', 'height': '32px', 'borderRadius': '2px',
                                'background': scolor, 'flex': 'none'}),
                html.Div([
                    html.Div(name, style={'fontSize': '13.5px', 'fontWeight': '500',
                                          'whiteSpace': 'nowrap', 'overflow': 'hidden',
                                          'textOverflow': 'ellipsis'}),
                    html.Div(stype, style={'fontSize': '11px', 'color': TEXT_SECONDARY}),
                ], style={'minWidth': '0'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'minWidth': '0'}),
            sparkline_svg(spark_vals, width=80),
            html.Div(fmt(pnl, signed=True),
                     style={'fontSize': '13px', 'fontWeight': '600',
                            'color': pnl_color(pnl), 'textAlign': 'right'}),
            html.Div(f"{row['win_rate']:.1f}%", style={'fontSize': '13px', 'textAlign': 'right',
                                                         'color': TEXT_SECONDARY}),
            html.Div(f"{pf:.1f}x", style={'fontSize': '13px', 'textAlign': 'right',
                                           'fontWeight': '500', 'color': pf_color}),
            html.Div(f"{int(row['trades']):,}", style={'fontSize': '13px', 'color': TEXT_SECONDARY,
                                                        'textAlign': 'right'}),
            html.Div("›", style={'fontSize': '13px', 'color': TEXT_SECONDARY, 'textAlign': 'right'}),
        ], id={'type': 'strategy-row', 'index': stype}, n_clicks=0, style={
            'display': 'grid', 'gridTemplateColumns': '1.4fr 80px 100px 70px 60px 70px 20px',
            'alignItems': 'center', 'gap': '14px', 'padding': '13px 20px',
            'borderBottom': f'1px solid {BORDER}', 'cursor': 'pointer',
        }, className='strategy-row'))

    return rows


# ─── DETAIL ───────────────────────────────────────────────────────────────────

def render_detail(merged_deals, strategy_id):
    df = prep_deals(merged_deals)
    if df.empty:
        return html.Div("No data.", style={'padding': '40px', 'color': TEXT_SECONDARY})

    df = df[df['strategy_type'] == strategy_id]
    if df.empty:
        return html.Div(f"No trades for {strategy_id}.",
                        style={'padding': '40px', 'color': TEXT_SECONDARY})

    name = STRATEGY_NAMES.get(strategy_id, strategy_id)
    pnl = df['total_profit'].sum()
    trades = len(df)
    win_rate = (df['total_profit'] > 0).mean() * 100
    avg_profit = df['total_profit'].mean()

    dfs = df.sort_values('time_close')
    cum = dfs['total_profit'].cumsum()
    dd = (cum - cum.cummax()).min()
    gains = df[df['total_profit'] > 0]['total_profit'].sum()
    losses = abs(df[df['total_profit'] < 0]['total_profit'].sum())
    pf = min(gains / losses, 99.0) if losses > 0 else 99.0
    sh = df['total_profit'].mean() / df['total_profit'].std() * np.sqrt(252) \
        if df['total_profit'].std() > 0 else 0.0

    dates = dfs['time_close'].tolist()
    cum_pnl = cum.tolist()

    sym_grp = df.groupby('symbol_open')['total_profit'].agg(
        pnl='sum', trades='count', win_rate=lambda x: (x > 0).mean() * 100
    ).reset_index().sort_values('pnl', ascending=False)

    sym_rows = []
    for _, r in sym_grp.iterrows():
        avg = r['pnl'] / r['trades']
        sym_rows.append(html.Div([
            html.Div(r['symbol_open'], style={'fontWeight': '500'}),
            html.Div(fmt(r['pnl'], signed=True),
                     style={'textAlign': 'right', 'color': pnl_color(r['pnl']), 'fontWeight': '500'}),
            html.Div(f"{r['win_rate']:.1f}%", style={'textAlign': 'right'}),
            html.Div(fmt(avg, signed=True), style={'textAlign': 'right', 'color': TEXT_SECONDARY}),
            html.Div(f"{int(r['trades'])}", style={'textAlign': 'right', 'color': TEXT_SECONDARY}),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 100px 90px 100px 80px',
                  'gap': '14px', 'padding': '11px 20px',
                  'borderTop': f'1px solid {BORDER}', 'fontSize': '13px'}))

    return html.Div([
        html.Div([
            html.Div([
                html.Span("Overview", style={'color': TEXT_SECONDARY}),
                html.Span(" / ", style={'color': TEXT_SECONDARY, 'margin': '0 6px'}),
                html.Span(STRATEGY_NAMES.get(strategy_id, strategy_id),
                          style={'color': TEXT_PRIMARY, 'fontWeight': '500'}),
            ], style={'fontSize': '12.5px', 'marginBottom': '14px',
                      'display': 'flex', 'alignItems': 'center'}),
            html.Div([
                serif(name, 24),
                html.Div(strategy_id, style={'fontSize': '12px', 'color': TEXT_SECONDARY}),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'flexWrap': 'wrap'}),
        ]),

        html.Div([
            kpi_card("Total P&L", fmt(pnl, signed=True), value_color=pnl_color(pnl)),
            kpi_card("Win Rate", f"{win_rate:.1f}%"),
            kpi_card("Avg Profit / Trade", fmt(avg_profit, signed=True), value_color=pnl_color(avg_profit)),
            kpi_card("Trades", f"{trades:,}"),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4,1fr)', 'gap': '14px'}),

        # Advanced metrics collapsible
        html.Div([
            html.Details([
                html.Summary([
                    html.Span("Advanced metrics",
                              style={'fontSize': '12.5px', 'color': TEXT_SECONDARY, 'fontWeight': '500'}),
                    html.Span("Sharpe · Max Drawdown · Profit Factor",
                              style={'fontSize': '11px', 'color': TEXT_SECONDARY, 'marginLeft': '8px'}),
                ], style={'padding': '14px 20px', 'cursor': 'pointer',
                           'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'}),
                html.Div([
                    html.Div([
                        html.Div("Sharpe", style={'fontSize': '11px', 'color': TEXT_SECONDARY}),
                        html.Div(f"{sh:.2f}", style={'fontSize': '15px', 'fontWeight': '600', 'marginTop': '3px'}),
                    ]),
                    html.Div([
                        html.Div("Max Drawdown", style={'fontSize': '11px', 'color': TEXT_SECONDARY}),
                        html.Div(fmt(dd, signed=True),
                                 style={'fontSize': '15px', 'fontWeight': '600', 'marginTop': '3px',
                                        'color': RED if dd < 0 else TEXT_PRIMARY}),
                    ]),
                    html.Div([
                        html.Div("Profit Factor", style={'fontSize': '11px', 'color': TEXT_SECONDARY}),
                        html.Div(f"{pf:.2f}", style={'fontSize': '15px', 'fontWeight': '600', 'marginTop': '3px'}),
                    ]),
                ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3,1fr)',
                          'gap': '14px', 'padding': '0 20px 18px'}),
            ]),
        ], style={'background': '#fff', 'border': f'1px solid {BORDER}', 'borderRadius': '12px'}),

        html.Div([
            html.Div("P&L trend", style={'fontSize': '13px', 'fontWeight': '600', 'marginBottom': '14px'}),
            dcc.Graph(figure=pnl_figure(dates, cum_pnl, height=320), config={'displayModeBar': False},
                      style={'height': '320px'}),
        ], style={**CARD}),

        html.Div([
            html.Div("Breakdown by symbol",
                     style={'padding': '16px 20px', 'borderBottom': f'1px solid {BORDER}',
                            'fontSize': '13px', 'fontWeight': '600'}),
            html.Div([
                html.Div("Symbol"), html.Div("P&L", style={'textAlign': 'right'}),
                html.Div("Win Rate", style={'textAlign': 'right'}),
                html.Div("Avg Profit", style={'textAlign': 'right'}),
                html.Div("Trades", style={'textAlign': 'right'}),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 100px 90px 100px 80px',
                      'gap': '14px', 'padding': '10px 20px',
                      'fontSize': '11px', 'color': TEXT_SECONDARY, 'fontWeight': '500'}),
            html.Div(sym_rows),
        ], style={'background': '#fff', 'border': f'1px solid {BORDER}',
                  'borderRadius': '12px', 'overflow': 'hidden'}),

    ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '24px', 'padding': '28px 40px 60px'})


# ─── TRADES ───────────────────────────────────────────────────────────────────

def _pos_duration(open_timestamp):
    from datetime import datetime
    try:
        opened = datetime.fromtimestamp(open_timestamp)
        delta = datetime.now() - opened
        total_mins = int(delta.total_seconds()) // 60
        d, h, m = total_mins // 1440, (total_mins % 1440) // 60, total_mins % 60
        if d > 0:
            return f"{d}d {h}h", RED
        elif h >= 8:
            return f"{h}h {m:02d}m", RED
        elif h >= 4:
            return f"{h}h {m:02d}m", '#B98A2E'
        elif h > 0:
            return f"{h}h {m:02d}m", TEXT_SECONDARY
        else:
            return f"{m}m", TEXT_SECONDARY
    except Exception:
        return '—', TEXT_SECONDARY


def render_trades(merged_deals, day_offset=0):
    today = date.today()
    day_offset = int(day_offset or 0)

    # If showing today and no closed trades, jump to the last trading day
    if day_offset == 0 and merged_deals is not None:
        df_tmp = prep_deals(merged_deals)
        if not df_tmp.empty:
            df_tmp['_date'] = df_tmp['time_close'].dt.date
            if today not in df_tmp['_date'].values:
                last_day = df_tmp['_date'].max()
                day_offset = (last_day - today).days  # negative number

    target = today + timedelta(days=day_offset)
    if day_offset == 0:
        date_label = "Today"
    elif day_offset == -1:
        date_label = "Yesterday"
    else:
        date_label = target.strftime('%a, %b %d')

    open_pnl = 0.0
    closed_pnl = 0.0
    closed_win_rate = 0.0
    open_rows = []
    closed_rows = []
    worst_pos = None

    # Live open positions (today only)
    if day_offset == 0:
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get()
            if positions:
                # Sort worst P&L first
                positions = sorted(positions, key=lambda p: p.profit)
                open_pnl = sum(p.profit for p in positions)
                worst_pos = positions[0] if positions else None
                for pos in positions:
                    stype = extract_strategy_type(pos.comment or '')
                    sname = STRATEGY_NAMES.get(stype, pos.comment or '—')
                    side = 'Buy' if pos.type == 0 else 'Sell'
                    dur_str, dur_color = _pos_duration(pos.time)
                    row_bg = 'rgba(181,71,58,0.04)' if pos.profit < 0 else 'rgba(61,122,84,0.04)'
                    open_rows.append(_trade_row_open(
                        "LIVE", sname, stype, pos.symbol, side,
                        f"{pos.price_open:.5g}", pos.profit, dur_str, dur_color, row_bg,
                        trade_key=f"o_{pos.ticket}",
                    ))
        except Exception:
            pass

    # Closed trades from history for the selected day
    df = prep_deals(merged_deals)
    if not df.empty:
        df['date'] = df['time_close'].dt.date
        day_df = df[df['date'] == target].sort_values('total_profit')  # worst first
        if not day_df.empty:
            closed_pnl = day_df['total_profit'].sum()
            wins = (day_df['total_profit'] > 0).sum()
            closed_win_rate = wins / len(day_df) * 100
            for _, r in day_df.iterrows():
                p = r['total_profit']
                side = 'Buy' if r.get('type_open', 0) == 0 else 'Sell'
                sym = r.get('symbol_open', '—')
                stype = r['strategy_type']
                entry = r.get('price_open')
                entry_str = f"{entry:.5g}" if entry else '—'
                sname = STRATEGY_NAMES.get(stype, stype)
                row_bg = 'rgba(181,71,58,0.04)' if p < 0 else 'rgba(61,122,84,0.04)'
                pos_id = r.get('position_id')
                tk = f"c_{int(float(pos_id))}" if pos_id is not None else None
                closed_rows.append(_trade_row_closed(
                    r['time_close'].strftime('%H:%M'), sname, stype,
                    sym, side, entry_str, p, row_bg,
                    trade_key=tk,
                ))

    worst_label = '—'
    worst_color = TEXT_SECONDARY
    if worst_pos is not None:
        worst_label = f"{worst_pos.symbol}  {fmt(worst_pos.profit, signed=True)}"
        worst_color = RED if worst_pos.profit < 0 else GREEN

    can_fwd = day_offset < 0

    def _section(title, header_fn, rows, empty_msg):
        return html.Div([
            html.Div(title, style={
                'padding': '12px 16px', 'fontSize': '12px', 'fontWeight': '600',
                'color': TEXT_SECONDARY, 'borderBottom': f'1px solid {BORDER}',
                'background': 'rgba(38,34,29,0.02)',
            }),
            header_fn(),
            html.Div(rows or [html.Div(empty_msg, style={
                'padding': '16px', 'color': TEXT_SECONDARY, 'fontSize': '13px'
            })]),
        ], style={'background': '#fff', 'border': f'1px solid {BORDER}',
                  'borderRadius': '12px', 'overflow': 'hidden'})

    return html.Div([
        html.Div([
            html.Div([
                serif("Trades", 26),
                html.Div("Open positions and closed trades for the day",
                         style={'fontSize': '13px', 'color': TEXT_SECONDARY, 'marginTop': '2px'}),
            ]),
            html.Div([
                html.Button("‹", id='prev-day-btn', n_clicks=0, style=_nav_btn_style()),
                html.Div(date_label, style={'fontSize': '13px', 'fontWeight': '500',
                                            'minWidth': '120px', 'textAlign': 'center'}),
                html.Button("›", id='next-day-btn', n_clicks=0, style={
                    **_nav_btn_style(),
                    'cursor': 'pointer' if can_fwd else 'default',
                    'color': TEXT_PRIMARY if can_fwd else 'rgba(38,34,29,0.25)',
                }),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px',
                      'background': '#fff', 'border': f'1px solid {BORDER}',
                      'borderRadius': '9px', 'padding': '4px 6px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
                  'flexWrap': 'wrap', 'gap': '14px'}),

        html.Div([
            kpi_card("Open P&L", fmt(open_pnl, signed=True),
                     f"{len(open_rows)} live positions", value_color=pnl_color(open_pnl)),
            kpi_card("Closed P&L", fmt(closed_pnl, signed=True),
                     f"{closed_win_rate:.1f}% win rate", value_color=pnl_color(closed_pnl)),
            kpi_card("Closed Trades", str(len(closed_rows)), date_label),
            kpi_card("Largest Loser", worst_label, "open position", value_color=worst_color),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4,1fr)', 'gap': '14px'}),

        _section(f"Open Positions · {len(open_rows)}", _trade_header_open, open_rows,
                 "No open positions."),
        _section(f"Closed · {date_label}", _trade_header_closed, closed_rows,
                 "No closed trades for this day."),

    ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '24px', 'padding': '28px 40px 60px'})


_OPEN_COLS = '46px 3px minmax(120px,1fr) 70px 44px 70px 70px 60px'
_CLOSED_COLS = '46px 3px minmax(120px,1fr) 70px 44px 70px 70px'


def _trade_header_open():
    return html.Div([
        html.Div("Time"), html.Div(), html.Div("Strategy"),
        html.Div("Symbol"), html.Div("Side"),
        html.Div("Entry", style={'textAlign': 'right'}),
        html.Div("P&L", style={'textAlign': 'right'}),
        html.Div("Duration", style={'textAlign': 'right'}),
    ], style={'display': 'grid', 'gridTemplateColumns': _OPEN_COLS,
              'gap': '8px', 'padding': '8px 16px', 'fontSize': '11px',
              'color': TEXT_SECONDARY, 'fontWeight': '500', 'borderBottom': f'1px solid {BORDER}'})


def _trade_header_closed():
    return html.Div([
        html.Div("Time"), html.Div(), html.Div("Strategy"),
        html.Div("Symbol"), html.Div("Side"),
        html.Div("Entry", style={'textAlign': 'right'}),
        html.Div("P&L", style={'textAlign': 'right'}),
    ], style={'display': 'grid', 'gridTemplateColumns': _CLOSED_COLS,
              'gap': '8px', 'padding': '8px 16px', 'fontSize': '11px',
              'color': TEXT_SECONDARY, 'fontWeight': '500', 'borderBottom': f'1px solid {BORDER}'})


def _trade_row_open(time_str, strategy, stype, symbol, side, entry, pnl_val,
                    dur_str, dur_color, row_bg, trade_key=None):
    scolor = STRATEGY_COLORS.get(stype, TEXT_SECONDARY)
    kwargs = {}
    if trade_key:
        kwargs['id'] = {'type': 'trade-row', 'index': trade_key}
        kwargs['n_clicks'] = 0
    _base_style = {
        'display': 'grid', 'gridTemplateColumns': _OPEN_COLS,
        'gap': '8px', 'padding': '11px 16px', 'alignItems': 'center',
        'borderTop': f'1px solid {BORDER}', 'background': row_bg,
        'cursor': 'pointer' if trade_key else 'default',
        'width': '100%', 'border': 'none', 'fontFamily': 'inherit', 'textAlign': 'left',
    }
    children = [
        html.Div(time_str, style={'color': ACCENT, 'fontSize': '11px',
                                   'fontWeight': '700', 'whiteSpace': 'nowrap'}),
        html.Div(style={'width': '3px', 'height': '100%', 'background': scolor,
                        'borderRadius': '2px'}),
        html.Div(strategy, style={'fontWeight': '500', 'fontSize': '12.5px',
                                   'whiteSpace': 'nowrap', 'overflow': 'hidden',
                                   'textOverflow': 'ellipsis', 'minWidth': '0'}),
        html.Div(symbol, style={'fontSize': '12.5px'}),
        html.Div(side, style={'color': GREEN if side == 'Buy' else RED, 'fontSize': '12px',
                               'fontWeight': '500'}),
        html.Div(entry, style={'textAlign': 'right', 'color': TEXT_SECONDARY, 'fontSize': '12px'}),
        html.Div(fmt(pnl_val, signed=True),
                 style={'textAlign': 'right', 'fontWeight': '600', 'fontSize': '13px',
                        'color': pnl_color(pnl_val)}),
        html.Div(dur_str, style={'textAlign': 'right', 'fontSize': '11.5px',
                                  'color': dur_color, 'fontWeight': '500'}),
    ]
    if trade_key:
        return html.Button(children, style=_base_style, **kwargs)
    return html.Div(children, style=_base_style)


def _trade_row_closed(time_str, strategy, stype, symbol, side, entry, pnl_val, row_bg,
                      trade_key=None):
    scolor = STRATEGY_COLORS.get(stype, TEXT_SECONDARY)
    kwargs = {}
    if trade_key:
        kwargs['id'] = {'type': 'trade-row', 'index': trade_key}
        kwargs['n_clicks'] = 0
    _base_style = {
        'display': 'grid', 'gridTemplateColumns': _CLOSED_COLS,
        'gap': '8px', 'padding': '10px 16px', 'alignItems': 'center',
        'borderTop': f'1px solid {BORDER}', 'background': row_bg,
        'cursor': 'pointer' if trade_key else 'default',
        'width': '100%', 'border': 'none', 'fontFamily': 'inherit', 'textAlign': 'left',
    }
    children = [
        html.Div(time_str, style={'color': TEXT_SECONDARY, 'fontSize': '12px',
                                   'whiteSpace': 'nowrap'}),
        html.Div(style={'width': '3px', 'height': '100%', 'background': scolor,
                        'borderRadius': '2px'}),
        html.Div(strategy, style={'fontWeight': '500', 'fontSize': '12.5px',
                                   'whiteSpace': 'nowrap', 'overflow': 'hidden',
                                   'textOverflow': 'ellipsis', 'minWidth': '0'}),
        html.Div(symbol, style={'fontSize': '12.5px'}),
        html.Div(side, style={'color': GREEN if side == 'Buy' else RED, 'fontSize': '12px'}),
        html.Div(entry, style={'textAlign': 'right', 'color': TEXT_SECONDARY, 'fontSize': '12px'}),
        html.Div(fmt(pnl_val, signed=True),
                 style={'textAlign': 'right', 'fontWeight': '500', 'fontSize': '13px',
                        'color': pnl_color(pnl_val)}),
    ]
    if trade_key:
        return html.Button(children, style=_base_style, **kwargs)
    return html.Div(children, style=_base_style)


# ─── TRADE MODAL ──────────────────────────────────────────────────────────────

def _meta_chip(label, value):
    return html.Div([
        html.Div(label, style={'fontSize': '11px', 'color': TEXT_SECONDARY, 'marginBottom': '2px'}),
        html.Div(value, style={'fontSize': '13px', 'fontWeight': '500'}),
    ])


def _trade_duration_str(trade_data):
    try:
        t0 = trade_data['time_open']
        t1 = trade_data.get('time_close') or pd.Timestamp.now().to_pydatetime()
        if hasattr(t0, 'to_pydatetime'):
            t0 = t0.to_pydatetime()
        if hasattr(t1, 'to_pydatetime'):
            t1 = t1.to_pydatetime()
        secs = int((t1 - t0).total_seconds())
        d, rem = divmod(secs, 86400)
        h, rem = divmod(rem, 3600)
        m = rem // 60
        if d > 0:
            return f"{d}d {h}h"
        elif h > 0:
            return f"{h}h {m:02d}m"
        return f"{m}m"
    except Exception:
        return '—'


def trade_candlestick_figure(candles_df, trade_data):
    fig = go.Figure()

    if candles_df is not None and not candles_df.empty:
        fig.add_trace(go.Candlestick(
            x=candles_df['time'],
            open=candles_df['open'], high=candles_df['high'],
            low=candles_df['low'], close=candles_df['close'],
            name='Price',
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            increasing_fillcolor='rgba(61,122,84,0.55)',
            decreasing_fillcolor='rgba(181,71,58,0.55)',
        ))

    t_open = trade_data.get('time_open')
    p_open = trade_data.get('price_open')
    t_close = trade_data.get('time_close')
    p_close = trade_data.get('price_close')
    pnl = trade_data.get('total_profit', 0)
    exit_color = GREEN if pnl >= 0 else RED

    if t_open and p_open:
        fig.add_trace(go.Scatter(
            x=[t_open], y=[p_open], mode='markers',
            marker=dict(symbol='triangle-up', size=13, color='#5B7FBA',
                        line=dict(color='white', width=2)),
            name='Entry',
            hovertemplate=f'Entry {p_open:.5g}<extra></extra>',
        ))
    if t_close and p_close:
        fig.add_trace(go.Scatter(
            x=[t_close], y=[p_close], mode='markers',
            marker=dict(symbol='triangle-down', size=13, color=exit_color,
                        line=dict(color='white', width=2)),
            name='Exit',
            hovertemplate=f'Exit {p_close:.5g}  P&L ${pnl:,.2f}<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=[t_open, t_close], y=[p_open, p_close], mode='lines',
            line=dict(color=exit_color, width=1.5, dash='dot'),
            showlegend=False, hoverinfo='skip',
        ))

    if candles_df is None or candles_df.empty:
        fig.add_annotation(text='No candle data available', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color=TEXT_SECONDARY))

    fig.update_layout(
        margin=dict(l=0, r=8, t=8, b=0), height=360,
        paper_bgcolor='white', plot_bgcolor='white', showlegend=False,
        xaxis=dict(showgrid=False, showline=False, zeroline=False,
                   tickfont=dict(size=10, color=TEXT_SECONDARY),
                   rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor='rgba(38,34,29,0.05)', zeroline=False,
                   showline=False, tickformat='.5g', side='right',
                   tickfont=dict(size=10, color=TEXT_SECONDARY)),
    )
    return fig


def trade_modal_overlay(trade_data, candles_df):
    symbol = trade_data.get('symbol', '—')
    side = trade_data.get('side', '—')
    pnl = trade_data.get('total_profit', 0)
    entry = trade_data.get('price_open', 0)
    exit_p = trade_data.get('price_close', 0)
    is_open = trade_data.get('is_open', False)

    fig = trade_candlestick_figure(candles_df, trade_data)

    return html.Div([
        # Backdrop — click to close
        html.Div(id={'type': 'modal-close', 'index': 'backdrop'}, n_clicks=0, style={
            'position': 'fixed', 'top': 0, 'left': 0,
            'width': '100vw', 'height': '100vh',
            'background': 'rgba(38,34,29,0.45)', 'zIndex': '999', 'cursor': 'pointer',
        }),
        # Card
        html.Div([
            # Header
            html.Div([
                html.Div([
                    html.Div(symbol, style={'fontSize': '17px', 'fontWeight': '700',
                                            'letterSpacing': '-0.01em'}),
                    html.Div(side, style={
                        'fontSize': '11px', 'fontWeight': '600',
                        'padding': '2px 8px', 'borderRadius': '5px',
                        'background': GREEN if side == 'Buy' else RED, 'color': 'white',
                    }),
                ] + ([html.Div('LIVE', style={
                    'fontSize': '10px', 'fontWeight': '700',
                    'padding': '2px 7px', 'borderRadius': '4px',
                    'background': ACCENT, 'color': 'white',
                })] if is_open else []),
                style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}),
                html.Div([
                    html.Div(fmt(pnl, signed=True), style={
                        'fontSize': '18px', 'fontWeight': '700', 'color': pnl_color(pnl),
                    }),
                    html.Button('×', id={'type': 'modal-close', 'index': 'btn'}, n_clicks=0,
                                style={'border': 'none', 'background': 'none', 'fontSize': '22px',
                                       'cursor': 'pointer', 'color': TEXT_SECONDARY,
                                       'padding': '0 4px', 'lineHeight': '1',
                                       'fontFamily': 'inherit'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '14px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                      'padding': '18px 22px 14px', 'borderBottom': f'1px solid {BORDER}'}),
            # Meta chips
            html.Div([
                _meta_chip("Entry", f"{entry:.5g}"),
                _meta_chip("Exit", f"{exit_p:.5g}" if not is_open else "Live"),
                _meta_chip("Duration", _trade_duration_str(trade_data)),
            ], style={'display': 'flex', 'gap': '24px', 'padding': '11px 22px',
                      'borderBottom': f'1px solid {BORDER}', 'background': 'rgba(38,34,29,0.015)'}),
            # Chart
            dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '360px'}),
        ], style={
            'position': 'fixed', 'top': '50%', 'left': '50%',
            'transform': 'translate(-50%, -50%)',
            'width': '700px', 'maxWidth': '94vw', 'maxHeight': '90vh',
            'background': 'white', 'borderRadius': '16px',
            'border': f'1px solid {BORDER}',
            'boxShadow': '0 24px 64px rgba(38,34,29,0.20)',
            'zIndex': '1000', 'overflow': 'hidden',
        }),
    ])


def _nav_btn_style():
    return {
        'padding': '6px 10px', 'borderRadius': '6px', 'cursor': 'pointer',
        'fontSize': '14px', 'background': 'none', 'border': 'none', 'fontFamily': 'inherit',
    }


# ─── SYSTEM ───────────────────────────────────────────────────────────────────

def render_system():
    try:
        from utils.pm2_utils import get_pm2_status
        processes = get_pm2_status()
    except Exception:
        processes = []

    infra, engines = [], []
    for p in processes:
        name = p.get('name', '').lower()
        if any(k in name for k in STRATEGY_ENGINE_KEYS):
            engines.append(p)
        else:
            infra.append(p)

    return html.Div([
        html.Div([
            serif("System", 26),
            html.Div("Running services and strategy engines",
                     style={'fontSize': '13px', 'color': TEXT_SECONDARY, 'marginTop': '2px'}),
        ]),
        _process_table("Infrastructure", infra),
        _process_table("Strategy engines", engines),
    ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '24px', 'padding': '28px 40px 60px'})


_SYS_COLS = '1.4fr 90px 100px 70px 60px 60px 1.6fr'


def _process_table(title, procs):
    header = html.Div([
        html.Div("Service"), html.Div("Status"), html.Div("Uptime"),
        html.Div("Restarts"), html.Div("CPU"), html.Div("Mem"), html.Div("Last seen"),
    ], style={'display': 'grid', 'gridTemplateColumns': _SYS_COLS,
              'gap': '12px', 'padding': '9px 20px',
              'fontSize': '11px', 'color': TEXT_SECONDARY, 'fontWeight': '500'})

    rows = []
    for p in procs:
        status = p.get('status', 'stopped')
        dot = GREEN if status == 'online' else RED if status == 'error' else '#B98A2E'
        label = 'Online' if status == 'online' else 'Errored' if status == 'error' else 'Stopped'
        cpu = p.get('cpu', 0)
        mem = p.get('memory', 0)
        rows.append(html.Div([
            html.Div(p.get('name', '—'), style={'fontWeight': '500', 'whiteSpace': 'nowrap',
                                                  'overflow': 'hidden', 'textOverflow': 'ellipsis'}),
            html.Div([
                html.Div(style={'width': '7px', 'height': '7px', 'borderRadius': '50%',
                                'background': dot, 'flex': 'none'}),
                html.Span(label),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px'}),
            html.Div(p.get('uptime', '—'), style={'color': TEXT_SECONDARY}),
            html.Div(str(p.get('restarts', 0)), style={'color': TEXT_SECONDARY}),
            html.Div(f"{cpu}%", style={'color': TEXT_SECONDARY}),
            html.Div(f"{mem}MB", style={'color': TEXT_SECONDARY}),
            html.Div("—", style={'color': TEXT_SECONDARY, 'fontFamily': 'ui-monospace,monospace',
                                  'fontSize': '11.5px'}),
        ], style={'display': 'grid', 'gridTemplateColumns': _SYS_COLS,
                  'gap': '12px', 'padding': '11px 20px', 'borderTop': f'1px solid {BORDER}',
                  'fontSize': '13px', 'alignItems': 'center'}))

    if not rows:
        rows = [html.Div("No processes found.",
                         style={'padding': '16px 20px', 'color': TEXT_SECONDARY, 'fontSize': '13px'})]

    return html.Div([
        html.Div(title, style={'padding': '14px 20px', 'borderBottom': f'1px solid {BORDER}',
                                'fontSize': '12.5px', 'fontWeight': '600', 'color': TEXT_SECONDARY}),
        header,
        html.Div(rows),
    ], style={'background': '#fff', 'border': f'1px solid {BORDER}',
              'borderRadius': '12px', 'overflow': 'hidden'})
