"""
Callbacks for the redesigned PM dashboard.
Three views: Overview, Trades, System.
"""

import json
from datetime import datetime, date

from dash import Input, Output, State, html, no_update, callback_context, ALL
import dash_bootstrap_components as dbc

from utils import (
    initialize_mt5,
    get_historical_data,
    process_deals_data,
    get_account_info,
)
from components.pm_views import (
    render_overview,
    render_trades,
    render_system,
    loading_placeholder,
    trade_modal_overlay,
)

DEFAULT_START_DATE = date(2025, 1, 1)
ACCENT = '#BF6A3D'
SIDEBAR_BG = '#EFECE6'
BORDER = 'rgba(38,34,29,0.10)'
TEXT_PRIMARY = '#26221D'
TEXT_SECONDARY = '#7A756C'
GREEN = '#3D7A54'

# In-process cache (shared across callbacks in same worker)
_cache = {
    "merged_deals": None,
    "account_info": None,
}

NAV_IDS = ['overview', 'trades', 'system']


def _fetch_trade_for_chart(trade_key):
    """Return (trade_data dict, candles_df) for a trade key like 'o_12345' or 'c_67890'."""
    import MetaTrader5 as mt5
    from datetime import datetime, timedelta

    trade_data = None

    if trade_key.startswith('o_'):
        ticket = int(trade_key[2:])
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return None, None
        pos = positions[0]
        time_open = datetime.fromtimestamp(pos.time)
        trade_data = {
            'symbol': pos.symbol,
            'time_open': time_open,
            'time_close': None,
            'price_open': pos.price_open,
            'price_close': pos.price_current,
            'total_profit': pos.profit,
            'side': 'Buy' if pos.type == 0 else 'Sell',
            'is_open': True,
        }
        duration_secs = (datetime.now() - time_open).total_seconds()

    elif trade_key.startswith('c_'):
        pos_id = int(trade_key[2:])
        merged = _cache.get('merged_deals')
        if merged is None or merged.empty:
            return None, None
        row = merged[merged['position_id'] == pos_id]
        if row.empty:
            return None, None
        r = row.iloc[0]
        time_open = r['time_open'].to_pydatetime()
        time_close = r['time_close'].to_pydatetime()
        trade_data = {
            'symbol': str(r.get('symbol_open', '')),
            'time_open': time_open,
            'time_close': time_close,
            'price_open': float(r.get('price_open', 0)),
            'price_close': float(r.get('price_close', 0)),
            'total_profit': float(r.get('profit_open', 0)) + float(r.get('profit_close', 0)),
            'side': 'Buy' if r.get('type_open', 0) == 0 else 'Sell',
            'is_open': False,
        }
        duration_secs = (time_close - time_open).total_seconds()
    else:
        return None, None

    # Pick timeframe and buffer based on trade duration
    if duration_secs < 3600:
        tf, buf = mt5.TIMEFRAME_M1, timedelta(minutes=20)
    elif duration_secs < 28800:
        tf, buf = mt5.TIMEFRAME_M5, timedelta(hours=2)
    elif duration_secs < 259200:
        tf, buf = mt5.TIMEFRAME_H1, timedelta(hours=10)
    else:
        tf, buf = mt5.TIMEFRAME_H4, timedelta(days=3)

    t0 = trade_data['time_open']
    t1 = trade_data['time_close'] or datetime.now()

    import pandas as pd
    rates = mt5.copy_rates_range(trade_data['symbol'], tf, t0 - buf, t1 + buf)
    if rates is None or len(rates) == 0:
        return trade_data, None

    candles_df = pd.DataFrame(rates)
    candles_df['time'] = pd.to_datetime(candles_df['time'], unit='s')
    return trade_data, candles_df


def register_callbacks(app):

    # ── 1. Auto-load MT5 data on startup ──────────────────────────────────────

    @app.callback(
        Output('data-store', 'data'),
        Output('account-info-store', 'data'),
        Output('connection-status', 'children'),
        Input('startup-trigger', 'n_intervals'),
        prevent_initial_call=False,
    )
    def auto_load_data(n_intervals):
        if n_intervals == 0:
            return no_update, no_update, "Connecting…"

        ok, msg = initialize_mt5()
        if not ok:
            return None, None, f"MT5 error: {msg}"

        from_dt = datetime.combine(DEFAULT_START_DATE, datetime.min.time())
        to_dt = datetime.now().replace(hour=23, minute=59, second=59)
        orders, deals, err = get_historical_data(from_dt, to_dt)

        if err or orders is None or deals is None:
            return None, None, f"Fetch error: {err or 'no data'}"

        merged = process_deals_data(deals)
        account_info = get_account_info()

        if merged is None or merged.empty:
            return None, account_info, "Connected — no trades found"

        _cache['merged_deals'] = merged
        _cache['account_info'] = account_info

        return {'ok': True, 'rows': len(merged)}, account_info, f"Connected · {len(merged)} trades"

    # ── 2. Nav clicks → nav-store ──────────────────────────────────────────────

    @app.callback(
        Output('nav-store', 'data'),
        Input('nav-overview', 'n_clicks'),
        Input('nav-trades', 'n_clicks'),
        Input('nav-system', 'n_clicks'),
        prevent_initial_call=True,
    )
    def update_nav(ov, tr, sy):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        mapping = {'nav-overview': 'overview', 'nav-trades': 'trades', 'nav-system': 'system'}
        return mapping.get(triggered, 'overview')

    # ── 3. Sidebar nav item styling ────────────────────────────────────────────

    @app.callback(
        *[Output(f'nav-{nav_id}', 'style') for nav_id in NAV_IDS],
        *[Output(f'nav-dot-{nav_id}', 'style') for nav_id in NAV_IDS],
        Input('nav-store', 'data'),
        Input('selected-strategy-store', 'data'),
    )
    def style_nav(nav, selected):
        active = nav if not selected else nav  # detail lives inside overview

        def btn_style(nav_id):
            is_active = nav_id == active or (nav_id == 'overview' and active == 'detail')
            return {
                'display': 'flex', 'alignItems': 'center', 'gap': '10px',
                'padding': '9px 10px', 'borderRadius': '8px', 'cursor': 'pointer',
                'fontSize': '13.5px', 'border': 'none', 'width': '100%',
                'textAlign': 'left', 'fontFamily': 'inherit',
                'background': '#fff' if is_active else 'transparent',
                'color': TEXT_PRIMARY if is_active else TEXT_SECONDARY,
                'fontWeight': '600' if is_active else '500',
            }

        def dot_style(nav_id):
            is_active = nav_id == active or (nav_id == 'overview' and active == 'detail')
            return {
                'width': '7px', 'height': '7px', 'borderRadius': '50%', 'flex': 'none',
                'background': ACCENT if is_active else 'rgba(38,34,29,0.25)',
            }

        return (
            *[btn_style(n) for n in NAV_IDS],
            *[dot_style(n) for n in NAV_IDS],
        )

    # ── 4. Strategy row click → selected-strategy-store ───────────────────────

    @app.callback(
        Output('selected-strategy-store', 'data'),
        Input({'type': 'strategy-row', 'index': ALL}, 'n_clicks'),
        Input('nav-overview', 'n_clicks'),
        Input('nav-trades', 'n_clicks'),
        Input('nav-system', 'n_clicks'),
        prevent_initial_call=True,
    )
    def handle_strategy_click(row_clicks, *_nav_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        prop = ctx.triggered[0]['prop_id']

        # Any nav click clears selection (Overview returns to list, others switch view)
        if any(x in prop for x in ('nav-overview', 'nav-trades', 'nav-system')):
            return None

        # Strategy row click → drill into detail
        if 'strategy-row' in prop:
            try:
                idx = json.loads(prop.split('.')[0])['index']
                return idx
            except Exception:
                return no_update

        return no_update

    # ── 5. Time range buttons ──────────────────────────────────────────────────

    @app.callback(
        Output('timerange-store', 'data'),
        Input({'type': 'timerange-btn', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True,
    )
    def update_timerange(clicks):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        prop = ctx.triggered[0]['prop_id']
        try:
            return json.loads(prop.split('.')[0])['index']
        except Exception:
            return no_update

    # ── 6. Day navigation ─────────────────────────────────────────────────────

    @app.callback(
        Output('day-offset-store', 'data'),
        Input('prev-day-btn', 'n_clicks'),
        Input('next-day-btn', 'n_clicks'),
        State('day-offset-store', 'data'),
        prevent_initial_call=True,
    )
    def update_day_offset(prev, nxt, offset):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        prop = ctx.triggered[0]['prop_id']
        offset = offset or 0
        if 'prev-day-btn' in prop:
            return offset - 1
        if 'next-day-btn' in prop:
            return min(0, offset + 1)
        return offset

    # ── 7. Trade row click → modal store ──────────────────────────────────────
    # nav-overview/nav-trades as static inputs anchor the callback so Dash 4.x
    # registers it at startup even before any trade-row components exist.

    @app.callback(
        Output('trade-modal-store', 'data'),
        Input({'type': 'trade-row', 'index': ALL}, 'n_clicks'),
        Input({'type': 'modal-close', 'index': ALL}, 'n_clicks'),
        Input('nav-overview', 'n_clicks'),
        Input('nav-trades', 'n_clicks'),
        Input('nav-system', 'n_clicks'),
    )
    def handle_trade_click(row_clicks, close_clicks, *_nav):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        prop = ctx.triggered[0]['prop_id']
        # Startup fire with no actual trigger
        if not prop or prop == '.':
            return no_update
        # Nav → close modal
        if any(x in prop for x in ('nav-overview', 'nav-trades', 'nav-system')):
            return None
        if 'modal-close' in prop:
            return None
        if 'trade-row' in prop:
            # Guard: ignore zero-click fires (initial render)
            try:
                idx = json.loads(prop.split('.')[0])['index']
                triggered_val = ctx.triggered[0].get('value', 0)
                if not triggered_val:
                    return no_update
                return {'key': idx}
            except Exception:
                return no_update
        return no_update

    # ── 8. Modal store → modal overlay ────────────────────────────────────────

    @app.callback(
        Output('trade-modal-container', 'children'),
        Input('trade-modal-store', 'data'),
    )
    def render_trade_modal_cb(data):
        if not data:
            return []
        trade_key = data.get('key', '')
        trade_data, candles_df = _fetch_trade_for_chart(trade_key)
        if trade_data is None:
            return []
        return trade_modal_overlay(trade_data, candles_df)

    # ── 9. Main content render ─────────────────────────────────────────────────

    @app.callback(
        Output('main-content', 'children'),
        Input('nav-store', 'data'),
        Input('data-store', 'data'),
        Input('account-info-store', 'data'),
        Input('selected-strategy-store', 'data'),
        Input('timerange-store', 'data'),
        Input('day-offset-store', 'data'),
    )
    def render_content(nav, data_store, account_info, selected_strategy, timerange, day_offset):
        merged = _cache.get('merged_deals')
        acc = _cache.get('account_info') or account_info or {}

        # Show loading state until data arrives
        if data_store is None and nav not in ('system',):
            return loading_placeholder()

        if nav == 'trades':
            return render_trades(merged, day_offset or 0)

        if nav == 'system':
            return render_system()

        # overview (default) + detail drill-down
        return render_overview(merged, acc, timerange or '1M', selected_strategy)
