"""
Callbacks Module
Handles all Dash callbacks for the MetaDAsh application
"""
from dash import Input, Output, State, html
import dash_bootstrap_components as dbc
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import pandas as pd
import calendar

from utils import (
    initialize_mt5,
    get_historical_data,
    process_deals_data,
    get_account_info,
    strategy_metrics,
    calculate_hourly_performance
)

from components import (
    render_overview_tab,
    render_detailed_tab,
    create_detailed_metrics_figure,
    render_pnl_tab,
    render_trades_tab,
    create_trades_table,
    render_raw_tab
)

# >>> NEW: import the calendar tab + helpers
from components.tab_daily_calendar import (
    render_daily_calendar_tab,    # layout builder
    MonthCtx,                     # month context dataclass
    _month_series,                # days grid generator
    _prep_instance_daily,         # daily aggregation
    _color_for_value,             # chip colors
    _format_money                 # money formatter
)

# Global storage for data
stored_data = {
    'history_orders': None,
    'history_deals': None,
    'merged_deals': None,
    'account_size': 100000
}

def register_callbacks(app):
    """
    Register all callbacks for the application
    """

    @app.callback(
        Output('connection-status', 'children'),
        Input('connect-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def connect_mt5(n_clicks):
        """Handle MT5 connection"""
        if n_clicks:
            success, message = initialize_mt5()
            if success:
                return dbc.Alert(message, color="success", className="status-alert")
            else:
                return dbc.Alert(message, color="danger", className="status-alert")
        return ""

    @app.callback(
        [Output('data-store', 'data'),
         Output('fetch-status', 'children'),
         Output('account-info-store', 'data')],
        [Input('fetch-btn', 'n_clicks')],
        [State('start-date', 'date'),
         State('end-date', 'date'),
         State('account-size', 'value')],
        prevent_initial_call=True
    )
    def fetch_data(n_clicks, start_date, end_date, account_size):
        """Fetch trading data from MT5"""
        if not n_clicks:
            return None, "", None

        # Convert dates
        from_date = datetime.strptime(start_date, '%Y-%m-%d')
        to_date = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)

        # Get historical data
        history_orders, history_deals, error = get_historical_data(from_date, to_date)

        if error:
            return None, dbc.Alert(error, color="danger", className="status-alert"), None

        if history_orders is None or history_deals is None:
            return None, dbc.Alert("Failed to retrieve data", color="danger", className="status-alert"), None

        # Process deals data
        merged_deals = process_deals_data(history_deals)

        if merged_deals is None or merged_deals.empty:
            return None, dbc.Alert("No valid trades found", color="warning", className="status-alert"), None

        # Get account info
        account_info = get_account_info()

        # Store data globally
        stored_data['history_orders'] = history_orders
        stored_data['history_deals'] = history_deals
        stored_data['merged_deals'] = merged_deals
        stored_data['account_size'] = account_size

        message = f"✓ Retrieved {len(history_orders)} orders and {len(history_deals)} deals"

        return (
            {'data_available': True},
            dbc.Alert(message, color="success", className="status-alert"),
            account_info
        )

    @app.callback(
        Output('tab-content', 'children'),
        [Input('tabs', 'active_tab'),
         Input('data-store', 'data'),
         Input('account-info-store', 'data')]
    )
    def render_tab_content(active_tab, data, account_info):
        """Render content based on selected tab"""
        if data is None or not data.get('data_available', False):
            return html.Div([
                html.H4("No Data Available", className="text-center text-muted mt-3"),
                html.P("Please connect to MT5 and fetch trading data to view analytics",
                      className="text-center text-muted")
            ], className="empty-state")

        merged_deals = stored_data['merged_deals']
        account_size = stored_data['account_size']

        if not account_info:
            account_info = {'balance': 0, 'equity': 0, 'margin': 0}

        if active_tab == "overview":
            return render_overview_tab(merged_deals, account_size, account_info)
        elif active_tab == "detailed":
            return render_detailed_tab(merged_deals, account_size)
        elif active_tab == "pnl":
            return render_pnl_tab(merged_deals, account_size)
        elif active_tab == "trades":
            return render_trades_tab(merged_deals)
        elif active_tab == "raw":
            return render_raw_tab(merged_deals, stored_data['history_orders'])
        # >>> NEW: calendar tab route
        elif active_tab == "calendar":
            # Pick sensible defaults from current data
            most_common_strategy = (merged_deals['comment_open']
                                    .dropna().mode().iloc[0] if not merged_deals.empty else None)
            most_common_symbol = (merged_deals['symbol_open']
                                  .dropna().mode().iloc[0] if not merged_deals.empty else None)
            return render_daily_calendar_tab(
                merged_deals=merged_deals,
                default_strategy=most_common_strategy or "metafvg",
                default_symbol=most_common_symbol or "EURUSD"
            )

        return html.Div()

    @app.callback(
        Output('detailed-metrics-graph', 'figure'),
        [Input('metrics-dropdown', 'value')],
        [State('data-store', 'data')]
    )
    def update_detailed_metrics(selected_metrics, data):
        """Update detailed metrics chart based on selection"""
        if not data or not selected_metrics:
            return go.Figure()

        merged_deals = stored_data['merged_deals']
        account_size = stored_data['account_size']

        if merged_deals is None:
            return go.Figure()

        # Calculate metrics
        strategy_metrics_df = merged_deals[
            ["profit_open", "profit_close", "comment_open", "symbol_open", "time_open"]
        ].copy()

        grouped_metrics = strategy_metrics_df.groupby(["comment_open", "symbol_open"]).apply(
            lambda x: strategy_metrics(x, account_size)
        )

        return create_detailed_metrics_figure(grouped_metrics, selected_metrics)

    @app.callback(
        Output('hourly-graph', 'figure'),
        [Input('strategy-dropdown', 'value')],
        [State('data-store', 'data')]
    )
    def update_hourly_graph(selected_strategy, data):
        """Update hourly performance graph"""
        if not data or not selected_strategy:
            return go.Figure()

        merged_deals = stored_data['merged_deals']
        account_size = stored_data['account_size']

        if merged_deals is None:
            return go.Figure()

        # Calculate hourly performance
        hourly_perf = calculate_hourly_performance(merged_deals, selected_strategy)

        fig = go.Figure()

        if not hourly_perf.empty:
            fig.add_trace(go.Bar(
                x=hourly_perf.index,
                y=hourly_perf["Average Profit by Trade"].fillna(0),
                name='Hourly Avg Profit',
                marker_color='#0066cc',
                hovertemplate='Hour: %{x}<br>Avg Profit: $%{y:.2f}<extra></extra>'
            ))

        fig.update_layout(
            title=f"Hourly Performance for {selected_strategy}",
            xaxis_title="Hour of Day",
            yaxis_title="Average Profit ($)",
            template='plotly_white',
            height=400
        )

        return fig

    @app.callback(
        Output('trades-table-container', 'children'),
        [Input('bot-dropdown', 'value')],
        [State('data-store', 'data')]
    )
    def update_trades_table(selected_bot, data):
        """Update trades table based on selected bot"""
        if not data or not selected_bot:
            return html.Div("Please select a bot/strategy", className="text-center text-muted")

        merged_deals = stored_data['merged_deals']

        if merged_deals is None:
            return html.Div("No data available", className="text-center text-muted")

        return create_trades_table(merged_deals, selected_bot)

    # Add missing import at the top if needed
    import dash_bootstrap_components as dbc

    # ------------------------------
    # NEW: Calendar callbacks
    # ------------------------------
    # ------------------------------
    # Calendar callbacks (deduped)
    # ------------------------------

    def _ctx_triggered_id():
        import dash
        ctx = dash.callback_context
        if not ctx.triggered:
            return None
        return ctx.triggered[0]["prop_id"].split(".")[0]

    # 1) Prev/Next buttons -> update the anchor date ONLY
    @app.callback(
        Output("cal_anchor_date", "date"),
        Input("cal_prev_month", "n_clicks"),
        Input("cal_next_month", "n_clicks"),
        State("cal_anchor_date", "date"),
        prevent_initial_call=True
    )
    def cal_change_month(prev_clicks, next_clicks, anchor_date):
        trigger = _ctx_triggered_id()
        dt = pd.to_datetime(anchor_date).date() if anchor_date else date.today().replace(day=1)
        mctx = MonthCtx(dt.year, dt.month)
        if trigger == "cal_prev_month":
            mctx = mctx.step(-1)
        elif trigger == "cal_next_month":
            mctx = mctx.step(+1)
        return f"{mctx.year:04d}-{mctx.month:02d}-01"

    # 2) Anchor date -> month label (single writer of cal_month_label.children)
    @app.callback(
        Output("cal_month_label", "children"),
        Input("cal_anchor_date", "date")
    )
    def cal_label_from_date(anchor_date):
        dt = pd.to_datetime(anchor_date).date() if anchor_date else date.today().replace(day=1)
        mctx = MonthCtx(dt.year, dt.month)
        return f"{calendar.month_name[mctx.month]} {mctx.year}"

    # 3) Anchor date + selectors -> grid + stats
    @app.callback(
        Output("cal_grid", "children"),
        Output("cal_stats", "children"),
        Input("cal_anchor_date", "date"),
        Input("cal_strategy", "value"),
        Input("cal_symbol", "value"),
    )
    def cal_render_grid(anchor_date, strategy, symbol):
        merged_deals = stored_data.get("merged_deals")
        # Build month context
        dt = pd.to_datetime(anchor_date).date() if anchor_date else date.today().replace(day=1)
        mctx = MonthCtx(dt.year, dt.month)

        if merged_deals is None or merged_deals.empty or not strategy or not symbol:
            cells = [html.Div([html.Div(str(d.day), className="calendar-day")],
                              className="calendar-cell is-out") for d in _month_series(mctx)]
            stats = html.Div("No trades for this month.", style={"color": "#6b7280"})
            return cells, stats

        # Daily aggregation for selected instance
        daily = _prep_instance_daily(merged_deals, strategy, symbol)
        daily = daily.set_index("date") if not daily.empty else pd.DataFrame(columns=["pnl", "n_trades"])

        # Grid cells
        cells = []
        for d in _month_series(mctx):
            in_month = (d.month == mctx.month)
            pnl = float(daily.loc[d, "pnl"]) if (not daily.empty and d in daily.index) else 0.0
            ntr = int(daily.loc[d, "n_trades"]) if (not daily.empty and d in daily.index) else 0

            c = _color_for_value(pnl)
            chip_style = {"background": c["bg"], "borderColor": c["border"], "color": c["text"]}

            cells.append(
                html.Div([
                    html.Div(str(d.day), className="calendar-day"),
                    html.Div([
                        html.Span(_format_money(pnl), className="pnl-chip", style=chip_style),
                        html.Span(f"{ntr} trades", className="trades-badge", title="Number of trades that day")
                    ], style={"display": "flex", "gap": "8px", "alignItems": "center"})
                ],
                    className="calendar-cell" + ("" if in_month else " is-out"),
                    title=f"{d:%b %d, %Y} — PnL: {_format_money(pnl)} • Trades: {ntr}")
            )

        # Monthly stats
        if daily.empty:
            stats = html.Div("No trades for this month.", style={"color": "#6b7280"})
        else:
            month_df = daily.loc[(daily.index >= mctx.first) & (daily.index <= mctx.last)]
            if month_df.empty:
                stats = html.Div("No trades for this month.", style={"color": "#6b7280"})
            else:
                total = month_df["pnl"].sum()
                win_days = (month_df["pnl"] > 0).sum()
                loss_days = (month_df["pnl"] < 0).sum()
                zero_days = (month_df["pnl"] == 0).sum()
                best = month_df["pnl"].max()
                worst = month_df["pnl"].min()
                avg = month_df["pnl"].mean()
                wr = 100 * win_days / max(1, len(month_df))
                stats = html.Div([
                    _stat_row("Total PnL", _format_money(total)),
                    _stat_row("Avg / day", _format_money(avg)),
                    _stat_row("Win rate (days)", f"{wr:.1f}%"),
                    _stat_row("Best day", _format_money(best)),
                    _stat_row("Worst day", _format_money(worst)),
                    _stat_row("Days: + / 0 / -", f"{win_days} / {zero_days} / {loss_days}"),
                ], style={"display": "grid", "gridTemplateColumns": "repeat(3,minmax(160px,1fr))", "gap": "10px"})

        return cells, stats


# ---- Local helper for stats card (flat, matches calendar)
def _stat_row(label: str, value: str) -> html.Div:
    return html.Div([
        html.Div(label, style={"fontSize":"12px", "color":"#6b7280"}),
        html.Div(value, style={"fontWeight":"800", "color":"#0f172a"})
    ], style={"border":"1px solid rgba(15,23,42,0.08)", "padding":"10px", "background":"#fff"})
