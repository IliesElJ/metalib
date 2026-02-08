"""
Callbacks Module
Handles all Dash callbacks for the MetaDAsh application
"""

from dash import Input, Output, State, html, no_update, callback_context, ALL, MATCH
import dash_bootstrap_components as dbc
from datetime import datetime, date
import plotly.graph_objects as go
import pandas as pd

from utils import (
    initialize_mt5,
    get_historical_data,
    process_deals_data,
    get_account_info,
    strategy_metrics,
    calculate_hourly_performance,
    get_candles_for_trade,
)
from utils.log_utils import (
    get_dates_for_strategy,
    read_log_file,
    get_log_statistics,
)
from utils.pm2_utils import (
    get_pm2_status,
    pm2_start,
    pm2_stop,
    pm2_restart,
)

from components import (
    render_overview_tab,
    render_detailed_tab,
    create_detailed_metrics_figure,
    render_pnl_tab,
    render_trades_tab,
    create_trades_table,
    render_raw_tab,
    render_strategy_type_tab,
    render_log_tab,
    create_log_stats_display,
    format_log_content,
    render_status_tab,
    create_status_summary,
    create_status_table,
    create_pm2_process_table,
    render_welcome_tab,
    render_instance_trades_tab,
    get_filtered_strategy_instances,
    get_dates_for_instance,
    create_instance_trades_grid,
    create_instance_trades_stats,
    create_trade_candlestick_chart,
    render_calibration_tab,
    create_results_table,
    create_results_chart,
    DEFAULT_STRATEGY_PARAMS,
)
from components.log_tab import get_filtered_instances
from components.detailed_tab import create_hourly_chart
from utils.health_utils import get_all_strategy_statuses, get_health_summary


# Default configuration
DEFAULT_START_DATE = date(2025, 1, 1)
DEFAULT_ACCOUNT_SIZE = 100000


# Global storage for data
stored_data = {
    "history_orders": None,
    "history_deals": None,
    "merged_deals": None,
    "account_size": DEFAULT_ACCOUNT_SIZE,
}


def register_callbacks(app):
    """
    Register all callbacks for the application
    """

    # ------------------------------
    # Auto-startup data loading
    # ------------------------------

    @app.callback(
        [
            Output("data-store", "data"),
            Output("connection-status", "children"),
            Output("account-info-store", "data"),
        ],
        [Input("startup-trigger", "n_intervals")],
        prevent_initial_call=False,
    )
    def auto_load_data(n_intervals):
        """Automatically connect to MT5 and load data on startup"""
        if n_intervals == 0:
            # First call - show loading status
            return (
                None,
                dbc.Alert(
                    "Connecting to MT5...",
                    color="info",
                    className="status-alert",
                    dismissable=True,
                ),
                None,
            )

        # Connect to MT5
        success, message = initialize_mt5()
        if not success:
            return (
                None,
                dbc.Alert(
                    f"MT5 Connection Failed: {message}",
                    color="danger",
                    className="status-alert",
                    dismissable=True,
                ),
                None,
            )

        # Fetch data with defaults
        from_date = datetime.combine(DEFAULT_START_DATE, datetime.min.time())
        to_date = datetime.now().replace(hour=23, minute=59, second=59)

        history_orders, history_deals, error = get_historical_data(from_date, to_date)

        if error:
            return (
                None,
                dbc.Alert(
                    f"Data Fetch Error: {error}",
                    color="danger",
                    className="status-alert",
                    dismissable=True,
                ),
                None,
            )

        if history_orders is None or history_deals is None:
            return (
                None,
                dbc.Alert(
                    "Failed to retrieve trading data",
                    color="danger",
                    className="status-alert",
                    dismissable=True,
                ),
                None,
            )

        # Process deals data
        merged_deals = process_deals_data(history_deals)

        if merged_deals is None or merged_deals.empty:
            return (
                None,
                dbc.Alert(
                    "No valid trades found in the specified period",
                    color="warning",
                    className="status-alert",
                    dismissable=True,
                ),
                None,
            )

        # Get account info
        account_info = get_account_info()

        # Store data globally
        stored_data["history_orders"] = history_orders
        stored_data["history_deals"] = history_deals
        stored_data["merged_deals"] = merged_deals
        stored_data["account_size"] = DEFAULT_ACCOUNT_SIZE

        # Success message (will auto-dismiss)
        success_message = dbc.Alert(
            f"Loaded {len(history_orders)} orders, {len(history_deals)} deals",
            color="success",
            className="status-alert",
            dismissable=True,
            duration=4000,  # Auto-dismiss after 4 seconds
        )

        return (
            {"data_available": True},
            success_message,
            account_info,
        )

    @app.callback(
        Output("tab-content", "children"),
        [
            Input("tabs", "active_tab"),
            Input("data-store", "data"),
            Input("account-info-store", "data"),
        ],
    )
    def render_tab_content(active_tab, data, account_info):
        """Render content based on selected tab"""
        # Welcome, Status, and Log tabs don't require MT5 data
        if active_tab == "welcome":
            return render_welcome_tab()

        if active_tab == "status":
            return render_status_tab()

        if active_tab == "logs":
            return render_log_tab()

        if active_tab == "calibration":
            return render_calibration_tab()

        if data is None or not data.get("data_available", False):
            return html.Div(
                [
                    dbc.Spinner(
                        color="primary",
                        size="lg",
                        spinner_style={"width": "3rem", "height": "3rem"},
                    ),
                    html.P(
                        "Loading trading data...",
                        className="text-center text-muted mt-3",
                        style={"fontSize": "14px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "minHeight": "300px",
                },
            )

        merged_deals = stored_data["merged_deals"]
        account_size = stored_data["account_size"]

        if not account_info:
            account_info = {"balance": 0, "equity": 0, "margin": 0}

        if active_tab == "overview":
            return render_overview_tab(merged_deals, account_size, account_info)
        elif active_tab == "strategy_types":
            return render_strategy_type_tab(merged_deals, account_size)
        elif active_tab == "detailed":
            return render_detailed_tab(merged_deals, account_size)
        elif active_tab == "pnl":
            return render_pnl_tab(merged_deals, account_size)
        elif active_tab == "trades":
            return render_trades_tab(merged_deals)
        elif active_tab == "instance_trades":
            return render_instance_trades_tab(merged_deals)
        elif active_tab == "raw":
            return render_raw_tab(merged_deals, stored_data["history_orders"])

        return html.Div()

    @app.callback(
        Output("detailed-metrics-graph", "figure"),
        [Input("metrics-dropdown", "value")],
        [State("data-store", "data")],
    )
    def update_detailed_metrics(selected_metrics, data):
        """Update detailed metrics chart based on selection"""
        if not data or not selected_metrics:
            return go.Figure()

        merged_deals = stored_data["merged_deals"]
        account_size = stored_data["account_size"]

        if merged_deals is None:
            return go.Figure()

        # Calculate metrics
        strategy_metrics_df = merged_deals[
            ["profit_open", "profit_close", "comment_open", "symbol_open", "time_open"]
        ].copy()

        grouped_metrics = strategy_metrics_df.groupby(
            ["comment_open", "symbol_open"]
        ).apply(lambda x: strategy_metrics(x, account_size))

        return create_detailed_metrics_figure(grouped_metrics, selected_metrics)

    @app.callback(
        [
            Output("hourly-graph", "figure"),
            Output("hourly-stats", "children"),
        ],
        [Input("strategy-dropdown", "value")],
        [State("data-store", "data")],
    )
    def update_hourly_graph(selected_strategy, data):
        """Update hourly performance graph and stats"""
        if not data or not selected_strategy:
            return go.Figure(), html.Div()

        merged_deals = stored_data["merged_deals"]

        if merged_deals is None:
            return go.Figure(), html.Div()

        # Use the new create_hourly_chart function
        fig, stats = create_hourly_chart(merged_deals, selected_strategy)

        return fig, stats

    @app.callback(
        Output("trades-table-container", "children"),
        [Input("bot-dropdown", "value")],
        [State("data-store", "data")],
    )
    def update_trades_table(selected_bot, data):
        """Update trades table based on selected bot"""
        if not data or not selected_bot:
            return html.Div(
                "Please select a bot/strategy", className="text-center text-muted"
            )

        merged_deals = stored_data["merged_deals"]

        if merged_deals is None:
            return html.Div("No data available", className="text-center text-muted")

        return create_trades_table(merged_deals, selected_bot)

    # ------------------------------
    # Log Tab callbacks
    # ------------------------------

    @app.callback(
        Output("log-strategy-dropdown", "options"),
        Output("log-strategy-dropdown", "value"),
        Input("log-strategy-type-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_log_strategy_instances(strategy_type):
        """Update strategy instances when type filter changes"""
        instances = get_filtered_instances(strategy_type)

        def format_label(s):
            parts = s.split("_")
            if len(parts) >= 2:
                strategy_t = parts[0].upper()
                instance_name = "_".join(parts[1:])
                return f"{strategy_t} - {instance_name}"
            return s

        options = [{"label": format_label(s), "value": s} for s in instances]
        default_value = instances[0] if instances else None

        return options, default_value

    @app.callback(
        Output("log-date-dropdown", "options"),
        Output("log-date-dropdown", "value"),
        Input("log-strategy-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_log_dates(strategy_instance):
        """Update available dates when strategy instance changes"""
        if not strategy_instance:
            return [], None

        dates = get_dates_for_strategy(strategy_instance)

        def format_date(date_str):
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                return dt.strftime("%B %d, %Y (%A)")
            except ValueError:
                return date_str

        def get_default_business_day(date_list):
            """Get the most recent business day from available dates."""
            for d in date_list:
                try:
                    dt = datetime.strptime(d, "%Y-%m-%d")
                    if dt.weekday() < 5:  # Mon-Fri
                        return d
                except ValueError:
                    continue
            return date_list[0] if date_list else None

        options = [{"label": format_date(d), "value": d} for d in dates]
        default_value = get_default_business_day(dates)

        return options, default_value

    @app.callback(
        Output("log-stats-container", "children"),
        Output("log-content-display", "children"),
        Output("log-filename-display", "children"),
        Input("log-strategy-dropdown", "value"),
        Input("log-date-dropdown", "value"),
        Input("log-refresh-btn", "n_clicks"),
        prevent_initial_call=False,
    )
    def update_log_display(strategy_instance, date_str, refresh_clicks):
        """Update log content and statistics display"""
        if not strategy_instance or not date_str:
            empty_stats = create_log_stats_display({
                "total_lines": 0,
                "timestamp_markers": 0,
                "errors": 0,
                "warnings": 0
            })
            empty_content = html.Div(
                "Please select a strategy instance and date to view logs.",
                style={"color": "#888", "textAlign": "center", "padding": "40px"}
            )
            return empty_stats, empty_content, ""

        # Read log file
        log_content = read_log_file(strategy_instance, date_str)
        filename = f"output_{strategy_instance}_{date_str}.log"

        if log_content is None:
            empty_stats = create_log_stats_display({
                "total_lines": 0,
                "timestamp_markers": 0,
                "errors": 0,
                "warnings": 0
            })
            error_content = html.Div(
                f"Log file not found for {strategy_instance} on {date_str}",
                style={"color": "#f48771", "textAlign": "center", "padding": "40px"}
            )
            return empty_stats, error_content, filename

        # Get statistics
        stats = get_log_statistics(log_content)
        stats_display = create_log_stats_display(stats)

        # Format content
        formatted_content = format_log_content(log_content)

        return stats_display, formatted_content, filename

    @app.callback(
        Output("download-log-file", "data"),
        Input("log-download-btn", "n_clicks"),
        State("log-strategy-dropdown", "value"),
        State("log-date-dropdown", "value"),
        prevent_initial_call=True,
    )
    def download_log_file(n_clicks, strategy_instance, date_str):
        """Handle log file download"""
        if not n_clicks or not strategy_instance or not date_str:
            return None

        log_content = read_log_file(strategy_instance, date_str)
        if log_content is None:
            return None

        filename = f"output_{strategy_instance}_{date_str}.log"
        return dict(content=log_content, filename=filename)

    # ------------------------------
    # Status Monitoring callbacks
    # ------------------------------

    @app.callback(
        Output("status-summary-container", "children"),
        Output("status-table-container", "children"),
        Output("status-last-refresh", "children"),
        Input("status-refresh-btn", "n_clicks"),
        Input("status-auto-refresh", "n_intervals"),
        prevent_initial_call=False,
    )
    def update_status_display(n_clicks, n_intervals):
        """Update status summary and table"""
        # Get current status
        summary = get_health_summary()
        statuses = get_all_strategy_statuses()

        # Create components
        summary_component = create_status_summary(summary)
        table_component = create_status_table(statuses)

        # Last refresh time
        last_refresh = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"

        return summary_component, table_component, last_refresh

    # ------------------------------
    # Instance Trades Tab callbacks
    # ------------------------------

    @app.callback(
        Output("instance-trades-strategy-dropdown", "options"),
        Output("instance-trades-strategy-dropdown", "value"),
        Input("instance-trades-type-dropdown", "value"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def update_instance_trades_strategy_options(strategy_type, data):
        """Update strategy instances when type filter changes"""
        if not data or not data.get("data_available"):
            return [], None

        merged_deals = stored_data["merged_deals"]
        instances = get_filtered_strategy_instances(strategy_type, merged_deals)

        def format_label(s):
            parts = s.split("_")
            if len(parts) >= 2:
                strategy_t = parts[0].upper()
                instance_name = "_".join(parts[1:])
                return f"{strategy_t} - {instance_name}"
            return s

        options = [{"label": format_label(s), "value": s} for s in instances]
        default_value = instances[0] if instances else None

        return options, default_value

    @app.callback(
        Output("instance-trades-date-dropdown", "options"),
        Output("instance-trades-date-dropdown", "value"),
        Input("instance-trades-strategy-dropdown", "value"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def update_instance_trades_date_options(strategy_instance, data):
        """Update available dates when strategy instance changes"""
        if not strategy_instance or not data or not data.get("data_available"):
            return [], None

        merged_deals = stored_data["merged_deals"]
        dates = get_dates_for_instance(strategy_instance, merged_deals)

        def format_date(date_str):
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                return dt.strftime("%B %d, %Y (%A)")
            except ValueError:
                return date_str

        options = [{"label": format_date(d), "value": d} for d in dates]
        default_value = dates[0] if dates else None

        return options, default_value

    @app.callback(
        Output("instance-trades-grid-container", "children"),
        Output("instance-trades-stats-container", "children"),
        Input("instance-trades-strategy-dropdown", "value"),
        Input("instance-trades-date-dropdown", "value"),
        State("data-store", "data"),
        prevent_initial_call=False,
    )
    def update_instance_trades_grid(strategy_instance, date_str, data):
        """Update AG Grid with trades for selected instance and date"""
        if not strategy_instance or not date_str:
            empty_msg = html.Div(
                "Please select a strategy instance and date to view trades.",
                style={"color": "#64748b", "textAlign": "center", "padding": "40px"},
            )
            return empty_msg, html.Div()

        if not data or not data.get("data_available"):
            return html.Div("Loading data...", style={"color": "#64748b", "textAlign": "center", "padding": "40px"}), html.Div()

        merged_deals = stored_data["merged_deals"]
        if merged_deals is None or merged_deals.empty:
            return html.Div("No trade data available.", style={"color": "#64748b", "textAlign": "center", "padding": "40px"}), html.Div()

        # Filter trades by instance and date
        instance_deals = merged_deals[merged_deals["comment_open"] == strategy_instance].copy()
        instance_deals["trade_date"] = instance_deals["time_open"].dt.strftime("%Y-%m-%d")
        filtered_deals = instance_deals[instance_deals["trade_date"] == date_str]

        if filtered_deals.empty:
            return html.Div(
                f"No trades found for {strategy_instance} on {date_str}.",
                style={"color": "#64748b", "textAlign": "center", "padding": "40px"},
            ), html.Div()

        # Create grid and stats
        grid = create_instance_trades_grid(filtered_deals)
        stats = create_instance_trades_stats(filtered_deals)

        return grid, stats

    @app.callback(
        Output("trade-chart-container", "children"),
        Output("trade-chart-container", "style"),
        Input("instance-trades-grid", "selectedRows"),
        State("instance-trades-strategy-dropdown", "value"),
        State("instance-trades-date-dropdown", "value"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def show_trade_chart(selected_rows, strategy_instance, date_str, data):
        """Show candlestick chart when a trade row is selected"""
        if not selected_rows or not strategy_instance or not date_str:
            return html.Div(), {"display": "none"}

        if not data or not data.get("data_available"):
            return html.Div(), {"display": "none"}

        # Get the first selected row
        row_data = selected_rows[0] if selected_rows else {}
        if not row_data:
            return html.Div(), {"display": "none"}

        # Extract trade info
        symbol = row_data.get("symbol_open", "")
        time_open_iso = row_data.get("time_open_iso")
        time_close_iso = row_data.get("time_close_iso")
        price_open = row_data.get("price_open", 0)
        price_close = row_data.get("price_close", 0)
        total_profit = row_data.get("total_profit", 0)

        if not time_open_iso:
            return html.Div("Could not retrieve trade time.", style={"color": "#ef4444"}), {
                "display": "block",
                "backgroundColor": "white",
                "padding": "24px",
                "borderRadius": "12px",
                "border": "1px solid #e2e8f0",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
            }

        # Parse times
        time_open = pd.to_datetime(time_open_iso)
        time_close = pd.to_datetime(time_close_iso) if time_close_iso else None

        # Fetch candle data from MT5
        candles_df = get_candles_for_trade(symbol, time_open, time_close, buffer_minutes=30)

        # Prepare trade data for chart
        trade_data = {
            "symbol": symbol,
            "time_open": time_open,
            "time_close": time_close,
            "price_open": price_open,
            "price_close": price_close,
            "total_profit": total_profit,
        }

        # Create chart
        from dash import dcc
        fig = create_trade_candlestick_chart(candles_df, trade_data)

        chart_container = html.Div([
            html.Div(
                [
                    html.Span(
                        "Trade Chart",
                        style={
                            "fontWeight": "600",
                            "fontSize": "16px",
                            "color": "#1e293b",
                        },
                    ),
                    html.Span(
                        f" - {symbol}",
                        style={
                            "color": "#64748b",
                            "fontSize": "14px",
                        },
                    ),
                ],
                style={"marginBottom": "16px"},
            ),
            dcc.Graph(figure=fig, config={"displayModeBar": True, "scrollZoom": True}),
        ])

        visible_style = {
            "display": "block",
            "backgroundColor": "white",
            "padding": "24px",
            "borderRadius": "12px",
            "border": "1px solid #e2e8f0",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
        }

        return chart_container, visible_style

    # ------------------------------
    # PM2 Process Manager callbacks
    # ------------------------------

    @app.callback(
        Output("pm2-process-table-container", "children"),
        Input("status-auto-refresh", "n_intervals"),
        Input("status-refresh-btn", "n_clicks"),
        Input("pm2-start-all-btn", "n_clicks"),
        Input("pm2-stop-all-btn", "n_clicks"),
        Input("pm2-restart-all-btn", "n_clicks"),
        prevent_initial_call=False,
    )
    def update_pm2_process_table(n_intervals, refresh_clicks, start_clicks, stop_clicks, restart_clicks):
        """Update PM2 process table"""
        processes = get_pm2_status()
        return create_pm2_process_table(processes)

    @app.callback(
        Output("pm2-action-feedback", "children"),
        Input("pm2-start-all-btn", "n_clicks"),
        Input("pm2-stop-all-btn", "n_clicks"),
        Input("pm2-restart-all-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def handle_pm2_bulk_actions(start_clicks, stop_clicks, restart_clicks):
        """Handle PM2 bulk action buttons"""
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "pm2-start-all-btn":
            result = pm2_start()
        elif button_id == "pm2-stop-all-btn":
            result = pm2_stop()
        elif button_id == "pm2-restart-all-btn":
            result = pm2_restart()
        else:
            return no_update

        if result["success"]:
            return dbc.Alert(
                result["message"],
                color="success",
                dismissable=True,
                duration=3000,
            )
        else:
            return dbc.Alert(
                f"Error: {result['message']}",
                color="danger",
                dismissable=True,
                duration=5000,
            )

    # ------------------------------
    # Overview Tab - Strategy Collapse Toggle
    # ------------------------------

    @app.callback(
        Output({"type": "strategy-collapse", "index": MATCH}, "is_open"),
        Input({"type": "strategy-header", "index": MATCH}, "n_clicks"),
        State({"type": "strategy-collapse", "index": MATCH}, "is_open"),
        prevent_initial_call=True,
    )
    def toggle_strategy_collapse(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("pm2-action-feedback", "children", allow_duplicate=True),
        Input({"type": "pm2-action-btn", "index": ALL, "action": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def handle_pm2_individual_actions(n_clicks_list):
        """Handle individual PM2 process action buttons"""
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list):
            return no_update

        # Get the triggered button info
        triggered = ctx.triggered[0]
        prop_id = triggered["prop_id"]

        # Parse the pattern-matching ID
        import json
        # prop_id looks like: '{"type":"pm2-action-btn","index":"metafvg","action":"restart"}.n_clicks'
        id_str = prop_id.rsplit(".", 1)[0]
        try:
            button_info = json.loads(id_str)
        except json.JSONDecodeError:
            return no_update

        process_name = button_info.get("index")
        action = button_info.get("action")

        if not process_name or not action:
            return no_update

        if action == "start":
            result = pm2_start(process_name)
        elif action == "stop":
            result = pm2_stop(process_name)
        elif action == "restart":
            result = pm2_restart(process_name)
        else:
            return no_update

        if result["success"]:
            return dbc.Alert(
                f"{process_name}: {result['message']}",
                color="success",
                dismissable=True,
                duration=3000,
            )
        else:
            return dbc.Alert(
                f"{process_name}: {result['message']}",
                color="danger",
                dismissable=True,
                duration=5000,
            )

    # ------------------------------
    # Weight Calibration Tab callbacks
    # ------------------------------

    @app.callback(
        Output({"type": "calib-weight-display", "index": ALL}, "children"),
        Input({"type": "calib-numerator", "index": ALL}, "value"),
        Input({"type": "calib-trades-per-day", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def update_weight_displays(numerators, trades_per_days):
        """Update computed weight displays when inputs change."""
        import numpy as np
        weights = []
        for num, tpd in zip(numerators, trades_per_days):
            if num is not None and tpd is not None and tpd > 0:
                weight = num / np.sqrt(tpd)
                weights.append(f"{weight:.3f}")
            else:
                weights.append("--")
        return weights

    @app.callback(
        Output("calib-results-store", "data"),
        Output("calib-results-table-container", "children"),
        Output("calib-results-chart-container", "children"),
        Output("calib-save-section", "style"),
        Output("calib-save-btn", "disabled"),
        Output("calib-status-msg", "children"),
        Input("calib-run-btn", "n_clicks"),
        State({"type": "calib-strategy-enabled", "index": ALL}, "value"),
        State({"type": "calib-strategy-enabled", "index": ALL}, "id"),
        State({"type": "calib-numerator", "index": ALL}, "value"),
        State({"type": "calib-trades-per-day", "index": ALL}, "value"),
        State("calib-risk-pct", "value"),
        State("calib-lookback-days", "value"),
        State("calib-config-dir", "value"),
        prevent_initial_call=True,
    )
    def run_calibration_optimization(
        n_clicks, enabled_list, enabled_ids, numerators, trades_per_days,
        risk_pct, lookback_days, config_dir
    ):
        """Run the MetaScale optimization and display results."""
        import numpy as np

        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update, no_update

        try:
            # Build strategy params dict from UI inputs
            # Format: {strategy_type: {"numerator": N, "trades_per_day": T}}
            strategy_params = {}
            for enabled, id_obj, num, tpd in zip(enabled_list, enabled_ids, numerators, trades_per_days):
                strategy_key = id_obj["index"]
                if enabled and num is not None and tpd is not None and tpd > 0:
                    strategy_params[strategy_key] = {
                        "numerator": num,
                        "trades_per_day": tpd,
                    }

            if not strategy_params:
                return (
                    None,
                    html.Div("No strategies enabled", className="text-warning text-center p-4"),
                    html.Div(),
                    {"display": "none"},
                    True,
                    dbc.Alert("Please enable at least one strategy", color="warning"),
                )

            # Run optimization using MetaScale
            from utils.calibration_utils import run_metascale_optimization

            result = run_metascale_optimization(
                strategy_params=strategy_params,
                risk_pct=risk_pct / 100.0,  # Convert from percentage
                lookback_days=lookback_days,
                config_dir=config_dir,
            )

            if result["success"]:
                weights_df = result["weights_df"]

                # Create results display
                table = create_results_table(weights_df)
                chart = create_results_chart(weights_df)

                return (
                    weights_df.to_dict("records"),
                    table,
                    chart,
                    {"display": "block"},
                    False,
                    dbc.Alert(
                        f"Optimization completed. {len(weights_df)} positions computed.",
                        color="success",
                        duration=4000,
                    ),
                )
            else:
                return (
                    None,
                    html.Div(f"Optimization failed: {result['error']}", className="text-danger text-center p-4"),
                    html.Div(),
                    {"display": "none"},
                    True,
                    dbc.Alert(f"Error: {result['error']}", color="danger"),
                )

        except Exception as e:
            return (
                None,
                html.Div(f"Error: {str(e)}", className="text-danger text-center p-4"),
                html.Div(),
                {"display": "none"},
                True,
                dbc.Alert(f"Error: {str(e)}", color="danger"),
            )

    @app.callback(
        Output("calib-save-feedback", "children"),
        Input("calib-save-btn", "n_clicks"),
        State("calib-results-store", "data"),
        State("calib-config-dir", "value"),
        prevent_initial_call=True,
    )
    def save_calibration_results(n_clicks, results_data, config_dir):
        """Save optimization results to YAML config files."""
        if not n_clicks or not results_data:
            return no_update

        try:
            from utils.calibration_utils import save_weights_to_yaml

            weights_df = pd.DataFrame(results_data)
            result = save_weights_to_yaml(weights_df, config_dir)

            if result["success"]:
                return dbc.Alert(
                    f"Saved to {result['files_updated']} config files",
                    color="success",
                    duration=4000,
                )
            else:
                return dbc.Alert(
                    f"Save failed: {result['error']}",
                    color="danger",
                    duration=5000,
                )

        except Exception as e:
            return dbc.Alert(
                f"Error saving: {str(e)}",
                color="danger",
                duration=5000,
            )
