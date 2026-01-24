"""
Callbacks Module
Handles all Dash callbacks for the MetaDAsh application
"""

from dash import Input, Output, State, html
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
)
from utils.log_utils import (
    get_dates_for_strategy,
    read_log_file,
    get_log_statistics,
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
    render_welcome_tab,
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
