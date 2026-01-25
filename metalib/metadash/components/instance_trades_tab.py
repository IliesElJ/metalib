"""
Instance Trades Tab Component
Displays trades for a selected strategy instance with AG Grid and candlestick charts
"""
from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.graph_objects as go
import pandas as pd


# Hardcoded strategy types (same as log_tab)
STRATEGY_TYPES = [
    {"value": "all", "label": "All Strategies"},
    {"value": "metago", "label": "MetaGO"},
    {"value": "metaob", "label": "MetaOB"},
    {"value": "metafvg", "label": "MetaFVG"},
    {"value": "metane", "label": "MetaNE"},
    {"value": "metaga", "label": "MetaGA"},
]


def get_filtered_strategy_instances(strategy_type, merged_deals):
    """
    Get strategy instances filtered by type from merged_deals.

    Args:
        strategy_type: Strategy type to filter by (e.g., 'metago', 'metafvg') or 'all'
        merged_deals: DataFrame with trade data

    Returns:
        List of filtered strategy instance names
    """
    if merged_deals is None or merged_deals.empty:
        return []

    all_instances = sorted(merged_deals["comment_open"].unique().tolist())

    if strategy_type == "all" or not strategy_type:
        return all_instances

    return [inst for inst in all_instances if inst.startswith(strategy_type)]


def get_dates_for_instance(strategy_instance, merged_deals):
    """
    Get available trade dates for a strategy instance.

    Args:
        strategy_instance: The strategy instance name
        merged_deals: DataFrame with trade data

    Returns:
        List of date strings in YYYY-MM-DD format, sorted newest first
    """
    if merged_deals is None or merged_deals.empty or not strategy_instance:
        return []

    instance_deals = merged_deals[merged_deals["comment_open"] == strategy_instance]
    if instance_deals.empty:
        return []

    # Extract unique dates from time_open
    dates = instance_deals["time_open"].dt.date.unique()
    # Sort descending (newest first) and format as strings
    sorted_dates = sorted(dates, reverse=True)
    return [d.strftime("%Y-%m-%d") for d in sorted_dates]


def _format_strategy_label(strategy_instance):
    """Format strategy instance name for display."""
    parts = strategy_instance.split("_")
    if len(parts) >= 2:
        strategy_type = parts[0].upper()
        instance_name = "_".join(parts[1:])
        return f"{strategy_type} - {instance_name}"
    return strategy_instance


def _format_date_label(date_str):
    """Format date string for display."""
    from datetime import datetime
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%B %d, %Y (%A)")
    except ValueError:
        return date_str


def render_instance_trades_tab(merged_deals=None):
    """
    Render the instance trades tab with cascading dropdowns and AG Grid.

    Args:
        merged_deals: Optional DataFrame with trade data for initial load

    Returns:
        Dash HTML component with instance trades interface
    """
    # Get initial values
    default_type = "all"
    strategy_instances = get_filtered_strategy_instances(default_type, merged_deals) if merged_deals is not None else []
    default_strategy = strategy_instances[0] if strategy_instances else None
    default_dates = get_dates_for_instance(default_strategy, merged_deals) if default_strategy and merged_deals is not None else []
    default_date = default_dates[0] if default_dates else None

    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "Instance Trades",
                                style={
                                    "fontSize": "24px",
                                    "fontWeight": "700",
                                    "color": "#1e293b",
                                },
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "gap": "10px"},
                    ),
                    html.P(
                        "View trades by strategy instance with interactive charts",
                        style={
                            "color": "#64748b",
                            "marginTop": "8px",
                            "marginBottom": "0",
                            "fontSize": "14px",
                        },
                    ),
                ],
                style={
                    "marginBottom": "24px",
                    "paddingBottom": "16px",
                    "borderBottom": "1px solid #e2e8f0",
                },
            ),
            # Selectors Card
            html.Div(
                [
                    dbc.Row(
                        [
                            # Strategy Type Filter
                            dbc.Col(
                                [
                                    html.Label(
                                        "Strategy Type",
                                        style={
                                            "fontWeight": "600",
                                            "fontSize": "13px",
                                            "color": "#475569",
                                            "marginBottom": "8px",
                                            "display": "block",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="instance-trades-type-dropdown",
                                        options=[
                                            {"label": st["label"], "value": st["value"]}
                                            for st in STRATEGY_TYPES
                                        ],
                                        value=default_type,
                                        placeholder="Filter by type...",
                                        style={"fontSize": "14px"},
                                        clearable=False,
                                    ),
                                ],
                                lg=3,
                                md=4,
                                sm=12,
                                className="mb-3 mb-lg-0",
                            ),
                            # Strategy Instance
                            dbc.Col(
                                [
                                    html.Label(
                                        "Strategy Instance",
                                        style={
                                            "fontWeight": "600",
                                            "fontSize": "13px",
                                            "color": "#475569",
                                            "marginBottom": "8px",
                                            "display": "block",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="instance-trades-strategy-dropdown",
                                        options=[
                                            {"label": _format_strategy_label(s), "value": s}
                                            for s in strategy_instances
                                        ],
                                        value=default_strategy,
                                        placeholder="Select a strategy instance...",
                                        style={"fontSize": "14px"},
                                        clearable=False,
                                    ),
                                ],
                                lg=5,
                                md=4,
                                sm=12,
                                className="mb-3 mb-lg-0",
                            ),
                            # Date
                            dbc.Col(
                                [
                                    html.Label(
                                        "Trade Date",
                                        style={
                                            "fontWeight": "600",
                                            "fontSize": "13px",
                                            "color": "#475569",
                                            "marginBottom": "8px",
                                            "display": "block",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="instance-trades-date-dropdown",
                                        options=[
                                            {"label": _format_date_label(d), "value": d}
                                            for d in default_dates
                                        ],
                                        value=default_date,
                                        placeholder="Select a date...",
                                        style={"fontSize": "14px"},
                                        clearable=False,
                                    ),
                                ],
                                lg=4,
                                md=4,
                                sm=12,
                            ),
                        ],
                    ),
                ],
                style={
                    "backgroundColor": "#f8fafc",
                    "padding": "20px",
                    "borderRadius": "12px",
                    "marginBottom": "20px",
                    "border": "1px solid #e2e8f0",
                },
            ),
            # Stats container
            html.Div(id="instance-trades-stats-container", className="mb-4"),
            # AG Grid container
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "Trades",
                                style={
                                    "fontWeight": "600",
                                    "fontSize": "16px",
                                    "color": "#1e293b",
                                },
                            ),
                            html.Span(
                                " (click a row to view chart)",
                                style={
                                    "color": "#94a3b8",
                                    "fontSize": "13px",
                                },
                            ),
                        ],
                        style={"marginBottom": "16px"},
                    ),
                    html.Div(
                        id="instance-trades-grid-container",
                        style={"minHeight": "300px"},
                    ),
                ],
                style={
                    "backgroundColor": "white",
                    "padding": "24px",
                    "borderRadius": "12px",
                    "border": "1px solid #e2e8f0",
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                    "marginBottom": "20px",
                },
            ),
            # Chart container
            html.Div(
                id="trade-chart-container",
                style={
                    "backgroundColor": "white",
                    "padding": "24px",
                    "borderRadius": "12px",
                    "border": "1px solid #e2e8f0",
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                    "display": "none",  # Hidden by default
                },
            ),
            # Hidden store for selected trade data
            dcc.Store(id="instance-trades-selected-row"),
        ],
        style={"padding": "0"},
    )


def create_instance_trades_grid(trades_df):
    """
    Create AG Grid component for displaying trades.

    Args:
        trades_df: DataFrame with filtered trades

    Returns:
        AG Grid component
    """
    if trades_df is None or trades_df.empty:
        return html.Div(
            "No trades found for the selected criteria.",
            style={
                "color": "#64748b",
                "textAlign": "center",
                "padding": "40px",
            },
        )

    # Prepare data for display
    display_df = trades_df.copy()
    display_df["total_profit"] = display_df["profit_open"] + display_df["profit_close"]

    # Format datetime columns
    display_df["time_open_str"] = display_df["time_open"].dt.strftime("%H:%M:%S")
    if "time_close" in display_df.columns:
        display_df["time_close_str"] = display_df["time_close"].dt.strftime("%H:%M:%S")
    else:
        display_df["time_close_str"] = ""

    # Calculate trade duration
    if "time_close" in display_df.columns:
        display_df["duration"] = (display_df["time_close"] - display_df["time_open"]).dt.total_seconds() / 60
        display_df["duration_str"] = display_df["duration"].apply(
            lambda x: f"{int(x)}m" if pd.notna(x) else ""
        )
    else:
        display_df["duration_str"] = ""

    # Select columns for display
    columns_to_show = [
        "symbol_open", "time_open_str", "time_close_str", "duration_str",
        "price_open", "price_close", "volume_open", "total_profit"
    ]
    available_cols = [col for col in columns_to_show if col in display_df.columns]

    # Column definitions
    column_defs = [
        {"field": "symbol_open", "headerName": "Symbol", "width": 100},
        {"field": "time_open_str", "headerName": "Open Time", "width": 110},
        {"field": "time_close_str", "headerName": "Close Time", "width": 110},
        {"field": "duration_str", "headerName": "Duration", "width": 90},
        {
            "field": "price_open",
            "headerName": "Open Price",
            "width": 120,
            "valueFormatter": {"function": "d3.format(',.5f')(params.value)"},
        },
        {
            "field": "price_close",
            "headerName": "Close Price",
            "width": 120,
            "valueFormatter": {"function": "d3.format(',.5f')(params.value)"},
        },
        {
            "field": "volume_open",
            "headerName": "Volume",
            "width": 90,
            "valueFormatter": {"function": "d3.format(',.2f')(params.value)"},
        },
        {
            "field": "total_profit",
            "headerName": "Profit",
            "width": 100,
            "valueFormatter": {"function": "d3.format('$,.2f')(params.value)"},
            "cellStyle": {
                "styleConditions": [
                    {"condition": "params.value > 0", "style": {"color": "#22c55e", "fontWeight": "600"}},
                    {"condition": "params.value < 0", "style": {"color": "#ef4444", "fontWeight": "600"}},
                ]
            },
        },
    ]

    # Filter column defs to only include available columns
    column_defs = [cd for cd in column_defs if cd["field"] in available_cols]

    # Prepare row data - include hidden columns for chart callback
    row_data = display_df.to_dict("records")

    # Add original timestamps as ISO strings for the callback
    for i, row in enumerate(row_data):
        row["time_open_iso"] = trades_df.iloc[i]["time_open"].isoformat()
        if "time_close" in trades_df.columns and pd.notna(trades_df.iloc[i]["time_close"]):
            row["time_close_iso"] = trades_df.iloc[i]["time_close"].isoformat()
        else:
            row["time_close_iso"] = None
        row["row_index"] = i

    return dag.AgGrid(
        id="instance-trades-grid",
        rowData=row_data,
        columnDefs=column_defs,
        defaultColDef={
            "sortable": True,
            "filter": True,
            "resizable": True,
        },
        dashGridOptions={
            "rowSelection": "single",
            "animateRows": True,
        },
        style={"height": "400px", "width": "100%"},
        className="ag-theme-alpine",
    )


def create_instance_trades_stats(trades_df):
    """
    Create statistics cards for the trades.

    Args:
        trades_df: DataFrame with filtered trades

    Returns:
        Dash component with stat cards
    """
    if trades_df is None or trades_df.empty:
        return html.Div()

    # Calculate stats
    trades_df = trades_df.copy()
    trades_df["total_profit"] = trades_df["profit_open"] + trades_df["profit_close"]

    total_trades = len(trades_df)
    total_profit = trades_df["total_profit"].sum()
    avg_profit = trades_df["total_profit"].mean()
    win_count = (trades_df["total_profit"] > 0).sum()
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    stat_configs = [
        {
            "title": "Total Trades",
            "value": str(total_trades),
            "icon": "#",
            "color": "#3b82f6",
            "bg": "#eff6ff",
        },
        {
            "title": "Total Profit",
            "value": f"${total_profit:,.2f}",
            "icon": "$",
            "color": "#22c55e" if total_profit >= 0 else "#ef4444",
            "bg": "#f0fdf4" if total_profit >= 0 else "#fef2f2",
        },
        {
            "title": "Avg Profit",
            "value": f"${avg_profit:,.2f}",
            "icon": "~",
            "color": "#06b6d4",
            "bg": "#ecfeff",
        },
        {
            "title": "Win Rate",
            "value": f"{win_rate:.1f}%",
            "icon": "%",
            "color": "#22c55e" if win_rate >= 50 else "#f59e0b",
            "bg": "#f0fdf4" if win_rate >= 50 else "#fffbeb",
        },
    ]

    return dbc.Row(
        [
            dbc.Col(
                _create_stat_card(cfg["title"], cfg["value"], cfg["icon"], cfg["color"], cfg["bg"]),
                lg=3,
                md=6,
                sm=6,
                xs=12,
                className="mb-3",
            )
            for cfg in stat_configs
        ],
    )


def _create_stat_card(title, value, icon, color, bg_color):
    """Create a statistic display card."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        icon,
                        style={
                            "width": "36px",
                            "height": "36px",
                            "borderRadius": "8px",
                            "backgroundColor": bg_color,
                            "color": color,
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "fontWeight": "700",
                            "fontSize": "16px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                title,
                                style={
                                    "fontSize": "12px",
                                    "color": "#64748b",
                                    "fontWeight": "500",
                                },
                            ),
                            html.Div(
                                value,
                                style={
                                    "fontSize": "20px",
                                    "fontWeight": "700",
                                    "color": "#1e293b",
                                    "lineHeight": "1.2",
                                },
                            ),
                        ],
                        style={"marginLeft": "12px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
        ],
        style={
            "padding": "16px",
            "borderRadius": "10px",
            "backgroundColor": "white",
            "border": "1px solid #e2e8f0",
            "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
        },
    )


def create_trade_candlestick_chart(candles_df, trade_data):
    """
    Create a candlestick chart with trade entry/exit markers.

    Args:
        candles_df: DataFrame with OHLC data (columns: time, open, high, low, close)
        trade_data: Dictionary with trade info (symbol, time_open, time_close, price_open, price_close, total_profit)

    Returns:
        Plotly figure with candlestick chart and markers
    """
    if candles_df is None or candles_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No candle data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#64748b"),
        )
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=400,
        )
        return fig

    # Create candlestick chart
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=candles_df["time"],
                open=candles_df["open"],
                high=candles_df["high"],
                low=candles_df["low"],
                close=candles_df["close"],
                name="Price",
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
            )
        ]
    )

    # Add entry marker
    time_open = pd.to_datetime(trade_data["time_open"])
    price_open = trade_data["price_open"]

    fig.add_trace(
        go.Scatter(
            x=[time_open],
            y=[price_open],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=15,
                color="#3b82f6",
                line=dict(color="white", width=2),
            ),
            name="Entry",
            hovertemplate=f"Entry<br>Time: {time_open}<br>Price: {price_open:.5f}<extra></extra>",
        )
    )

    # Add exit marker if available
    if trade_data.get("time_close"):
        time_close = pd.to_datetime(trade_data["time_close"])
        price_close = trade_data["price_close"]
        profit = trade_data.get("total_profit", 0)
        marker_color = "#22c55e" if profit >= 0 else "#ef4444"

        fig.add_trace(
            go.Scatter(
                x=[time_close],
                y=[price_close],
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=15,
                    color=marker_color,
                    line=dict(color="white", width=2),
                ),
                name="Exit",
                hovertemplate=f"Exit<br>Time: {time_close}<br>Price: {price_close:.5f}<br>Profit: ${profit:.2f}<extra></extra>",
            )
        )

        # Add connecting line
        fig.add_trace(
            go.Scatter(
                x=[time_open, time_close],
                y=[price_open, price_close],
                mode="lines",
                line=dict(color=marker_color, width=2, dash="dot"),
                name="Trade Path",
                showlegend=False,
            )
        )

    # Update layout
    symbol = trade_data.get("symbol", "")
    profit_str = f"${trade_data.get('total_profit', 0):,.2f}"
    profit_color = "#22c55e" if trade_data.get("total_profit", 0) >= 0 else "#ef4444"

    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> Trade Chart <span style='color:{profit_color}'>{profit_str}</span>",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Time",
            rangeslider=dict(visible=False),
            gridcolor="#f1f5f9",
        ),
        yaxis=dict(
            title="Price",
            gridcolor="#f1f5f9",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=450,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    return fig
