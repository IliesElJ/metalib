"""
Detailed Analysis Tab Component
"""
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from utils.metrics import strategy_metrics
from .common_ui import (
    create_page_header,
    create_section_card,
    style_plotly_chart,
    format_currency,
    COLORS,
    CHART_COLORS,
)


def render_detailed_tab(merged_deals, account_size):
    """
    Render the detailed analysis tab with modern styling
    """
    # Get unique strategies for dropdown
    strategies = merged_deals['comment_open'].unique()
    default_strategy = strategies[0] if len(strategies) > 0 else None

    # Calculate metrics
    strategy_metrics_df = merged_deals[
        ["profit_open", "profit_close", "comment_open", "symbol_open", "time_open"]
    ].copy()

    grouped_metrics = strategy_metrics_df.groupby(["comment_open", "symbol_open"]).apply(
        lambda x: strategy_metrics(x, account_size)
    )

    # Create duration analysis if time_close exists
    duration_section = None
    duration_stats = None
    if "time_close" in merged_deals.columns:
        duration_fig, duration_stats = create_duration_chart(merged_deals)
        duration_section = create_section_card(
            "Trade Duration Analysis",
            html.Div([
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(figure=duration_fig, config={"displayModeBar": False}),
                        lg=8, md=12,
                    ),
                    dbc.Col(
                        duration_stats,
                        lg=4, md=12,
                    ),
                ]),
            ]),
            subtitle="Distribution of trade holding times",
        )

    # Create hourly performance chart for default strategy
    hourly_fig = None
    hourly_stats = None
    if default_strategy:
        hourly_fig, hourly_stats = create_hourly_chart(merged_deals, default_strategy)

    return html.Div([
        # Header
        create_page_header(
            "Detailed Analysis",
            "Deep dive into strategy performance metrics and trading patterns"
        ),

        # Metrics Comparison Section
        create_section_card(
            "Metrics Comparison",
            html.Div([
                # Metric selector with chip-style appearance
                html.Div([
                    html.Label(
                        "Select Metrics to Compare:",
                        style={
                            "fontWeight": "600",
                            "fontSize": "13px",
                            "color": COLORS["text_medium"],
                            "marginBottom": "12px",
                            "display": "block",
                        },
                    ),
                    dcc.Dropdown(
                        id='metrics-dropdown',
                        options=[
                            {'label': 'Total Profit', 'value': 'Total Profit'},
                            {'label': 'Win Rate (%)', 'value': 'Win Rate (%)'},
                            {'label': 'Sharpe Ratio', 'value': 'Sharpe Ratio'},
                            {'label': 'Max Drawdown (%)', 'value': 'Max Drawdown (%)'},
                            {'label': 'Profit Factor', 'value': 'Profit Factor'},
                            {'label': 'Average Profit', 'value': 'Average Profit by Trade'},
                        ],
                        value=["Total Profit", "Win Rate (%)", "Sharpe Ratio"],
                        multi=True,
                        style={"fontSize": "14px"},
                        placeholder="Select metrics...",
                    ),
                ], style={"marginBottom": "20px"}),

                # Chart
                dcc.Graph(id='detailed-metrics-graph', config={"displayModeBar": False}),
            ]),
            subtitle="Compare key performance indicators across strategies",
        ),

        # Duration analysis section
        duration_section if duration_section else html.Div(),

        # Hourly Performance Section
        create_section_card(
            "Hourly Performance",
            html.Div([
                # Strategy selector
                html.Div([
                    html.Label(
                        "Select Strategy:",
                        style={
                            "fontWeight": "600",
                            "fontSize": "13px",
                            "color": COLORS["text_medium"],
                            "marginBottom": "8px",
                            "display": "block",
                        },
                    ),
                    dcc.Dropdown(
                        id='strategy-dropdown',
                        options=[{'label': s, 'value': s} for s in strategies],
                        value=default_strategy,
                        style={"fontSize": "14px", "maxWidth": "400px"},
                        clearable=False,
                    ),
                ], style={"marginBottom": "20px"}),

                # Hourly chart and stats
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(id='hourly-graph', config={"displayModeBar": False}),
                        lg=8, md=12,
                    ),
                    dbc.Col(
                        html.Div(id='hourly-stats'),
                        lg=4, md=12,
                    ),
                ]),
            ]),
            subtitle="Analyze performance patterns by hour of day",
        ),
    ])


def create_duration_chart(merged_deals):
    """
    Create trade duration distribution chart with statistics
    """
    merged_deals = merged_deals.copy()
    merged_deals["duration"] = (
        merged_deals["time_close"] - merged_deals["time_open"]
    ).dt.total_seconds() / 3600  # Convert to hours

    fig = go.Figure()

    symbols = merged_deals["symbol_open"].unique()
    for i, symbol in enumerate(symbols):
        symbol_data = merged_deals[merged_deals["symbol_open"] == symbol]
        fig.add_trace(go.Histogram(
            x=symbol_data["duration"],
            name=symbol,
            opacity=0.7,
            nbinsx=30,
            marker_color=CHART_COLORS[i % len(CHART_COLORS)],
            hovertemplate='Duration: %{x:.1f} hours<br>Count: %{y}<extra></extra>'
        ))

    fig.update_layout(
        xaxis_title="Duration (hours)",
        yaxis_title="Number of Trades",
        barmode='overlay',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    styled_fig = style_plotly_chart(fig, height=350)

    # Calculate duration statistics
    avg_duration = merged_deals["duration"].mean()
    median_duration = merged_deals["duration"].median()
    min_duration = merged_deals["duration"].min()
    max_duration = merged_deals["duration"].max()

    stats = html.Div([
        _create_duration_stat_card("Average Duration", f"{avg_duration:.1f} hrs"),
        _create_duration_stat_card("Median Duration", f"{median_duration:.1f} hrs"),
        _create_duration_stat_card("Shortest Trade", f"{min_duration:.1f} hrs"),
        _create_duration_stat_card("Longest Trade", f"{max_duration:.1f} hrs"),
    ], style={"display": "flex", "flexDirection": "column", "gap": "12px"})

    return styled_fig, stats


def _create_duration_stat_card(label, value):
    """Create a mini stat card for duration statistics."""
    return html.Div([
        html.Div(
            label,
            style={
                "fontSize": "12px",
                "color": COLORS["text_light"],
                "marginBottom": "4px",
            },
        ),
        html.Div(
            value,
            style={
                "fontSize": "18px",
                "fontWeight": "700",
                "color": COLORS["text_dark"],
            },
        ),
    ], style={
        "padding": "12px 16px",
        "backgroundColor": COLORS["background"],
        "borderRadius": "8px",
        "borderLeft": f"3px solid {COLORS['primary']}",
    })


def create_hourly_chart(merged_deals, strategy):
    """
    Create hourly performance chart for a specific strategy
    """
    strategy_deals = merged_deals[merged_deals["comment_open"] == strategy].copy()

    if strategy_deals.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected strategy",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS["text_light"]),
        )
        return style_plotly_chart(fig, height=350), html.Div()

    # Extract hour from time_open
    strategy_deals["hour"] = strategy_deals["time_open"].dt.hour
    strategy_deals["total_profit"] = strategy_deals["profit_open"] + strategy_deals["profit_close"]

    # Group by hour
    hourly_data = strategy_deals.groupby("hour").agg({
        "total_profit": ["sum", "mean", "count"],
    }).reset_index()
    hourly_data.columns = ["hour", "total_profit", "avg_profit", "trade_count"]

    # Create colors based on profit
    colors = [COLORS["success"] if p > 0 else COLORS["danger"] for p in hourly_data["total_profit"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=hourly_data["hour"],
        y=hourly_data["total_profit"],
        marker_color=colors,
        text=hourly_data["trade_count"],
        textposition="outside",
        texttemplate="%{text} trades",
        hovertemplate="Hour: %{x}:00<br>Total Profit: $%{y:,.2f}<br>Trades: %{text}<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_light"], line_width=1)

    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Total Profit ($)",
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(24)),
            ticktext=[f"{h:02d}:00" for h in range(24)],
            tickangle=-45,
        ),
    )

    styled_fig = style_plotly_chart(fig, height=350)

    # Calculate hourly stats
    best_hour = hourly_data.loc[hourly_data["total_profit"].idxmax()]
    worst_hour = hourly_data.loc[hourly_data["total_profit"].idxmin()]
    most_active = hourly_data.loc[hourly_data["trade_count"].idxmax()]
    total_trades = hourly_data["trade_count"].sum()

    stats = html.Div([
        _create_hourly_stat_card(
            "Best Hour",
            f"{int(best_hour['hour']):02d}:00",
            format_currency(best_hour["total_profit"]),
            "success"
        ),
        _create_hourly_stat_card(
            "Worst Hour",
            f"{int(worst_hour['hour']):02d}:00",
            format_currency(worst_hour["total_profit"]),
            "danger"
        ),
        _create_hourly_stat_card(
            "Most Active",
            f"{int(most_active['hour']):02d}:00",
            f"{int(most_active['trade_count'])} trades",
            "primary"
        ),
        _create_hourly_stat_card(
            "Total Trades",
            str(int(total_trades)),
            f"across {len(hourly_data)} hours",
            "neutral"
        ),
    ], style={"display": "flex", "flexDirection": "column", "gap": "12px"})

    return styled_fig, stats


def _create_hourly_stat_card(label, value, subtitle, color):
    """Create a mini stat card for hourly statistics."""
    color_map = {
        "success": {"bg": "#f0fdf4", "border": "#86efac", "text": "#166534"},
        "danger": {"bg": "#fef2f2", "border": "#fca5a5", "text": "#991b1b"},
        "primary": {"bg": "#eff6ff", "border": "#93c5fd", "text": "#1d4ed8"},
        "neutral": {"bg": "#f8fafc", "border": "#cbd5e1", "text": "#475569"},
    }
    colors = color_map.get(color, color_map["neutral"])

    return html.Div([
        html.Div(
            label,
            style={
                "fontSize": "12px",
                "color": COLORS["text_light"],
                "marginBottom": "4px",
            },
        ),
        html.Div(
            value,
            style={
                "fontSize": "20px",
                "fontWeight": "700",
                "color": colors["text"],
            },
        ),
        html.Div(
            subtitle,
            style={
                "fontSize": "12px",
                "color": COLORS["text_light"],
                "marginTop": "2px",
            },
        ),
    ], style={
        "padding": "12px 16px",
        "backgroundColor": colors["bg"],
        "borderRadius": "8px",
        "borderLeft": f"3px solid {colors['border']}",
    })


def create_detailed_metrics_figure(grouped_metrics, selected_metrics):
    """
    Create detailed metrics comparison chart with modern styling
    """
    plot_data = grouped_metrics.reset_index()
    plot_data['strategy_symbol'] = plot_data['comment_open'] + ' - ' + plot_data['symbol_open']

    fig = go.Figure()

    # Map metrics to chart colors
    color_map = {
        'Total Profit': CHART_COLORS[0],
        'Win Rate (%)': CHART_COLORS[1],
        'Sharpe Ratio': CHART_COLORS[2],
        'Max Drawdown (%)': CHART_COLORS[3],
        'Profit Factor': CHART_COLORS[4],
        'Average Profit by Trade': CHART_COLORS[5],
    }

    for metric in selected_metrics:
        if metric in plot_data.columns:
            fig.add_trace(go.Bar(
                x=plot_data['strategy_symbol'],
                y=plot_data[metric],
                name=metric,
                marker_color=color_map.get(metric, CHART_COLORS[0]),
                hovertemplate=f'{metric}: %{{y:,.2f}}<extra></extra>'
            ))

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Value",
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        xaxis_tickangle=-45,
    )

    return style_plotly_chart(fig, height=450)
