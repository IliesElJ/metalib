"""
Overview Tab Component
"""
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from utils.metrics import strategy_metrics, calculate_daily_performance
from .common_ui import (
    create_page_header,
    create_stat_card,
    create_section_card,
    create_styled_table,
    style_plotly_chart,
    format_currency,
    format_percentage,
    COLORS,
    CHART_COLORS,
)


def render_overview_tab(merged_deals, account_size, account_info):
    """
    Render the overview tab with account info and strategy metrics
    """
    # Calculate strategy metrics
    strategy_metrics_df = merged_deals[
        ["profit_open", "profit_close", "comment_open", "symbol_open", "time_open"]
    ].copy()

    # Group by strategy and symbol
    grouped_metrics = strategy_metrics_df.groupby(["comment_open", "symbol_open"]).apply(
        lambda x: strategy_metrics(x, account_size)
    )

    # Calculate quick stats
    total_profit = merged_deals["profit_open"].sum() + merged_deals["profit_close"].sum()
    total_trades = len(merged_deals)
    winning_trades = len(merged_deals[(merged_deals["profit_open"] + merged_deals["profit_close"]) > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    # Find best performing strategy
    if not grouped_metrics.empty:
        best_strategy_idx = grouped_metrics["Total Profit"].idxmax()
        best_strategy = f"{best_strategy_idx[0]}"
    else:
        best_strategy = "N/A"

    # Calculate daily change (if we have time data)
    daily_change = None
    daily_change_pct = None
    if "time_open" in merged_deals.columns:
        import pandas as pd
        today = pd.Timestamp.now().date()
        today_deals = merged_deals[pd.to_datetime(merged_deals["time_open"]).dt.date == today]
        if not today_deals.empty:
            daily_change = today_deals["profit_open"].sum() + today_deals["profit_close"].sum()
            daily_change_pct = (daily_change / account_info["balance"] * 100) if account_info["balance"] > 0 else 0

    # Create strategy comparison figure
    comparison_fig = create_strategy_comparison_chart(grouped_metrics)

    return html.Div(
        [
            # Header
            create_page_header(
                "Account Overview",
                "Real-time snapshot of your trading account and strategy performance"
            ),

            # Account Cards Row
            dbc.Row(
                [
                    dbc.Col(
                        create_stat_card(
                            "Balance",
                            format_currency(account_info["balance"]),
                            subtitle=f"{format_percentage(daily_change_pct, show_sign=True)} today" if daily_change_pct else None,
                            color="primary",
                            icon="$",
                        ),
                        lg=4, md=4, sm=12,
                        className="mb-3",
                    ),
                    dbc.Col(
                        create_stat_card(
                            "Equity",
                            format_currency(account_info["equity"]),
                            subtitle=format_currency(account_info["equity"] - account_info["balance"], show_sign=True) if account_info["equity"] != account_info["balance"] else "No open P&L",
                            color="success" if account_info["equity"] >= account_info["balance"] else "danger",
                            icon="E",
                        ),
                        lg=4, md=4, sm=12,
                        className="mb-3",
                    ),
                    dbc.Col(
                        create_stat_card(
                            "Margin Used",
                            format_currency(account_info["margin"]),
                            subtitle=f"{(account_info['margin'] / account_info['equity'] * 100):.1f}% of equity" if account_info["equity"] > 0 else None,
                            color="cyan",
                            icon="M",
                        ),
                        lg=4, md=4, sm=12,
                        className="mb-3",
                    ),
                ],
                className="mb-4",
            ),

            # Quick Stats Row
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                _create_quick_stat("Total Trades", str(total_trades), "#"),
                                lg=3, md=6, sm=6, xs=12,
                                className="mb-3",
                            ),
                            dbc.Col(
                                _create_quick_stat(
                                    "Win Rate",
                                    format_percentage(win_rate),
                                    "%",
                                    "success" if win_rate >= 50 else "danger"
                                ),
                                lg=3, md=6, sm=6, xs=12,
                                className="mb-3",
                            ),
                            dbc.Col(
                                _create_quick_stat(
                                    "Total P&L",
                                    format_currency(total_profit),
                                    "$",
                                    "success" if total_profit >= 0 else "danger"
                                ),
                                lg=3, md=6, sm=6, xs=12,
                                className="mb-3",
                            ),
                            dbc.Col(
                                _create_quick_stat("Best Strategy", best_strategy, "★"),
                                lg=3, md=6, sm=6, xs=12,
                                className="mb-3",
                            ),
                        ],
                    ),
                ],
                style={
                    "backgroundColor": COLORS["background"],
                    "padding": "16px",
                    "borderRadius": "12px",
                    "marginBottom": "24px",
                },
            ),

            # Strategy Metrics Table
            create_section_card(
                "Strategy Performance Metrics",
                create_metrics_table(grouped_metrics),
                subtitle=f"{len(grouped_metrics)} strategy-symbol combinations",
            ),

            # Strategy Comparison Chart
            create_section_card(
                "Performance Comparison",
                dcc.Graph(figure=comparison_fig, config={"displayModeBar": False}),
                subtitle="Key metrics across all strategies",
            ),
        ]
    )


def _create_quick_stat(label, value, icon, color="neutral"):
    """Create a compact quick stat display."""
    color_map = {
        "success": COLORS["success"],
        "danger": COLORS["danger"],
        "warning": COLORS["warning"],
        "neutral": COLORS["text_dark"],
    }

    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        icon,
                        style={
                            "color": color_map.get(color, COLORS["primary"]),
                            "marginRight": "8px",
                            "fontSize": "14px",
                        },
                    ),
                    html.Span(
                        label,
                        style={
                            "color": COLORS["text_light"],
                            "fontSize": "12px",
                            "fontWeight": "500",
                        },
                    ),
                ],
            ),
            html.Div(
                value,
                style={
                    "fontSize": "20px",
                    "fontWeight": "700",
                    "color": color_map.get(color, COLORS["text_dark"]),
                    "marginTop": "4px",
                },
            ),
        ],
        style={
            "backgroundColor": "white",
            "padding": "16px",
            "borderRadius": "8px",
            "border": f"1px solid {COLORS['border']}",
        },
    )


def create_strategy_comparison_chart(grouped_metrics):
    """
    Create strategy comparison bar chart
    """
    plot_metrics = ["Total Profit", "Average Profit by Trade", "Win Rate (%)", "Sharpe Ratio"]
    plot_data = grouped_metrics.reset_index()
    plot_data["strategy_symbol"] = plot_data["comment_open"] + " - " + plot_data["symbol_open"]

    fig = go.Figure()

    colors = [CHART_COLORS[0], CHART_COLORS[1], CHART_COLORS[2], CHART_COLORS[4]]

    for i, metric in enumerate(plot_metrics):
        if metric in plot_data.columns:
            fig.add_trace(
                go.Bar(
                    x=plot_data["strategy_symbol"],
                    y=plot_data[metric],
                    name=metric,
                    marker_color=colors[i % len(colors)],
                    hovertemplate=f"{metric}: %{{y:,.2f}}<extra></extra>",
                )
            )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Value",
        barmode="group",
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


def create_metrics_table(grouped_metrics):
    """
    Create a formatted metrics table
    """
    df = grouped_metrics.reset_index()

    # Rename columns for display
    df = df.rename(columns={
        "comment_open": "Strategy",
        "symbol_open": "Symbol",
    })

    # Select and order columns
    display_cols = ["Strategy", "Symbol", "Total Profit", "Win Rate (%)",
                    "Average Profit by Trade", "Sharpe Ratio", "Max Drawdown (%)",
                    "Profit Factor", "Number of Trades"]

    available_cols = [c for c in display_cols if c in df.columns]
    df = df[available_cols]

    # Build column definitions
    columns = []
    for col in df.columns:
        col_def = {"name": col, "id": col}
        if col not in ["Strategy", "Symbol"]:
            col_def["type"] = "numeric"
            col_def["format"] = {"specifier": ".2f"}
        columns.append(col_def)

    return create_styled_table(
        data=df.to_dict("records"),
        columns=columns,
    )
