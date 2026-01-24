"""
PnL Performance Tab Component
"""
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from utils.metrics import calculate_streak_analysis
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


def render_pnl_tab(merged_deals, account_size):
    """
    Render the PnL performance tab
    """
    # Calculate PnL metrics
    merged_deals = merged_deals.copy()
    merged_deals["total_profit"] = merged_deals["profit_open"] + merged_deals["profit_close"]
    merged_deals_sorted = merged_deals.sort_values("time_open")
    merged_deals_sorted["cumulative_profit"] = merged_deals_sorted["total_profit"].cumsum()

    # Calculate key metrics for summary cards
    total_pnl = merged_deals_sorted["total_profit"].sum()
    max_equity = account_size + merged_deals_sorted["cumulative_profit"].cummax()
    current_equity = account_size + merged_deals_sorted["cumulative_profit"].iloc[-1] if len(merged_deals_sorted) > 0 else account_size
    drawdown_series = (current_equity - max_equity) / max_equity * 100
    max_drawdown = merged_deals_sorted["cumulative_profit"].cummax() - merged_deals_sorted["cumulative_profit"]
    max_drawdown_pct = (max_drawdown / (account_size + merged_deals_sorted["cumulative_profit"].cummax()) * 100).max()

    # Sharpe ratio calculation
    if len(merged_deals_sorted) > 1:
        returns = merged_deals_sorted["total_profit"] / account_size
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    else:
        sharpe = 0

    # Win rate
    wins = (merged_deals_sorted["total_profit"] > 0).sum()
    total_trades = len(merged_deals_sorted)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    # Profit factor
    gross_profit = merged_deals_sorted[merged_deals_sorted["total_profit"] > 0]["total_profit"].sum()
    gross_loss = abs(merged_deals_sorted[merged_deals_sorted["total_profit"] < 0]["total_profit"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Create charts
    equity_fig = create_equity_curve(merged_deals_sorted, account_size)
    drawdown_fig = create_drawdown_chart(merged_deals_sorted, account_size)
    symbol_fig = create_symbol_profit_chart(merged_deals_sorted)
    scatter_fig = create_trade_scatter(merged_deals_sorted)
    streak_fig, streak_stats = create_streak_analysis(merged_deals_sorted)

    # Strategy performance table
    strategy_perf = calculate_strategy_performance(merged_deals)

    return html.Div(
        [
            # Header
            create_page_header(
                "PnL Performance Analysis",
                "Track your profit/loss, drawdowns, and trading patterns over time"
            ),

            # Key Metrics Summary Row
            dbc.Row(
                [
                    dbc.Col(
                        create_stat_card(
                            "Total P&L",
                            format_currency(total_pnl),
                            color="success" if total_pnl >= 0 else "danger",
                            icon="$",
                        ),
                        lg=True, md=6, sm=6, xs=12,
                        className="mb-3",
                    ),
                    dbc.Col(
                        create_stat_card(
                            "Max Drawdown",
                            format_percentage(-max_drawdown_pct),
                            color="danger" if max_drawdown_pct > 10 else "warning" if max_drawdown_pct > 5 else "success",
                            icon="↓",
                        ),
                        lg=True, md=6, sm=6, xs=12,
                        className="mb-3",
                    ),
                    dbc.Col(
                        create_stat_card(
                            "Sharpe Ratio",
                            f"{sharpe:.2f}",
                            color="success" if sharpe > 1 else "warning" if sharpe > 0 else "danger",
                            icon="σ",
                        ),
                        lg=True, md=6, sm=6, xs=12,
                        className="mb-3",
                    ),
                    dbc.Col(
                        create_stat_card(
                            "Win Rate",
                            format_percentage(win_rate),
                            color="success" if win_rate >= 50 else "danger",
                            icon="%",
                        ),
                        lg=True, md=6, sm=6, xs=12,
                        className="mb-3",
                    ),
                    dbc.Col(
                        create_stat_card(
                            "Profit Factor",
                            f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞",
                            color="success" if profit_factor > 1.5 else "warning" if profit_factor > 1 else "danger",
                            icon="×",
                        ),
                        lg=True, md=6, sm=6, xs=12,
                        className="mb-3",
                    ),
                ],
                className="mb-4",
            ),

            # Equity and Drawdown Charts (2-column)
            dbc.Row(
                [
                    dbc.Col(
                        create_section_card(
                            "Equity Curve",
                            dcc.Graph(figure=equity_fig, config={"displayModeBar": False}),
                            subtitle="Account value over time",
                        ),
                        lg=6, md=12,
                    ),
                    dbc.Col(
                        create_section_card(
                            "Drawdown",
                            dcc.Graph(figure=drawdown_fig, config={"displayModeBar": False}),
                            subtitle="Peak-to-trough decline",
                        ),
                        lg=6, md=12,
                    ),
                ],
            ),

            # Cumulative Profit by Symbol
            create_section_card(
                "Cumulative Profit by Symbol",
                dcc.Graph(figure=symbol_fig, config={"displayModeBar": False}),
                subtitle="Performance breakdown by trading instrument",
            ),

            # Individual Trade Performance
            create_section_card(
                "Individual Trade Performance",
                dcc.Graph(figure=scatter_fig, config={"displayModeBar": False}),
                subtitle="Each trade plotted by date and profit/loss",
            ),

            # Streak Analysis (2-column)
            dbc.Row(
                [
                    dbc.Col(
                        create_section_card(
                            "Win/Loss Streaks",
                            dcc.Graph(figure=streak_fig, config={"displayModeBar": False}),
                        ),
                        lg=8, md=12,
                    ),
                    dbc.Col(
                        create_section_card(
                            "Streak Statistics",
                            streak_stats,
                        ),
                        lg=4, md=12,
                    ),
                ],
            ),

            # Strategy Performance Table
            create_section_card(
                "Performance by Strategy",
                create_strategy_table(strategy_perf),
                subtitle="Aggregated metrics for each strategy",
            ),
        ]
    )


def create_equity_curve(equity_data, account_size):
    """
    Create account equity curve chart
    """
    equity_data = equity_data.copy()
    equity_data["equity"] = account_size + equity_data["cumulative_profit"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_data["time_open"],
            y=equity_data["equity"],
            mode="lines",
            name="Account Equity",
            line=dict(color=CHART_COLORS[0], width=2),
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.1)",
            hovertemplate="Date: %{x|%Y-%m-%d %H:%M}<br>Equity: $%{y:,.2f}<extra></extra>",
        )
    )

    # Add starting balance line
    fig.add_hline(
        y=account_size,
        line_dash="dash",
        line_color=COLORS["text_light"],
        annotation_text="Starting Balance",
        annotation_position="top left",
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Equity ($)",
        showlegend=False,
    )

    return style_plotly_chart(fig, height=350)


def create_drawdown_chart(equity_data, account_size):
    """
    Create drawdown chart
    """
    equity_data = equity_data.copy()
    equity_data["equity"] = account_size + equity_data["cumulative_profit"]
    equity_data["running_max"] = equity_data["equity"].cummax()
    equity_data["drawdown"] = (equity_data["equity"] / equity_data["running_max"] - 1) * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_data["time_open"],
            y=equity_data["drawdown"],
            mode="lines",
            line=dict(color=COLORS["danger"], width=2),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.15)",
            name="Drawdown",
            hovertemplate="Date: %{x|%Y-%m-%d %H:%M}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Drawdown (%)",
        showlegend=False,
    )

    return style_plotly_chart(fig, height=350)


def create_symbol_profit_chart(merged_deals_sorted):
    """
    Create cumulative profit by symbol chart
    """
    symbols = merged_deals_sorted["symbol_open"].unique()
    fig = go.Figure()

    for i, symbol in enumerate(symbols):
        symbol_data = merged_deals_sorted[merged_deals_sorted["symbol_open"] == symbol].copy()
        symbol_data = symbol_data.sort_values("time_open")
        symbol_data["cumulative_profit"] = symbol_data["total_profit"].cumsum()

        fig.add_trace(
            go.Scatter(
                x=symbol_data["time_open"],
                y=symbol_data["cumulative_profit"],
                mode="lines",
                name=symbol,
                line=dict(width=2, color=CHART_COLORS[i % len(CHART_COLORS)]),
                hovertemplate=f"{symbol}<br>%{{x|%Y-%m-%d}}<br>${{y:,.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Cumulative Profit ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    return style_plotly_chart(fig, height=350)


def create_trade_scatter(merged_deals_sorted):
    """
    Create trade scatter plot
    """
    fig = go.Figure()

    # Color trades by profit/loss
    colors = [COLORS["success"] if p > 0 else COLORS["danger"] for p in merged_deals_sorted["total_profit"]]

    # Scale marker size based on profit magnitude
    max_profit = np.abs(merged_deals_sorted["total_profit"]).max()
    marker_sizes = np.abs(merged_deals_sorted["total_profit"]) / max(1, max_profit) * 25 + 8

    fig.add_trace(
        go.Scatter(
            x=merged_deals_sorted["time_open"],
            y=merged_deals_sorted["total_profit"],
            mode="markers",
            marker=dict(
                size=marker_sizes,
                color=colors,
                opacity=0.7,
                line=dict(width=1, color="white"),
            ),
            text=merged_deals_sorted["comment_open"],
            customdata=merged_deals_sorted["symbol_open"],
            hovertemplate="Date: %{x|%Y-%m-%d %H:%M}<br>Profit: $%{y:,.2f}<br>Strategy: %{text}<br>Symbol: %{customdata}<extra></extra>",
        )
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_light"], line_width=1)

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Profit/Loss ($)",
        showlegend=False,
    )

    return style_plotly_chart(fig, height=400)


def create_streak_analysis(merged_deals_sorted):
    """
    Create streak analysis chart and statistics
    """
    streaks = calculate_streak_analysis(merged_deals_sorted)

    win_streaks = streaks[streaks["win"]]
    loss_streaks = streaks[~streaks["win"]]

    fig = go.Figure()

    if not win_streaks.empty:
        fig.add_trace(
            go.Bar(
                x=win_streaks.index,
                y=win_streaks["streak_length"],
                name="Win Streaks",
                marker_color=COLORS["success"],
                text=win_streaks["total_profit"].round(2),
                hovertemplate="Streak #%{x}<br>Length: %{y} trades<br>Profit: $%{text}<extra></extra>",
            )
        )

    if not loss_streaks.empty:
        fig.add_trace(
            go.Bar(
                x=loss_streaks.index,
                y=-loss_streaks["streak_length"],
                name="Loss Streaks",
                marker_color=COLORS["danger"],
                text=loss_streaks["total_profit"].round(2),
                hovertemplate="Streak #%{x}<br>Length: %{y} trades<br>Loss: $%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Streak #",
        yaxis_title="Streak Length",
        barmode="relative",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    styled_fig = style_plotly_chart(fig, height=300)

    # Calculate streak statistics - styled cards
    longest_win = win_streaks["streak_length"].max() if not win_streaks.empty else 0
    longest_loss = loss_streaks["streak_length"].max() if not loss_streaks.empty else 0
    best_profit = win_streaks["total_profit"].max() if not win_streaks.empty else 0
    worst_loss = loss_streaks["total_profit"].min() if not loss_streaks.empty else 0

    stats = html.Div(
        [
            _create_streak_stat_card("Longest Win Streak", f"{longest_win} trades", "success"),
            _create_streak_stat_card("Longest Loss Streak", f"{longest_loss} trades", "danger"),
            _create_streak_stat_card("Best Streak Profit", format_currency(best_profit), "success"),
            _create_streak_stat_card("Worst Streak Loss", format_currency(worst_loss), "danger"),
        ],
        style={"display": "flex", "flexDirection": "column", "gap": "12px"},
    )

    return styled_fig, stats


def _create_streak_stat_card(label, value, color):
    """Create a mini stat card for streak statistics."""
    color_map = {
        "success": {"bg": "#f0fdf4", "border": "#86efac", "text": "#166534"},
        "danger": {"bg": "#fef2f2", "border": "#fca5a5", "text": "#991b1b"},
    }
    colors = color_map.get(color, color_map["success"])

    return html.Div(
        [
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
                    "color": colors["text"],
                },
            ),
        ],
        style={
            "padding": "12px 16px",
            "backgroundColor": colors["bg"],
            "borderRadius": "8px",
            "borderLeft": f"3px solid {colors['border']}",
        },
    )


def calculate_strategy_performance(merged_deals):
    """
    Calculate performance metrics by strategy
    """
    merged_deals = merged_deals.copy()
    merged_deals["total_profit"] = merged_deals["profit_open"] + merged_deals["profit_close"]

    strategy_perf = merged_deals.groupby("comment_open").agg(
        {"total_profit": ["sum", "mean", "count", "std"]}
    ).round(2)

    strategy_perf.columns = ["Total Profit", "Average Profit", "Number of Trades", "Std Dev"]

    # Calculate win rate
    win_rates = merged_deals.groupby("comment_open")["total_profit"].apply(
        lambda x: 100 * (x > 0).mean()
    ).round(2)

    strategy_perf["Win Rate (%)"] = win_rates
    strategy_perf = strategy_perf.reset_index()
    strategy_perf.rename(columns={"comment_open": "Strategy"}, inplace=True)

    # Reorder columns
    strategy_perf = strategy_perf[["Strategy", "Total Profit", "Average Profit", "Win Rate (%)", "Number of Trades", "Std Dev"]]

    return strategy_perf


def create_strategy_table(strategy_perf):
    """
    Create strategy performance table
    """
    columns = []
    for col in strategy_perf.columns:
        col_def = {"name": col, "id": col}
        if col != "Strategy":
            col_def["type"] = "numeric"
            col_def["format"] = {"specifier": ".2f"}
        columns.append(col_def)

    return create_styled_table(
        data=strategy_perf.to_dict("records"),
        columns=columns,
    )
