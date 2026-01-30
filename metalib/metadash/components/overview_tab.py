"""
Overview Tab Component
"""
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from utils.metrics import strategy_metrics, calculate_daily_performance
from .common_ui import (
    create_page_header,
    create_stat_card,
    create_section_card,
    style_plotly_chart,
    format_currency,
    format_percentage,
    COLORS,
    CHART_COLORS,
)


# Strategy name mapping - matches comment_open prefixes (longest match first)
STRATEGY_NAME_MAP = {
    'metago_weekly_': 'Weekly True Open',
    'metaga_': 'XGBoost Decision Tree',
    'metago_': 'Monthly True Open',
    'metaob_': 'Order Blocks',
    'metane_': 'Seasonality Decision Tree',
    'metafvg_': 'Fair Value Gaps',
}

# Sort by prefix length (longest first) for correct matching
_SORTED_PREFIXES = sorted(STRATEGY_NAME_MAP.keys(), key=len, reverse=True)

# Grid template for collapsible table columns
_GRID_TEMPLATE = "2.5fr 1fr 1fr 1fr 1fr 1fr 1fr 0.8fr"


def get_strategy_type_key(comment_open):
    """Return the matched prefix key for grouping (handles metago_weekly_ vs metago_)."""
    if not comment_open or not isinstance(comment_open, str):
        return comment_open or "unknown"
    for prefix in _SORTED_PREFIXES:
        if comment_open.startswith(prefix):
            return prefix
    return comment_open


def get_strategy_display_name(comment_open):
    """Match longest prefix first, return human-readable name."""
    if not comment_open or not isinstance(comment_open, str):
        return comment_open or "Unknown"
    for prefix in _SORTED_PREFIXES:
        if comment_open.startswith(prefix):
            return STRATEGY_NAME_MAP[prefix]
    return comment_open


def _format_profit_factor(value):
    """Format profit factor, handling infinity."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "---"
    if isinstance(value, float) and np.isinf(value):
        return "inf"
    return f"{value:.2f}"


def render_overview_tab(merged_deals, account_size, account_info):
    """
    Render the overview tab with account info and strategy metrics
    """
    # Calculate quick stats
    total_profit = merged_deals["profit_open"].sum() + merged_deals["profit_close"].sum()
    total_trades = len(merged_deals)
    winning_trades = len(merged_deals[(merged_deals["profit_open"] + merged_deals["profit_close"]) > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    # Find best performing strategy (using human-readable name)
    strategy_metrics_df = merged_deals[
        ["profit_open", "profit_close", "comment_open", "symbol_open", "time_open"]
    ].copy()
    grouped_metrics = strategy_metrics_df.groupby(["comment_open", "symbol_open"]).apply(
        lambda x: strategy_metrics(x, account_size)
    )

    if not grouped_metrics.empty:
        best_strategy_idx = grouped_metrics["Total Profit"].idxmax()
        best_strategy = get_strategy_display_name(best_strategy_idx[0])
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

    # Count strategy types for subtitle
    type_keys = strategy_metrics_df["comment_open"].apply(get_strategy_type_key).nunique()

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

            # Strategy Metrics Table (collapsible)
            create_section_card(
                "Strategy Performance Metrics",
                create_collapsible_metrics_table(merged_deals, account_size),
                subtitle=f"{type_keys} strategy types",
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


def create_collapsible_metrics_table(merged_deals, account_size):
    """
    Create a collapsible metrics table grouped by strategy type.
    Each strategy type gets a clickable header row with aggregated metrics
    and a collapsible section containing per-symbol child rows.
    """
    deals_df = merged_deals[
        ["profit_open", "profit_close", "comment_open", "symbol_open", "time_open"]
    ].copy()
    deals_df["strategy_type_key"] = deals_df["comment_open"].apply(get_strategy_type_key)

    # Column names
    col_names = [
        "Strategy / Symbol", "Total Profit", "Win Rate", "Avg Profit",
        "Sharpe", "Max DD", "Profit Factor", "# Trades",
    ]

    # Header row (column labels)
    header_cells = [
        html.Div(
            col,
            style={
                "fontWeight": "600",
                "fontSize": "11px",
                "color": COLORS["text_medium"],
                "textTransform": "uppercase",
                "letterSpacing": "0.5px",
            },
        )
        for col in col_names
    ]

    header_row = html.Div(
        header_cells,
        style={
            "display": "grid",
            "gridTemplateColumns": _GRID_TEMPLATE,
            "padding": "12px 16px",
            "backgroundColor": COLORS["background"],
            "borderBottom": f"2px solid {COLORS['border']}",
        },
    )

    # Group deals by strategy type, sorted by total profit descending
    type_groups = deals_df.groupby("strategy_type_key")
    sorted_groups = sorted(
        type_groups,
        key=lambda g: -(g[1]["profit_open"].sum() + g[1]["profit_close"].sum()),
    )

    content_rows = []

    for type_key, type_deals in sorted_groups:
        display_name = STRATEGY_NAME_MAP.get(type_key, type_key)

        # Compute aggregated metrics for this strategy type
        agg_metrics = strategy_metrics(type_deals, account_size)

        # Build clickable header row
        profit_color = COLORS["success"] if agg_metrics["Total Profit"] >= 0 else COLORS["danger"]

        strategy_header = html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            "▶ ",
                            style={
                                "fontSize": "10px",
                                "color": COLORS["text_light"],
                                "marginRight": "6px",
                            },
                        ),
                        html.Span(display_name, style={"fontWeight": "600"}),
                    ],
                ),
                html.Div(
                    format_currency(agg_metrics["Total Profit"]),
                    style={"color": profit_color, "fontWeight": "600"},
                ),
                html.Div(f'{agg_metrics["Win Rate (%)"]:.1f}%'),
                html.Div(format_currency(agg_metrics["Average Profit by Trade"])),
                html.Div(f'{agg_metrics["Sharpe Ratio"]:.2f}'),
                html.Div(f'{agg_metrics["Max Drawdown (%)"]:.1f}%'),
                html.Div(_format_profit_factor(agg_metrics["Profit Factor"])),
                html.Div(f'{int(agg_metrics["Number of Trades"])}'),
            ],
            id={"type": "strategy-header", "index": type_key},
            n_clicks=0,
            style={
                "display": "grid",
                "gridTemplateColumns": _GRID_TEMPLATE,
                "padding": "12px 16px",
                "backgroundColor": "#f1f5f9",
                "cursor": "pointer",
                "borderBottom": f"1px solid {COLORS['border']}",
                "fontSize": "13px",
                "color": COLORS["text_dark"],
                "alignItems": "center",
            },
        )

        # Build per-symbol child rows
        child_rows = []
        symbol_groups = type_deals.groupby("symbol_open")
        sorted_symbols = sorted(
            symbol_groups,
            key=lambda g: -(g[1]["profit_open"].sum() + g[1]["profit_close"].sum()),
        )

        for idx, (symbol, symbol_deals) in enumerate(sorted_symbols):
            sym_metrics = strategy_metrics(symbol_deals, account_size)
            bg_color = "white" if idx % 2 == 0 else COLORS["background"]
            sym_profit_color = COLORS["success"] if sym_metrics["Total Profit"] >= 0 else COLORS["danger"]

            child_row = html.Div(
                [
                    html.Div(
                        symbol,
                        style={"paddingLeft": "28px", "color": COLORS["text_medium"]},
                    ),
                    html.Div(
                        format_currency(sym_metrics["Total Profit"]),
                        style={"color": sym_profit_color},
                    ),
                    html.Div(f'{sym_metrics["Win Rate (%)"]:.1f}%'),
                    html.Div(format_currency(sym_metrics["Average Profit by Trade"])),
                    html.Div(f'{sym_metrics["Sharpe Ratio"]:.2f}'),
                    html.Div(f'{sym_metrics["Max Drawdown (%)"]:.1f}%'),
                    html.Div(_format_profit_factor(sym_metrics["Profit Factor"])),
                    html.Div(f'{int(sym_metrics["Number of Trades"])}'),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": _GRID_TEMPLATE,
                    "padding": "10px 16px",
                    "backgroundColor": bg_color,
                    "borderBottom": f"1px solid {COLORS['border_light']}",
                    "fontSize": "13px",
                    "color": COLORS["text_dark"],
                    "alignItems": "center",
                },
            )
            child_rows.append(child_row)

        # Wrap child rows in a Collapse component
        collapse = dbc.Collapse(
            html.Div(child_rows),
            id={"type": "strategy-collapse", "index": type_key},
            is_open=False,
        )

        content_rows.append(strategy_header)
        content_rows.append(collapse)

    return html.Div(
        [header_row] + content_rows,
        style={
            "borderRadius": "8px",
            "overflow": "hidden",
            "border": f"1px solid {COLORS['border']}",
        },
    )
