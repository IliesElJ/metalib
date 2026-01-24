"""
Status Monitoring Tab Component
Displays health status of all trading strategies.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc
from utils.health_utils import get_all_strategy_statuses, get_health_summary, is_weekend


def render_status_tab():
    """
    Render the status monitoring tab.

    Returns:
        Dash HTML component with status monitoring interface
    """
    summary = get_health_summary()
    weekend = is_weekend()

    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "Strategy Status Monitor",
                                style={
                                    "fontSize": "24px",
                                    "fontWeight": "700",
                                    "color": "#1e293b",
                                },
                            ),
                            # Weekend badge
                            html.Span(
                                "WEEKEND",
                                style={
                                    "backgroundColor": "#8b5cf6",
                                    "color": "white",
                                    "padding": "4px 12px",
                                    "borderRadius": "12px",
                                    "fontSize": "12px",
                                    "fontWeight": "600",
                                    "marginLeft": "12px",
                                },
                            ) if weekend else None,
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.P(
                        "Real-time health monitoring of all trading strategies based on log activity",
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
            # Summary cards
            html.Div(id="status-summary-container", className="mb-4"),
            # Refresh button
            html.Div(
                [
                    dbc.Button(
                        "Refresh Status",
                        id="status-refresh-btn",
                        color="primary",
                        size="sm",
                        className="mb-3",
                    ),
                    html.Span(
                        id="status-last-refresh",
                        style={
                            "marginLeft": "12px",
                            "color": "#64748b",
                            "fontSize": "13px",
                        },
                    ),
                ],
            ),
            # Status table
            html.Div(
                id="status-table-container",
                style={
                    "backgroundColor": "white",
                    "borderRadius": "12px",
                    "border": "1px solid #e2e8f0",
                    "overflow": "hidden",
                },
            ),
            # Auto-refresh interval (every 60 seconds)
            dcc.Interval(
                id="status-auto-refresh",
                interval=60 * 1000,  # 60 seconds
                n_intervals=0,
            ),
        ],
        style={"padding": "0"},
    )


def create_status_summary(summary):
    """
    Create summary cards showing counts by status.

    Args:
        summary: Dict with status counts

    Returns:
        Dash component with summary cards
    """
    weekend = summary.get("is_weekend", False)

    cards = [
        {
            "title": "Total Strategies",
            "value": summary.get("total", 0),
            "color": "#3b82f6",
            "bg": "#eff6ff",
            "icon": "#",
        },
        {
            "title": "Running",
            "value": summary.get("running", 0),
            "color": "#22c55e",
            "bg": "#f0fdf4",
            "icon": "OK",
        },
        {
            "title": "Stale",
            "value": summary.get("stale", 0),
            "color": "#f59e0b",
            "bg": "#fffbeb",
            "icon": "?",
        },
        {
            "title": "Stopped",
            "value": summary.get("stopped", 0),
            "color": "#ef4444",
            "bg": "#fef2f2",
            "icon": "!",
        },
    ]

    if weekend:
        cards.append({
            "title": "Weekend Mode",
            "value": summary.get("weekend", 0),
            "color": "#8b5cf6",
            "bg": "#f5f3ff",
            "icon": "~",
        })

    return dbc.Row(
        [
            dbc.Col(
                _create_summary_card(card["title"], card["value"], card["icon"], card["color"], card["bg"]),
                lg=2 if weekend else 3,
                md=4,
                sm=6,
                xs=12,
                className="mb-3",
            )
            for card in cards
        ],
    )


def _create_summary_card(title, value, icon, color, bg_color):
    """Create a summary card."""
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
                            "fontSize": "14px",
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
                                str(value),
                                style={
                                    "fontSize": "24px",
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


def create_status_table(statuses):
    """
    Create the status table showing all strategies.

    Args:
        statuses: List of status dicts

    Returns:
        Dash component with status table
    """
    if not statuses:
        return html.Div(
            "No strategies configured",
            style={
                "padding": "40px",
                "textAlign": "center",
                "color": "#64748b",
            },
        )

    # Table header
    header = html.Div(
        [
            html.Div("Status", style={"width": "100px", "fontWeight": "600"}),
            html.Div("Strategy", style={"flex": "1", "fontWeight": "600"}),
            html.Div("Type", style={"width": "100px", "fontWeight": "600"}),
            html.Div("Symbols", style={"width": "120px", "fontWeight": "600"}),
            html.Div("Timeframe", style={"width": "120px", "fontWeight": "600"}),
            html.Div("Last Activity", style={"width": "150px", "fontWeight": "600"}),
            html.Div("Threshold", style={"width": "100px", "fontWeight": "600"}),
        ],
        style={
            "display": "flex",
            "padding": "12px 20px",
            "backgroundColor": "#f8fafc",
            "borderBottom": "1px solid #e2e8f0",
            "fontSize": "13px",
            "color": "#475569",
        },
    )

    # Table rows
    rows = [_create_status_row(s) for s in statuses]

    return html.Div([header] + rows)


def _create_status_row(status):
    """Create a single status row."""
    status_badges = {
        "running": {"text": "Running", "bg": "#dcfce7", "color": "#166534", "border": "#86efac"},
        "stale": {"text": "Stale", "bg": "#fef3c7", "color": "#92400e", "border": "#fcd34d"},
        "stopped": {"text": "Stopped", "bg": "#fee2e2", "color": "#991b1b", "border": "#fca5a5"},
        "weekend": {"text": "Weekend", "bg": "#ede9fe", "color": "#5b21b6", "border": "#c4b5fd"},
        "unknown": {"text": "Unknown", "bg": "#f1f5f9", "color": "#475569", "border": "#cbd5e1"},
    }

    badge_info = status_badges.get(status["status"], status_badges["unknown"])

    # Format timeframe for display
    timeframe_display = status.get("timeframe", "").replace("mt5.TIMEFRAME_", "")

    return html.Div(
        [
            # Status badge
            html.Div(
                html.Span(
                    badge_info["text"],
                    style={
                        "backgroundColor": badge_info["bg"],
                        "color": badge_info["color"],
                        "border": f"1px solid {badge_info['border']}",
                        "padding": "4px 10px",
                        "borderRadius": "6px",
                        "fontSize": "12px",
                        "fontWeight": "600",
                    },
                ),
                style={"width": "100px"},
            ),
            # Strategy tag
            html.Div(
                html.Span(
                    status["tag"],
                    style={
                        "fontWeight": "600",
                        "color": "#1e293b",
                    },
                ),
                style={"flex": "1"},
            ),
            # Strategy type
            html.Div(
                status.get("strategy_type", "").upper(),
                style={
                    "width": "100px",
                    "color": "#64748b",
                    "fontSize": "13px",
                },
            ),
            # Symbols
            html.Div(
                ", ".join(status.get("symbols", [])),
                style={
                    "width": "120px",
                    "color": "#64748b",
                    "fontSize": "13px",
                },
            ),
            # Timeframe
            html.Div(
                timeframe_display,
                style={
                    "width": "120px",
                    "color": "#64748b",
                    "fontSize": "13px",
                },
            ),
            # Last activity
            html.Div(
                [
                    html.Div(
                        status.get("last_activity_str", "N/A"),
                        style={
                            "fontWeight": "500",
                            "color": "#1e293b" if status["status"] == "running" else "#ef4444" if status["status"] == "stopped" else "#64748b",
                        },
                    ),
                    html.Div(
                        status.get("log_date", ""),
                        style={
                            "fontSize": "11px",
                            "color": "#94a3b8",
                        },
                    ) if status.get("log_date") else None,
                ],
                style={"width": "150px"},
            ),
            # Threshold info
            html.Div(
                f"{status.get('stale_threshold', 0)}m / {status.get('stopped_threshold', 0)}m",
                style={
                    "width": "100px",
                    "color": "#94a3b8",
                    "fontSize": "12px",
                },
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "padding": "14px 20px",
            "borderBottom": "1px solid #f1f5f9",
            "fontSize": "14px",
            "backgroundColor": "#fef2f2" if status["status"] == "stopped" else "white",
        },
    )
