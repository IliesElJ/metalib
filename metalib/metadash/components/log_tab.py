"""
Log Viewer Tab Component
"""
from dash import html, dcc
import dash_bootstrap_components as dbc
from utils.log_utils import (
    get_strategy_instances,
    get_dates_for_strategy,
    read_log_file,
    parse_log_content,
    get_log_statistics,
)

# Hardcoded strategy types
STRATEGY_TYPES = [
    {"value": "all", "label": "All Strategies"},
    {"value": "metago", "label": "MetaGO"},
    {"value": "metaob", "label": "MetaOB"},
    {"value": "metafvg", "label": "MetaFVG"},
    {"value": "metane", "label": "MetaNE"},
    {"value": "metaga", "label": "MetaGA"},
]


def get_filtered_instances(strategy_type):
    """
    Get strategy instances filtered by type.

    Args:
        strategy_type: Strategy type to filter by (e.g., 'metago', 'metafvg') or 'all'

    Returns:
        List of filtered strategy instance names
    """
    all_instances = get_strategy_instances()

    if strategy_type == "all" or not strategy_type:
        return all_instances

    return [inst for inst in all_instances if inst.startswith(strategy_type)]


def _get_default_date(dates):
    """
    Get the most recent business day from available dates.

    Args:
        dates: List of date strings in YYYY-MM-DD format (sorted newest first)

    Returns:
        The most recent business day date string, or None if no dates available
    """
    from datetime import datetime

    if not dates:
        return None

    for date_str in dates:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            # weekday() returns 0-4 for Mon-Fri, 5-6 for Sat-Sun
            if dt.weekday() < 5:
                return date_str
        except ValueError:
            continue

    # If no business day found, return the first available date
    return dates[0] if dates else None


def render_log_tab():
    """
    Render the log viewer tab with strategy and date selectors.

    Returns:
        Dash HTML component with log viewer interface
    """
    # Get available strategy instances (default to all)
    default_type = "all"
    strategy_instances = get_filtered_instances(default_type)

    default_strategy = strategy_instances[0] if strategy_instances else None
    default_dates = get_dates_for_strategy(default_strategy) if default_strategy else []
    default_date = _get_default_date(default_dates)

    return html.Div(
        [
            # Header with icon
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "Strategy Log Viewer",
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
                        "View and analyze execution logs from your trading strategies",
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
                                        id="log-strategy-type-dropdown",
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
                                        id="log-strategy-dropdown",
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
                                        "Log Date",
                                        style={
                                            "fontWeight": "600",
                                            "fontSize": "13px",
                                            "color": "#475569",
                                            "marginBottom": "8px",
                                            "display": "block",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="log-date-dropdown",
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
            # Statistics cards
            html.Div(id="log-stats-container", className="mb-4"),
            # Log content display
            html.Div(
                [
                    # Header bar
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "Log Output",
                                        style={
                                            "fontWeight": "600",
                                            "fontSize": "16px",
                                            "color": "#1e293b",
                                        },
                                    ),
                                    html.Span(
                                        id="log-filename-display",
                                        style={
                                            "color": "#94a3b8",
                                            "fontSize": "13px",
                                            "marginLeft": "12px",
                                        },
                                    ),
                                ],
                                style={"display": "flex", "alignItems": "center"},
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        [
                                            html.Span("Refresh"),
                                        ],
                                        id="log-refresh-btn",
                                        color="light",
                                        size="sm",
                                        style={
                                            "border": "1px solid #e2e8f0",
                                            "marginRight": "8px",
                                        },
                                    ),
                                    dbc.Button(
                                        [
                                            html.Span("Download"),
                                        ],
                                        id="log-download-btn",
                                        color="primary",
                                        size="sm",
                                    ),
                                ],
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "marginBottom": "16px",
                        },
                    ),
                    # Log content area with terminal styling
                    html.Div(
                        id="log-content-display",
                        style={
                            "backgroundColor": "#0f172a",
                            "color": "#e2e8f0",
                            "padding": "24px",
                            "borderRadius": "8px",
                            "fontFamily": "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
                            "fontSize": "12.5px",
                            "lineHeight": "1.7",
                            "maxHeight": "600px",
                            "overflowY": "auto",
                            "whiteSpace": "pre-wrap",
                            "wordBreak": "break-word",
                            "boxShadow": "inset 0 2px 4px rgba(0,0,0,0.2)",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "white",
                    "padding": "24px",
                    "borderRadius": "12px",
                    "border": "1px solid #e2e8f0",
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                },
            ),
            # Hidden download component
            dcc.Download(id="download-log-file"),
        ],
        style={"padding": "0"},
    )


def _format_strategy_label(strategy_instance):
    """Format strategy instance name for display."""
    # Split by underscore and capitalize
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


def create_log_stats_display(stats):
    """
    Create statistics cards for log file.

    Args:
        stats: Dictionary with log statistics

    Returns:
        Dash HTML component with stat cards
    """
    stat_configs = [
        {
            "title": "Total Lines",
            "value": stats.get("total_lines", 0),
            "icon": "#",
            "color": "#3b82f6",
            "bg": "#eff6ff",
        },
        {
            "title": "Updates",
            "value": stats.get("timestamp_markers", 0),
            "icon": "~",
            "color": "#06b6d4",
            "bg": "#ecfeff",
        },
        {
            "title": "Errors",
            "value": stats.get("errors", 0),
            "icon": "!",
            "color": "#ef4444",
            "bg": "#fef2f2",
        },
        {
            "title": "Warnings",
            "value": stats.get("warnings", 0),
            "icon": "?",
            "color": "#f59e0b",
            "bg": "#fffbeb",
        },
    ]

    return dbc.Row(
        [
            dbc.Col(
                create_stat_card(cfg["title"], cfg["value"], cfg["icon"], cfg["color"], cfg["bg"]),
                lg=3,
                md=6,
                sm=6,
                xs=12,
                className="mb-3",
            )
            for cfg in stat_configs
        ],
    )


def create_stat_card(title, value, icon, color, bg_color):
    """
    Create a statistic display card with modern styling.

    Args:
        title: Card title
        value: Value to display
        icon: Icon character
        color: Accent color
        bg_color: Background color

    Returns:
        Dash HTML component
    """
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


def format_log_content(log_content):
    """
    Format log content for nice display with syntax highlighting.

    Args:
        log_content: Raw log content string

    Returns:
        List of formatted HTML components
    """
    if not log_content:
        return html.Div(
            [
                html.Div(
                    "No log content available",
                    style={
                        "fontSize": "16px",
                        "fontWeight": "500",
                        "marginBottom": "8px",
                    },
                ),
                html.Div(
                    "Please select a strategy instance and date to view logs.",
                    style={"fontSize": "13px", "opacity": "0.7"},
                ),
            ],
            style={
                "color": "#94a3b8",
                "textAlign": "center",
                "padding": "60px 40px",
            },
        )

    import re
    entries = parse_log_content(log_content)
    formatted_lines = []
    current_block = []  # Group entries between timestamps

    for i, entry in enumerate(entries):
        msg = entry["message"]
        entry_type = entry.get("type", "log_entry")

        # Style based on entry type and level
        if entry_type == "timestamp_marker":
            # If we have accumulated entries, wrap them
            if current_block:
                formatted_lines.append(
                    html.Div(current_block, style={"marginBottom": "8px"})
                )
                current_block = []

            # Extract timestamp for nice formatting
            ts_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", msg)
            timestamp_str = ts_match.group(1) if ts_match else ""

            formatted_lines.append(
                html.Div(
                    [
                        html.Span(
                            timestamp_str,
                            style={
                                "backgroundColor": "#22d3ee",
                                "color": "#0f172a",
                                "padding": "2px 8px",
                                "borderRadius": "4px",
                                "fontWeight": "600",
                                "fontSize": "11px",
                                "marginRight": "10px",
                            },
                        ),
                        html.Span(
                            "Update",
                            style={
                                "color": "#22d3ee",
                                "fontWeight": "500",
                                "fontSize": "12px",
                            },
                        ),
                    ],
                    style={
                        "marginTop": "16px",
                        "marginBottom": "8px",
                        "paddingBottom": "8px",
                        "borderBottom": "1px solid #334155",
                    },
                )
            )
        elif entry["level"] == "ERROR":
            # Error lines - red with icon
            line_children = [
                html.Span(
                    "ERR",
                    style={
                        "backgroundColor": "rgba(239, 68, 68, 0.2)",
                        "color": "#f87171",
                        "padding": "1px 6px",
                        "borderRadius": "3px",
                        "fontSize": "10px",
                        "fontWeight": "600",
                        "marginRight": "10px",
                    },
                ),
            ]
            line_children.extend(_format_log_line(msg, "#fca5a5"))
            current_block.append(
                html.Div(
                    line_children,
                    style={
                        "padding": "4px 0",
                        "borderLeft": "2px solid #ef4444",
                        "paddingLeft": "12px",
                        "marginLeft": "-2px",
                        "color": "#fca5a5",
                    },
                )
            )
        elif entry["level"] == "WARNING":
            # Warning lines - amber
            line_children = [
                html.Span(
                    "WRN",
                    style={
                        "backgroundColor": "rgba(245, 158, 11, 0.2)",
                        "color": "#fbbf24",
                        "padding": "1px 6px",
                        "borderRadius": "3px",
                        "fontSize": "10px",
                        "fontWeight": "600",
                        "marginRight": "10px",
                    },
                ),
            ]
            line_children.extend(_format_log_line(msg, "#fcd34d"))
            current_block.append(
                html.Div(
                    line_children,
                    style={
                        "padding": "4px 0",
                        "borderLeft": "2px solid #f59e0b",
                        "paddingLeft": "12px",
                        "marginLeft": "-2px",
                        "color": "#fcd34d",
                    },
                )
            )
        elif entry["level"] == "SUCCESS":
            # Success lines - green with highlight
            line_children = [
                html.Span(
                    "OK",
                    style={
                        "backgroundColor": "rgba(34, 197, 94, 0.2)",
                        "color": "#4ade80",
                        "padding": "1px 6px",
                        "borderRadius": "3px",
                        "fontSize": "10px",
                        "fontWeight": "600",
                        "marginRight": "10px",
                    },
                ),
            ]
            line_children.extend(_format_log_line(msg, "#86efac"))
            current_block.append(
                html.Div(
                    line_children,
                    style={
                        "padding": "4px 0",
                        "borderLeft": "2px solid #22c55e",
                        "paddingLeft": "12px",
                        "marginLeft": "-2px",
                        "color": "#86efac",
                    },
                )
            )
        else:
            # Regular lines with syntax highlighting
            current_block.append(
                html.Div(
                    _format_log_line(msg, "#cbd5e1"),
                    style={
                        "color": "#cbd5e1",
                        "padding": "2px 0",
                        "paddingLeft": "12px",
                    },
                )
            )

    # Don't forget the last block
    if current_block:
        formatted_lines.append(html.Div(current_block))

    return html.Div(formatted_lines)


def _format_log_line(msg, default_color):
    """
    Format a log line with syntax highlighting.

    Args:
        msg: Log message string
        default_color: Default text color

    Returns:
        String for simple messages, or list of children for highlighted content
    """
    import re

    # Pattern to match strategy prefix like "metafvg_dax::"
    prefix_match = re.match(r"^(\w+_\w+)::(.*)$", msg)
    if prefix_match:
        prefix = prefix_match.group(1)
        rest = prefix_match.group(2)

        children = [
            html.Span(
                prefix,
                style={
                    "color": "#a78bfa",
                    "fontWeight": "500",
                },
            ),
            html.Span("::", style={"color": "#64748b"}),
        ]

        # Add the rest of the message with value highlighting
        rest_children = _highlight_values_in_text(rest, default_color)
        children.extend(rest_children)

        return children

    # No prefix - just highlight values
    return _highlight_values_in_text(msg, default_color)


def _highlight_values_in_text(text, default_color):
    """
    Highlight numeric values and prices in text.

    Args:
        text: Text to process
        default_color: Default color for non-highlighted text

    Returns:
        List of strings and html.Span elements
    """
    import re

    children = []
    last_end = 0

    # Find all price/number patterns
    for match in re.finditer(r"(\$[\d,]+\.?\d*|\b\d+\.?\d*\b)", text):
        # Add text before match
        if match.start() > last_end:
            children.append(text[last_end:match.start()])

        # Add highlighted number
        value = match.group(1)
        if value.startswith("$"):
            children.append(
                html.Span(
                    value,
                    style={"color": "#fbbf24", "fontWeight": "500"},
                )
            )
        else:
            children.append(
                html.Span(value, style={"color": "#60a5fa"})
            )

        last_end = match.end()

    # Add remaining text
    if last_end < len(text):
        children.append(text[last_end:])

    # If no matches found, return the original text
    if not children:
        return [text]

    return children
