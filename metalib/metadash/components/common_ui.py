"""
Common UI Components
Reusable styled components for consistent UI across all tabs.
"""
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


# Color palette
COLORS = {
    "primary": "#3b82f6",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "purple": "#8b5cf6",
    "cyan": "#06b6d4",
    "pink": "#ec4899",
    "indigo": "#6366f1",
    "text_dark": "#1e293b",
    "text_medium": "#475569",
    "text_light": "#94a3b8",
    "background": "#f8fafc",
    "card_bg": "#ffffff",
    "border": "#e2e8f0",
    "border_light": "#f1f5f9",
}

# Chart colors
CHART_COLORS = [
    "#3b82f6",  # blue
    "#22c55e",  # green
    "#f59e0b",  # amber
    "#ef4444",  # red
    "#8b5cf6",  # purple
    "#06b6d4",  # cyan
    "#ec4899",  # pink
    "#6366f1",  # indigo
]


def create_page_header(title, description=None):
    """
    Create a consistent page header with title and optional description.

    Args:
        title: Page title
        description: Optional description text

    Returns:
        Dash HTML component
    """
    children = [
        html.Span(
            title,
            style={
                "fontSize": "24px",
                "fontWeight": "700",
                "color": COLORS["text_dark"],
            },
        ),
    ]

    if description:
        children.append(
            html.P(
                description,
                style={
                    "color": COLORS["text_medium"],
                    "marginTop": "8px",
                    "marginBottom": "0",
                    "fontSize": "14px",
                },
            )
        )

    return html.Div(
        children,
        style={
            "marginBottom": "24px",
            "paddingBottom": "16px",
            "borderBottom": f"1px solid {COLORS['border']}",
        },
    )


def create_stat_card(title, value, subtitle=None, color="primary", icon=None):
    """
    Create a styled statistics card.

    Args:
        title: Card title (small text above value)
        value: Main value to display
        subtitle: Optional subtitle below value
        color: Color theme (primary, success, warning, danger, purple, cyan)
        icon: Optional icon character

    Returns:
        Dash HTML component
    """
    color_map = {
        "primary": {"main": "#3b82f6", "bg": "#eff6ff", "border": "#bfdbfe"},
        "success": {"main": "#22c55e", "bg": "#f0fdf4", "border": "#86efac"},
        "warning": {"main": "#f59e0b", "bg": "#fffbeb", "border": "#fcd34d"},
        "danger": {"main": "#ef4444", "bg": "#fef2f2", "border": "#fca5a5"},
        "purple": {"main": "#8b5cf6", "bg": "#f5f3ff", "border": "#c4b5fd"},
        "cyan": {"main": "#06b6d4", "bg": "#ecfeff", "border": "#67e8f9"},
        "neutral": {"main": "#64748b", "bg": "#f8fafc", "border": "#cbd5e1"},
    }

    colors = color_map.get(color, color_map["primary"])

    icon_element = None
    if icon:
        icon_element = html.Div(
            icon,
            style={
                "width": "40px",
                "height": "40px",
                "borderRadius": "10px",
                "backgroundColor": colors["bg"],
                "color": colors["main"],
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "fontWeight": "700",
                "fontSize": "18px",
                "marginBottom": "12px",
            },
        )

    subtitle_element = None
    if subtitle:
        # Determine subtitle color based on content
        subtitle_color = COLORS["text_light"]
        if isinstance(subtitle, str):
            if subtitle.startswith("+"):
                subtitle_color = COLORS["success"]
            elif subtitle.startswith("-"):
                subtitle_color = COLORS["danger"]

        subtitle_element = html.Div(
            subtitle,
            style={
                "fontSize": "12px",
                "color": subtitle_color,
                "marginTop": "4px",
                "fontWeight": "500",
            },
        )

    return html.Div(
        [
            icon_element,
            html.Div(
                title,
                style={
                    "fontSize": "12px",
                    "color": COLORS["text_light"],
                    "fontWeight": "600",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.5px",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "28px",
                    "fontWeight": "700",
                    "color": COLORS["text_dark"],
                    "lineHeight": "1.2",
                    "marginTop": "4px",
                },
            ),
            subtitle_element,
        ],
        style={
            "padding": "20px",
            "borderRadius": "12px",
            "backgroundColor": "white",
            "border": f"1px solid {COLORS['border']}",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
            "borderLeft": f"4px solid {colors['main']}",
        },
    )


def create_section_card(title, children, subtitle=None):
    """
    Create a styled section card wrapper.

    Args:
        title: Section title
        children: Content to wrap
        subtitle: Optional subtitle/description

    Returns:
        Dash HTML component
    """
    header_children = [
        html.Span(
            title,
            style={
                "fontSize": "16px",
                "fontWeight": "600",
                "color": COLORS["text_dark"],
            },
        ),
    ]

    if subtitle:
        header_children.append(
            html.Span(
                subtitle,
                style={
                    "fontSize": "13px",
                    "color": COLORS["text_light"],
                    "marginLeft": "12px",
                },
            )
        )

    return html.Div(
        [
            html.Div(
                header_children,
                style={
                    "marginBottom": "16px",
                    "paddingBottom": "12px",
                    "borderBottom": f"1px solid {COLORS['border_light']}",
                },
            ),
            html.Div(children),
        ],
        style={
            "padding": "24px",
            "borderRadius": "12px",
            "backgroundColor": "white",
            "border": f"1px solid {COLORS['border']}",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
            "marginBottom": "20px",
        },
    )


def create_mini_stat(label, value, color="neutral"):
    """
    Create a compact inline stat display.

    Args:
        label: Stat label
        value: Stat value
        color: Color theme

    Returns:
        Dash HTML component
    """
    color_map = {
        "success": COLORS["success"],
        "danger": COLORS["danger"],
        "warning": COLORS["warning"],
        "neutral": COLORS["text_dark"],
    }

    return html.Div(
        [
            html.Span(
                label,
                style={
                    "fontSize": "12px",
                    "color": COLORS["text_light"],
                    "marginRight": "8px",
                },
            ),
            html.Span(
                value,
                style={
                    "fontSize": "14px",
                    "fontWeight": "600",
                    "color": color_map.get(color, COLORS["text_dark"]),
                },
            ),
        ],
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "padding": "8px 12px",
            "backgroundColor": COLORS["background"],
            "borderRadius": "6px",
            "marginRight": "8px",
        },
    )


def create_styled_dropdown(id, options, value, placeholder="Select...", label=None):
    """
    Create a styled dropdown with optional label.

    Args:
        id: Component ID
        options: Dropdown options
        value: Default value
        placeholder: Placeholder text
        label: Optional label above dropdown

    Returns:
        Dash HTML component
    """
    children = []

    if label:
        children.append(
            html.Label(
                label,
                style={
                    "fontWeight": "600",
                    "fontSize": "13px",
                    "color": COLORS["text_medium"],
                    "marginBottom": "8px",
                    "display": "block",
                },
            )
        )

    children.append(
        dcc.Dropdown(
            id=id,
            options=options,
            value=value,
            placeholder=placeholder,
            style={"fontSize": "14px"},
            clearable=False,
        )
    )

    return html.Div(children)


def create_chip_selector(id, options, value):
    """
    Create a chip/toggle style multi-selector.

    Args:
        id: Component ID
        options: List of {'label': str, 'value': str} dicts
        value: List of selected values

    Returns:
        Dash HTML component
    """
    return dcc.Checklist(
        id=id,
        options=options,
        value=value,
        inline=True,
        inputStyle={"display": "none"},
        labelStyle={
            "display": "inline-block",
            "padding": "8px 16px",
            "margin": "4px",
            "borderRadius": "20px",
            "border": f"1px solid {COLORS['border']}",
            "backgroundColor": "white",
            "cursor": "pointer",
            "fontSize": "13px",
            "fontWeight": "500",
            "transition": "all 0.2s",
        },
        inputClassName="chip-input",
        labelClassName="chip-label",
    )


def create_styled_table(data, columns, id=None, sort_action="native"):
    """
    Create a modern styled DataTable.

    Args:
        data: Table data (list of dicts)
        columns: Column definitions
        id: Optional component ID
        sort_action: Sort action type

    Returns:
        Dash DataTable component
    """
    table_props = {
        "data": data,
        "columns": columns,
        "sort_action": sort_action,
        "style_cell": {
            "textAlign": "left",
            "padding": "12px 16px",
            "fontSize": "13px",
            "fontFamily": "Inter, system-ui, sans-serif",
            "border": "none",
            "borderBottom": f"1px solid {COLORS['border_light']}",
        },
        "style_header": {
            "backgroundColor": COLORS["background"],
            "fontWeight": "600",
            "color": COLORS["text_medium"],
            "borderBottom": f"2px solid {COLORS['border']}",
            "textTransform": "uppercase",
            "fontSize": "11px",
            "letterSpacing": "0.5px",
        },
        "style_data": {
            "backgroundColor": "white",
            "color": COLORS["text_dark"],
        },
        "style_data_conditional": [
            {
                "if": {"row_index": "odd"},
                "backgroundColor": COLORS["background"],
            },
            {
                "if": {"column_id": "Total Profit", "filter_query": "{Total Profit} > 0"},
                "color": COLORS["success"],
                "fontWeight": "600",
            },
            {
                "if": {"column_id": "Total Profit", "filter_query": "{Total Profit} < 0"},
                "color": COLORS["danger"],
                "fontWeight": "600",
            },
            {
                "if": {"column_id": "total_profit", "filter_query": "{total_profit} > 0"},
                "color": COLORS["success"],
                "fontWeight": "600",
            },
            {
                "if": {"column_id": "total_profit", "filter_query": "{total_profit} < 0"},
                "color": COLORS["danger"],
                "fontWeight": "600",
            },
        ],
        "style_table": {
            "borderRadius": "8px",
            "overflow": "hidden",
            "border": f"1px solid {COLORS['border']}",
        },
    }

    if id:
        table_props["id"] = id

    return dash_table.DataTable(**table_props)


def style_plotly_chart(fig, height=400):
    """
    Apply consistent styling to a Plotly figure.

    Args:
        fig: Plotly figure object
        height: Chart height in pixels

    Returns:
        Styled Plotly figure
    """
    fig.update_layout(
        template="plotly_white",
        font=dict(
            family="Inter, system-ui, sans-serif",
            size=12,
            color=COLORS["text_medium"],
        ),
        title=dict(
            font=dict(size=16, color=COLORS["text_dark"]),
            x=0,
            xanchor="left",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(size=11),
        ),
        height=height,
    )

    fig.update_xaxes(
        gridcolor=COLORS["border_light"],
        zerolinecolor=COLORS["border"],
        tickfont=dict(size=11),
    )

    fig.update_yaxes(
        gridcolor=COLORS["border_light"],
        zerolinecolor=COLORS["border"],
        tickfont=dict(size=11),
    )

    return fig


def format_currency(value, show_sign=False):
    """Format a number as currency."""
    if value is None:
        return "N/A"
    sign = "+" if show_sign and value > 0 else ""
    return f"{sign}${value:,.2f}"


def format_percentage(value, show_sign=False):
    """Format a number as percentage."""
    if value is None:
        return "N/A"
    sign = "+" if show_sign and value > 0 else ""
    return f"{sign}{value:.1f}%"
