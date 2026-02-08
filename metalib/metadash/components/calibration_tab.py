"""
Calibration Tab Component
Provides UI for running MetaScale weight optimization and saving results to YAML configs.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory for metalib imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from .common_ui import COLORS

# Map common_ui COLORS to the keys we use
COLORS = {
    **COLORS,
    "text_primary": COLORS.get("text_dark", "#1e293b"),
    "text_secondary": COLORS.get("text_medium", "#475569"),
    "bg_secondary": COLORS.get("background", "#f8fafc"),
}

# Import defaults from metascal - add "enabled" flag for UI
try:
    from metalib.metascal import DEFAULT_STRATEGY_PARAMS as _METASCAL_DEFAULTS
    # Add enabled flag (all True by default except metamlp)
    DEFAULT_STRATEGY_PARAMS = {
        k: {**v, "enabled": True}
        for k, v in _METASCAL_DEFAULTS.items()
    }
    # Add metamlp as disabled by default (not in metascal defaults yet)
    if "metamlp" not in DEFAULT_STRATEGY_PARAMS:
        DEFAULT_STRATEGY_PARAMS["metamlp"] = {"numerator": 10, "trades_per_day": 24, "enabled": False}
except ImportError:
    # Fallback if metascal not importable
    DEFAULT_STRATEGY_PARAMS = {
        "metafvg": {"numerator": 20, "trades_per_day": 48, "enabled": True},
        "metago": {"numerator": 1.5, "trades_per_day": 1, "enabled": True},
        "metane": {"numerator": 10, "trades_per_day": 12, "enabled": True},
        "metaga": {"numerator": 30, "trades_per_day": 48, "enabled": True},
        "metaob": {"numerator": 1, "trades_per_day": 1, "enabled": True},
        "metamlp": {"numerator": 10, "trades_per_day": 24, "enabled": False},
    }

STRATEGY_DISPLAY_NAMES = {
    "metafvg": "Fair Value Gaps",
    "metago": "Monthly True Open",
    "metane": "Seasonality DT",
    "metaga": "XGBoost DT",
    "metaob": "Order Blocks",
    "metamlp": "Multi-Horizon MLP",
}


def create_strategy_weight_row(strategy_key: str, params: dict) -> dbc.Row:
    """Create a row for configuring a single strategy's weight parameters."""
    display_name = STRATEGY_DISPLAY_NAMES.get(strategy_key, strategy_key)

    return dbc.Row(
        [
            # Enable/disable toggle
            dbc.Col(
                dbc.Checkbox(
                    id={"type": "calib-strategy-enabled", "index": strategy_key},
                    value=params["enabled"],
                    className="mt-2",
                ),
                width=1,
                className="d-flex align-items-center",
            ),
            # Strategy name
            dbc.Col(
                html.Span(
                    display_name,
                    style={
                        "fontWeight": "500",
                        "fontSize": "14px",
                        "color": COLORS["text_primary"],
                    },
                ),
                width=3,
                className="d-flex align-items-center",
            ),
            # Numerator input
            dbc.Col(
                [
                    html.Label(
                        "Numerator",
                        style={"fontSize": "11px", "color": COLORS["text_secondary"]},
                    ),
                    dbc.Input(
                        id={"type": "calib-numerator", "index": strategy_key},
                        type="number",
                        value=params["numerator"],
                        min=0.1,
                        step=0.1,
                        size="sm",
                        style={"width": "80px"},
                    ),
                ],
                width=2,
            ),
            # Trades per day input
            dbc.Col(
                [
                    html.Label(
                        "Trades/Day",
                        style={"fontSize": "11px", "color": COLORS["text_secondary"]},
                    ),
                    dbc.Input(
                        id={"type": "calib-trades-per-day", "index": strategy_key},
                        type="number",
                        value=params["trades_per_day"],
                        min=1,
                        step=1,
                        size="sm",
                        style={"width": "80px"},
                    ),
                ],
                width=2,
            ),
            # Computed weight display
            dbc.Col(
                [
                    html.Label(
                        "Weight",
                        style={"fontSize": "11px", "color": COLORS["text_secondary"]},
                    ),
                    html.Div(
                        id={"type": "calib-weight-display", "index": strategy_key},
                        children=f"{params['numerator'] / np.sqrt(params['trades_per_day']):.3f}",
                        style={
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "color": COLORS["primary"],
                            "padding": "4px 8px",
                            "backgroundColor": COLORS["bg_secondary"],
                            "borderRadius": "4px",
                            "display": "inline-block",
                        },
                    ),
                ],
                width=2,
            ),
        ],
        className="mb-3 py-2",
        style={
            "borderBottom": f"1px solid {COLORS['border']}",
        },
    )


def create_calibration_controls() -> dbc.Card:
    """Create the main calibration controls panel."""
    strategy_rows = [
        create_strategy_weight_row(key, params)
        for key, params in DEFAULT_STRATEGY_PARAMS.items()
    ]

    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5(
                    "Strategy Weight Configuration",
                    className="mb-0",
                    style={"fontWeight": "600", "fontSize": "16px"},
                ),
                style={"backgroundColor": COLORS["bg_secondary"]},
            ),
            dbc.CardBody(
                [
                    # Header row
                    dbc.Row(
                        [
                            dbc.Col(html.Strong("On"), width=1),
                            dbc.Col(html.Strong("Strategy"), width=3),
                            dbc.Col(html.Strong("Numerator"), width=2),
                            dbc.Col(html.Strong("Trades/Day"), width=2),
                            dbc.Col(html.Strong("Weight = N/sqrt(T)"), width=2),
                        ],
                        className="mb-2 pb-2",
                        style={
                            "borderBottom": f"2px solid {COLORS['border']}",
                            "fontSize": "12px",
                            "color": COLORS["text_secondary"],
                        },
                    ),
                    # Strategy rows
                    *strategy_rows,
                ]
            ),
        ],
        className="mb-4",
        style={"border": f"1px solid {COLORS['border']}"},
    )


def create_optimization_params() -> dbc.Card:
    """Create the optimization parameters panel."""
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5(
                    "Optimization Parameters",
                    className="mb-0",
                    style={"fontWeight": "600", "fontSize": "16px"},
                ),
                style={"backgroundColor": COLORS["bg_secondary"]},
            ),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Risk Percentage (%)",
                                        style={"fontWeight": "500"},
                                    ),
                                    dbc.Input(
                                        id="calib-risk-pct",
                                        type="number",
                                        value=1.5,
                                        min=0.1,
                                        max=10.0,
                                        step=0.1,
                                        style={"width": "120px"},
                                    ),
                                    html.Small(
                                        "Percentage of account balance to risk",
                                        className="text-muted",
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Lookback Days",
                                        style={"fontWeight": "500"},
                                    ),
                                    dbc.Input(
                                        id="calib-lookback-days",
                                        type="number",
                                        value=92,
                                        min=30,
                                        max=365,
                                        step=1,
                                        style={"width": "120px"},
                                    ),
                                    html.Small(
                                        "Days of price history for covariance",
                                        className="text-muted",
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Config Directory",
                                        style={"fontWeight": "500"},
                                    ),
                                    dbc.Input(
                                        id="calib-config-dir",
                                        type="text",
                                        value="../config/prod",
                                        style={"width": "200px"},
                                    ),
                                    html.Small(
                                        "Path to YAML config files",
                                        className="text-muted",
                                    ),
                                ],
                                width=4,
                            ),
                        ],
                    ),
                    html.Hr(),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-play me-2"),
                                        "Run Optimization",
                                    ],
                                    id="calib-run-btn",
                                    color="primary",
                                    size="lg",
                                    className="me-3",
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                dbc.Spinner(
                                    html.Div(id="calib-status-msg"),
                                    color="primary",
                                    size="sm",
                                ),
                                className="d-flex align-items-center",
                            ),
                        ],
                        className="mt-3",
                    ),
                ],
            ),
        ],
        className="mb-4",
        style={"border": f"1px solid {COLORS['border']}"},
    )


def create_results_section() -> html.Div:
    """Create the results display section (populated by callback)."""
    return html.Div(
        [
            # Results table container
            html.Div(id="calib-results-table-container"),
            # Results chart container
            html.Div(id="calib-results-chart-container", className="mt-4"),
            # Save button (hidden until results available)
            html.Div(
                id="calib-save-section",
                children=[
                    html.Hr(),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-save me-2"),
                                        "Save to YAML Configs",
                                    ],
                                    id="calib-save-btn",
                                    color="success",
                                    size="lg",
                                    disabled=True,
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                html.Div(id="calib-save-feedback"),
                                className="d-flex align-items-center",
                            ),
                        ],
                    ),
                ],
                style={"display": "none"},
            ),
        ],
    )


def create_results_table(weights_df: pd.DataFrame) -> dbc.Table:
    """Create the results table from optimization output."""
    if weights_df is None or weights_df.empty:
        return html.Div("No results available", className="text-muted text-center p-4")

    # Build table header
    header = html.Thead(
        html.Tr(
            [
                html.Th("Strategy"),
                html.Th("Symbol"),
                html.Th("Old Size"),
                html.Th("New Size"),
                html.Th("Change"),
            ]
        )
    )

    # Build table rows
    rows = []
    for _, row in weights_df.iterrows():
        old_size = row.get("old_size", 0)
        new_size = row.get("new_size", 0)
        change = new_size - old_size
        change_pct = (change / old_size * 100) if old_size > 0 else 0

        change_color = "green" if change > 0 else "red" if change < 0 else "gray"
        change_text = f"{change:+.2f} ({change_pct:+.1f}%)"

        rows.append(
            html.Tr(
                [
                    html.Td(row["strategy_type"]),
                    html.Td(row["symbol"]),
                    html.Td(f"{old_size:.2f}"),
                    html.Td(f"{new_size:.2f}", style={"fontWeight": "600"}),
                    html.Td(change_text, style={"color": change_color}),
                ]
            )
        )

    body = html.Tbody(rows)

    return dbc.Table(
        [header, body],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        size="sm",
    )


def create_results_chart(weights_df: pd.DataFrame) -> dcc.Graph:
    """Create bar chart comparing old vs new position sizes."""
    if weights_df is None or weights_df.empty:
        return html.Div()

    # Create labels
    labels = [f"{row['strategy_type']}\n{row['symbol']}"
              for _, row in weights_df.iterrows()]

    fig = go.Figure()

    # Old sizes
    fig.add_trace(
        go.Bar(
            name="Old Size",
            x=labels,
            y=weights_df["old_size"],
            marker_color=COLORS["text_secondary"],
            opacity=0.6,
        )
    )

    # New sizes
    fig.add_trace(
        go.Bar(
            name="New Size",
            x=labels,
            y=weights_df["new_size"],
            marker_color=COLORS["primary"],
        )
    )

    fig.update_layout(
        title="Position Size Comparison (Old vs New)",
        xaxis_title="Strategy / Symbol",
        yaxis_title="Position Size (lots)",
        barmode="group",
        template="plotly_white",
        height=400,
        margin=dict(l=60, r=40, t=60, b=120),
        xaxis_tickangle=-45,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def render_calibration_tab() -> html.Div:
    """Render the complete calibration tab."""
    return html.Div(
        [
            # Tab header
            html.Div(
                [
                    html.H4(
                        "Weight Calibration",
                        style={
                            "fontWeight": "600",
                            "color": COLORS["text_primary"],
                            "marginBottom": "8px",
                        },
                    ),
                    html.P(
                        "Configure strategy weights and run portfolio optimization to compute "
                        "optimal position sizes. Results are saved to YAML config files.",
                        style={
                            "color": COLORS["text_secondary"],
                            "fontSize": "14px",
                            "marginBottom": "24px",
                        },
                    ),
                ],
            ),
            # Strategy weights configuration
            create_calibration_controls(),
            # Optimization parameters
            create_optimization_params(),
            # Results section (populated by callback)
            create_results_section(),
            # Store for optimization results
            dcc.Store(id="calib-results-store"),
        ],
        style={"padding": "24px"},
    )
