"""
Indicators Tab Component
Visualizes strategy indicator time series overlaid on price data.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import MetaTrader5 as mt5

from .common_ui import COLORS, create_page_header, create_section_card, create_stat_card
from utils.indicator_utils import (
    load_indicator_data,
    list_available_tags,
    classify_indicators,
    PRICE_LEVEL_INDICATORS,
)

# Indicator display colors
INDICATOR_COLORS = {
    "sma_target": "#f59e0b",
    "bb_upper": "#8b5cf6",
    "bb_sma": "#6366f1",
    "bb_lower": "#8b5cf6",
    "tp": "#22c55e",
    "sl": "#ef4444",
    "entry": "#3b82f6",
    "exit": "#ec4899",
    "price": "#1e293b",
    "pred_h4": "#3b82f6",
    "pred_h6": "#22c55e",
    "pred_h8": "#f59e0b",
    "vote": "#8b5cf6",
    "quorum": "#06b6d4",
    "half_life": "#ec4899",
    "bb_period": "#6366f1",
    "state": "#ef4444",
}

# Subplot indicator groups (ordered)
PREDICTION_INDICATORS = {"pred_h4", "pred_h6", "pred_h8"}
STATE_INDICATORS = {"state"}
EVOLUTION_INDICATORS = {"half_life", "bb_period"}


def render_indicators_tab():
    """
    Returns the initial layout for the Indicators tab.
    Dropdowns are populated via callbacks.
    """
    return html.Div(
        [
            create_page_header(
                "Strategy Indicators",
                "Visualize saved indicator time series overlaid on price data",
            ),
            # Controls row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                "Strategy Tag",
                                style={
                                    "fontWeight": "600",
                                    "fontSize": "13px",
                                    "color": COLORS["text_medium"],
                                    "marginBottom": "8px",
                                    "display": "block",
                                },
                            ),
                            dcc.Dropdown(
                                id="indicators-tag-dropdown",
                                options=[],
                                value=None,
                                placeholder="Select a strategy tag...",
                                style={"fontSize": "14px"},
                                clearable=False,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label(
                                "Date Range",
                                style={
                                    "fontWeight": "600",
                                    "fontSize": "13px",
                                    "color": COLORS["text_medium"],
                                    "marginBottom": "8px",
                                    "display": "block",
                                },
                            ),
                            dcc.DatePickerRange(
                                id="indicators-date-range",
                                display_format="YYYY-MM-DD",
                                style={"fontSize": "14px"},
                            ),
                        ],
                        md=5,
                    ),
                    dbc.Col(
                        [
                            html.Label(
                                "\u00a0",
                                style={
                                    "display": "block",
                                    "marginBottom": "8px",
                                    "fontSize": "13px",
                                },
                            ),
                            dbc.Button(
                                "Refresh",
                                id="indicators-refresh-btn",
                                color="primary",
                                size="sm",
                                style={"width": "100%"},
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="mb-4",
            ),
            # Latest values cards
            html.Div(id="indicators-info-cards"),
            # Chart area
            html.Div(
                id="indicators-chart-container",
                children=[
                    html.Div(
                        "Select a strategy tag to view indicators.",
                        style={
                            "color": COLORS["text_light"],
                            "textAlign": "center",
                            "padding": "60px 20px",
                            "fontSize": "14px",
                        },
                    )
                ],
            ),
        ],
        style={"padding": "24px"},
    )


def pull_price_for_indicators(symbol, df_indicators):
    """
    Fetch OHLC data from MT5 covering the indicator time range.

    Args:
        symbol: MT5 symbol string
        df_indicators: DataFrame with 'timestamp' column

    Returns:
        pd.DataFrame with OHLC data or None
    """
    if df_indicators is None or df_indicators.empty:
        return None
    if "timestamp" not in df_indicators.columns:
        return None

    start = df_indicators["timestamp"].min()
    end = df_indicators["timestamp"].max()

    try:
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start, end)
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df
    except Exception:
        return None


def create_indicator_chart(tag, df_indicators, df_price=None):
    """
    Build a multi-subplot Plotly figure with indicators and optional price data.

    Args:
        tag: Strategy tag string
        df_indicators: DataFrame of indicator values with 'timestamp' column
        df_price: Optional OHLC DataFrame from MT5

    Returns:
        plotly.graph_objects.Figure
    """
    if df_indicators is None or df_indicators.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No indicator data available for this tag.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS["text_light"]),
        )
        fig.update_layout(
            template="plotly_white",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    classified = classify_indicators(df_indicators.columns)
    overlay_cols = classified["overlay"]
    subplot_cols = classified["subplot"]

    # Determine subplot structure
    has_predictions = bool(PREDICTION_INDICATORS & set(subplot_cols))
    has_state = bool(STATE_INDICATORS & set(subplot_cols))
    has_evolution = bool(EVOLUTION_INDICATORS & set(subplot_cols))

    # Other subplots (vote, quorum, etc.)
    other_cols = [
        c for c in subplot_cols
        if c not in PREDICTION_INDICATORS
        and c not in STATE_INDICATORS
        and c not in EVOLUTION_INDICATORS
    ]
    has_other = bool(other_cols)

    n_subplots = 1  # price + overlay always
    subplot_labels = ["Price & Overlays"]
    row_heights = [0.5]

    if has_predictions:
        n_subplots += 1
        subplot_labels.append("Predictions")
        row_heights.append(0.15)
    if has_state:
        n_subplots += 1
        subplot_labels.append("State")
        row_heights.append(0.1)
    if has_evolution:
        n_subplots += 1
        subplot_labels.append("Half-life / BB Period")
        row_heights.append(0.12)
    if has_other:
        n_subplots += 1
        subplot_labels.append("Other Indicators")
        row_heights.append(0.13)

    # Normalize heights
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=subplot_labels,
    )

    timestamps = df_indicators["timestamp"]

    # --- Row 1: Price + Overlays ---
    # Add price data if available
    if df_price is not None and not df_price.empty:
        fig.add_trace(
            go.Candlestick(
                x=df_price["time"],
                open=df_price["open"],
                high=df_price["high"],
                low=df_price["low"],
                close=df_price["close"],
                name="Price",
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
                showlegend=True,
            ),
            row=1, col=1,
        )

    # Add overlay indicators
    for col in overlay_cols:
        if col not in df_indicators.columns:
            continue
        color = INDICATOR_COLORS.get(col, "#64748b")
        dash_style = "dash" if col in ("tp", "sl") else ("dot" if col in ("bb_upper", "bb_lower") else "solid")
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=pd.to_numeric(df_indicators[col], errors="coerce"),
                mode="lines+markers",
                name=col,
                line=dict(color=color, width=1.5, dash=dash_style),
                marker=dict(size=3),
            ),
            row=1, col=1,
        )

    # --- Subplot rows ---
    current_row = 2

    # Predictions subplot
    if has_predictions:
        for col in sorted(PREDICTION_INDICATORS & set(subplot_cols)):
            if col not in df_indicators.columns:
                continue
            color = INDICATOR_COLORS.get(col, "#64748b")
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=pd.to_numeric(df_indicators[col], errors="coerce"),
                    mode="lines+markers",
                    name=col,
                    line=dict(color=color, width=1.5),
                    marker=dict(size=3),
                ),
                row=current_row, col=1,
            )
        fig.update_yaxes(title_text="Pred", row=current_row, col=1)
        current_row += 1

    # State subplot
    if has_state:
        if "state" in df_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=pd.to_numeric(df_indicators["state"], errors="coerce"),
                    mode="lines",
                    name="state",
                    line=dict(
                        color=INDICATOR_COLORS.get("state", "#ef4444"),
                        width=2,
                        shape="hv",
                    ),
                    fill="tozeroy",
                    fillcolor="rgba(239,68,68,0.1)",
                ),
                row=current_row, col=1,
            )
        fig.update_yaxes(
            title_text="State",
            tickvals=[-2, -1, 0, 1],
            row=current_row, col=1,
        )
        current_row += 1

    # Evolution subplot (half_life, bb_period)
    if has_evolution:
        for col in sorted(EVOLUTION_INDICATORS & set(subplot_cols)):
            if col not in df_indicators.columns:
                continue
            color = INDICATOR_COLORS.get(col, "#64748b")
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=pd.to_numeric(df_indicators[col], errors="coerce"),
                    mode="lines+markers",
                    name=col,
                    line=dict(color=color, width=1.5),
                    marker=dict(size=3),
                ),
                row=current_row, col=1,
            )
        fig.update_yaxes(title_text="Period", row=current_row, col=1)
        current_row += 1

    # Other indicators subplot
    if has_other:
        other_palette = ["#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6", "#06b6d4", "#ec4899"]
        for i, col in enumerate(other_cols):
            if col not in df_indicators.columns:
                continue
            color = INDICATOR_COLORS.get(col, other_palette[i % len(other_palette)])
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=pd.to_numeric(df_indicators[col], errors="coerce"),
                    mode="lines+markers",
                    name=col,
                    line=dict(color=color, width=1.5),
                    marker=dict(size=3),
                ),
                row=current_row, col=1,
            )
        fig.update_yaxes(title_text="Value", row=current_row, col=1)

    # Global layout
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=COLORS["text_medium"]),
        title=dict(
            text=f"Indicators: {tag}",
            font=dict(size=16, color=COLORS["text_dark"]),
            x=0, xanchor="left",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=30, t=60, b=40),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=200 + n_subplots * 180,
        xaxis_rangeslider_visible=False,
    )

    # Style all axes
    for i in range(1, n_subplots + 1):
        fig.update_xaxes(
            gridcolor=COLORS["border_light"],
            zerolinecolor=COLORS["border"],
            tickfont=dict(size=11),
            row=i, col=1,
        )
        fig.update_yaxes(
            gridcolor=COLORS["border_light"],
            zerolinecolor=COLORS["border"],
            tickfont=dict(size=11),
            row=i, col=1,
        )

    return fig


def create_indicator_info_cards(df_indicators):
    """
    Create info cards showing latest indicator values.

    Args:
        df_indicators: DataFrame with indicator data

    Returns:
        Dash component with stat cards
    """
    if df_indicators is None or df_indicators.empty:
        return html.Div()

    latest = df_indicators.iloc[-1]
    cards = []

    # Timestamp card
    ts = latest.get("timestamp", "N/A")
    if pd.notna(ts):
        ts_str = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M")
    else:
        ts_str = "N/A"
    cards.append(
        dbc.Col(create_stat_card("Last Update", ts_str, color="primary"), md=3)
    )

    # State card
    if "state" in latest.index and pd.notna(latest["state"]):
        state_val = int(float(latest["state"]))
        state_map = {1: "LONG", -1: "SHORT", 0: "NEUTRAL", -2: "EXIT"}
        state_color = {1: "success", -1: "danger", 0: "neutral", -2: "warning"}
        cards.append(
            dbc.Col(
                create_stat_card(
                    "State",
                    state_map.get(state_val, str(state_val)),
                    color=state_color.get(state_val, "neutral"),
                ),
                md=3,
            )
        )

    # Price card
    if "price" in latest.index and pd.notna(latest["price"]):
        cards.append(
            dbc.Col(
                create_stat_card("Price", f"{float(latest['price']):.5f}", color="cyan"),
                md=3,
            )
        )

    # Rows count
    cards.append(
        dbc.Col(
            create_stat_card("Data Points", str(len(df_indicators)), color="purple"),
            md=3,
        )
    )

    return dbc.Row(cards, className="mb-4")
