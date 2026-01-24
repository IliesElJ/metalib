"""
Strategy Type Performance Tab Component
"""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from utils.metrics import (
    calculate_strategy_type_metrics,
    calculate_strategy_type_cumulative,
)


def render_strategy_type_tab(merged_deals, account_size):
    """
    Render the strategy type performance tab with aggregated metrics and cumulative charts.

    Args:
        merged_deals: DataFrame with merged trading deals
        account_size: Account size for calculations

    Returns:
        Dash HTML component with strategy type performance analysis
    """
    # Calculate strategy type metrics
    strategy_type_metrics = calculate_strategy_type_metrics(merged_deals, account_size)

    # Calculate cumulative performance
    cumulative_data = calculate_strategy_type_cumulative(merged_deals)

    # Create visualizations
    metrics_table = create_strategy_type_metrics_table(strategy_type_metrics)
    comparison_chart = create_strategy_type_comparison_chart(strategy_type_metrics)
    cumulative_chart = create_cumulative_performance_chart(cumulative_data)

    return html.Div(
        [
            # Header section
            html.Div(
                [
                    html.H3(
                        "Strategy Type Performance Analysis", className="section-title"
                    ),
                    html.P(
                        "Aggregated performance metrics by strategy type (extracted from the first word before '_' in tags)",
                        className="section-description",
                    ),
                ],
                className="mb-4",
            ),
            # Metrics table
            html.Div(
                [
                    html.H4(
                        "Performance Metrics by Strategy Type",
                        className="subsection-title",
                    ),
                    metrics_table,
                ],
                className="mb-4",
            ),
            # Comparison charts
            html.Div(
                [
                    html.H4("Strategy Type Comparison", className="subsection-title"),
                    dcc.Graph(figure=comparison_chart, className="graph-container"),
                ],
                className="mb-4",
            ),
            # Cumulative performance
            html.Div(
                [
                    html.H4(
                        "Cumulative Performance Over Time", className="subsection-title"
                    ),
                    dcc.Graph(figure=cumulative_chart, className="graph-container"),
                ],
                className="mb-4",
            ),
        ]
    )


def create_strategy_type_metrics_table(strategy_type_metrics):
    """
    Create a formatted metrics table for strategy types.

    Args:
        strategy_type_metrics: DataFrame with strategy type metrics

    Returns:
        Dash DataTable component
    """
    df = strategy_type_metrics.reset_index()

    # Sort by Total Profit descending
    df = df.sort_values("Total Profit", ascending=False)

    # Format Last Trade datetime column
    if "Last Trade" in df.columns:
        df["Last Trade"] = df["Last Trade"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if not pd.isna(x) else ""
        )

    # Format numeric columns
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns

    columns = []
    for col in df.columns:
        if col in numeric_columns and col not in ["Strategy Type"]:
            columns.append(
                {
                    "name": col,
                    "id": col,
                    "type": "numeric",
                    "format": {"specifier": ".2f"},
                }
            )
        else:
            columns.append({"name": col, "id": col})

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=columns,
        style_cell={
            "textAlign": "left",
            "padding": "12px",
            "fontSize": "14px",
        },
        style_data_conditional=[
            {
                "if": {
                    "column_id": "Total Profit",
                    "filter_query": "{Total Profit} > 0",
                },
                "color": "#28a745",
                "fontWeight": "bold",
            },
            {
                "if": {
                    "column_id": "Total Profit",
                    "filter_query": "{Total Profit} < 0",
                },
                "color": "#dc3545",
                "fontWeight": "bold",
            },
            {"if": {"row_index": "odd"}, "backgroundColor": "rgba(0, 0, 0, 0.02)"},
        ],
        style_header={
            "backgroundColor": "rgba(0, 102, 204, 0.1)",
            "fontWeight": "bold",
            "textAlign": "center",
            "fontSize": "14px",
            "padding": "12px",
        },
        style_table={"overflowX": "auto"},
    )


def create_strategy_type_comparison_chart(strategy_type_metrics):
    """
    Create comparison bar chart for key metrics across strategy types.

    Args:
        strategy_type_metrics: DataFrame with strategy type metrics

    Returns:
        Plotly figure object
    """
    df = strategy_type_metrics.reset_index()

    # Sort by Total Profit
    df = df.sort_values("Total Profit", ascending=True)

    fig = go.Figure()

    # Add Total Profit bars
    fig.add_trace(
        go.Bar(
            y=df["Strategy Type"],
            x=df["Total Profit"],
            name="Total Profit",
            orientation="h",
            marker_color=[
                "#28a745" if x > 0 else "#dc3545" for x in df["Total Profit"]
            ],
            text=df["Total Profit"].apply(lambda x: f"${x:,.2f}"),
            textposition="auto",
            hovertemplate="%{y}<br>Total Profit: $%{x:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Total Profit by Strategy Type",
        xaxis_title="Total Profit ($)",
        yaxis_title="Strategy Type",
        template="plotly_white",
        height=max(400, len(df) * 60),
        showlegend=False,
        margin=dict(l=150, r=50, t=80, b=50),
    )

    return fig


def create_cumulative_performance_chart(cumulative_data):
    """
    Create line chart showing cumulative performance over time by strategy type.

    Args:
        cumulative_data: DataFrame with cumulative profit data

    Returns:
        Plotly figure object
    """
    if cumulative_data.empty:
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    fig = go.Figure()

    # Define color palette
    colors = [
        "#0066cc",
        "#28a745",
        "#ffc107",
        "#dc3545",
        "#6f42c1",
        "#17a2b8",
        "#fd7e14",
        "#e83e8c",
        "#20c997",
        "#6c757d",
    ]

    strategy_types = cumulative_data["strategy_type"].unique()

    for i, strategy_type in enumerate(strategy_types):
        strategy_data = cumulative_data[
            cumulative_data["strategy_type"] == strategy_type
        ]

        fig.add_trace(
            go.Scatter(
                x=strategy_data["datetime"],
                y=strategy_data["cumulative_profit"],
                mode="lines",
                name=strategy_type,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=(
                    f"<b>{strategy_type}</b><br>"
                    + "Date: %{x|%Y-%m-%d %H:%M}<br>"
                    + "Cumulative: $%{y:,.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Cumulative Profit by Strategy Type Over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative Profit ($)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    return fig
