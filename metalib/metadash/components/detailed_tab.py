"""
Detailed Analysis Tab Component
"""
from dash import html, dcc
import plotly.graph_objects as go
from metalib.metadash.utils.metrics import strategy_metrics

def render_detailed_tab(merged_deals, account_size):
    """
    Render the detailed analysis tab
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
    duration_content = []
    if "time_close" in merged_deals.columns:
        duration_fig = create_duration_chart(merged_deals)
        duration_content = [
            html.Div([
                html.H4("Trade Duration Analysis", className="section-subtitle"),
                dcc.Graph(figure=duration_fig)
            ], className="graph-container mb-4")
        ]

    return html.Div([
        html.H3("Detailed Analysis", className="section-title"),

        # Metrics selector
        html.Div([
            html.Label("Select Metrics to Display:", className="form-label"),
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
                className="metrics-selector"
            )
        ], className="mb-4"),

        # Detailed metrics chart
        html.Div([
            dcc.Graph(id='detailed-metrics-graph')
        ], className="graph-container mb-4"),

        # Duration analysis
        *duration_content,

        # Hourly performance
        html.Div([
            html.H4("Hourly Performance Analysis", className="section-subtitle"),
            html.Label("Select Strategy:", className="form-label"),
            dcc.Dropdown(
                id='strategy-dropdown',
                options=[{'label': s, 'value': s} for s in strategies],
                value=default_strategy,
                className="strategy-selector mb-3"
            ),
            dcc.Graph(id='hourly-graph')
        ], className="graph-container")
    ])

def create_duration_chart(merged_deals):
    """
    Create trade duration distribution chart
    """
    merged_deals["duration"] = (
        merged_deals["time_close"] - merged_deals["time_open"]
    ).dt.total_seconds() / 3600  # Convert to hours

    fig = go.Figure()

    colors = ['#0066cc', '#28a745', '#ffc107', '#dc3545', '#17a2b8', '#6610f2']

    for i, symbol in enumerate(merged_deals["symbol_open"].unique()):
        symbol_data = merged_deals[merged_deals["symbol_open"] == symbol]
        fig.add_trace(go.Histogram(
            x=symbol_data["duration"],
            name=symbol,
            opacity=0.7,
            nbinsx=50,
            marker_color=colors[i % len(colors)],
            hovertemplate='Duration: %{x:.1f} hours<br>Count: %{y}<extra></extra>'
        ))

    fig.update_layout(
        title="Trade Duration Distribution",
        xaxis_title="Duration (hours)",
        yaxis_title="Count",
        barmode='overlay',
        template='plotly_white',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def create_detailed_metrics_figure(grouped_metrics, selected_metrics):
    """
    Create detailed metrics comparison chart
    """
    plot_data = grouped_metrics.reset_index()
    plot_data['strategy_symbol'] = plot_data['comment_open'] + ' - ' + plot_data['symbol_open']

    fig = go.Figure()

    colors = {
        'Total Profit': '#0066cc',
        'Win Rate (%)': '#28a745',
        'Sharpe Ratio': '#ffc107',
        'Max Drawdown (%)': '#dc3545',
        'Profit Factor': '#17a2b8',
        'Average Profit by Trade': '#6610f2'
    }

    for metric in selected_metrics:
        if metric in plot_data.columns:
            fig.add_trace(go.Bar(
                x=plot_data['strategy_symbol'],
                y=plot_data[metric],
                name=metric,
                marker_color=colors.get(metric, '#666'),
                hovertemplate=f'{metric}: %{{y:.2f}}<extra></extra>'
            ))

    fig.update_layout(
        title="Strategy Metrics Comparison",
        xaxis_title="Strategy - Symbol",
        yaxis_title="Value",
        barmode='group',
        template='plotly_white',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig
