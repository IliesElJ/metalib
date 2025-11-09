"""
Overview Tab Component
"""
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from utils.metrics import strategy_metrics, calculate_daily_performance

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
    
    # Calculate daily performance
    daily_profit = calculate_daily_performance(merged_deals)
    
    # Create daily performance figure
    daily_fig = create_daily_performance_chart(daily_profit)
    
    # Create strategy comparison figure
    comparison_fig = create_strategy_comparison_chart(grouped_metrics)
    
    return html.Div([
        # Account Overview Section
        html.Div([
            html.H3("📊 Account Overview", className="section-title"),
            dbc.Row([
                dbc.Col([create_metric_card("Balance", f"${account_info['balance']:.2f}", "primary")], width=4),
                dbc.Col([create_metric_card("Equity", f"${account_info['equity']:.2f}", "success")], width=4),
                dbc.Col([create_metric_card("Margin", f"${account_info['margin']:.2f}", "info")], width=4),
            ], className="mb-4"),
        ]),
        
        # Daily Performance Chart
        html.Div([
            dcc.Graph(figure=daily_fig, className="graph-container")
        ], className="mb-4"),
        
        # Strategy Metrics Table
        html.Div([
            html.H3("Strategy Performance Metrics", className="section-title"),
            create_metrics_table(grouped_metrics)
        ], className="mb-4"),
        
        # Strategy Comparison Chart
        html.Div([
            dcc.Graph(figure=comparison_fig, className="graph-container")
        ])
    ])

def create_metric_card(title, value, color_type="primary"):
    """
    Create a metric display card
    """
    return html.Div([
        html.H5(title, className="metric-title"),
        html.H4(value, className="metric-value")
    ], className=f"metric-card {color_type}")

def create_daily_performance_chart(daily_profit):
    """
    Create daily performance line chart
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_profit['date'],
        y=daily_profit['profit'],
        mode='lines+markers',
        name='Daily Profit',
        line=dict(color='#0066cc', width=2),
        marker=dict(size=6),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Profit: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Daily Performance",
        xaxis_title="Date",
        yaxis_title="Profit ($)",
        xaxis=dict(tickformat='%Y-%m-%d'),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_strategy_comparison_chart(grouped_metrics):
    """
    Create strategy comparison bar chart
    """
    plot_metrics = ["Total Profit", "Average Profit by Trade", "Win Rate (%)", "Loss Rate (%)"]
    plot_data = grouped_metrics.reset_index()
    plot_data['strategy_symbol'] = plot_data['comment_open'] + ' - ' + plot_data['symbol_open']
    
    fig = go.Figure()
    
    colors = ['#0066cc', '#28a745', '#ffc107', '#dc3545']
    
    for i, metric in enumerate(plot_metrics):
        if metric in plot_data.columns:
            fig.add_trace(go.Bar(
                x=plot_data['strategy_symbol'],
                y=plot_data[metric],
                name=metric,
                marker_color=colors[i % len(colors)],
                hovertemplate=f'{metric}: %{{y:,.2f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title="Strategy Performance Comparison",
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

def create_metrics_table(grouped_metrics):
    """
    Create a formatted metrics table
    """
    df = grouped_metrics.reset_index()
    
    # Format numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    columns = []
    for col in df.columns:
        if col in numeric_columns and col not in ['comment_open', 'symbol_open']:
            columns.append({
                "name": col,
                "id": col,
                "type": "numeric",
                "format": {"specifier": ".2f"}
            })
        else:
            columns.append({"name": col, "id": col})
    
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=columns,
        style_cell={
            'textAlign': 'left',
            'padding': '10px'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgba(0, 0, 0, 0.02)'
            },
            {
                'if': {'column_id': 'Total Profit', 'filter_query': '{Total Profit} > 0'},
                'color': '#28a745',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'Total Profit', 'filter_query': '{Total Profit} < 0'},
                'color': '#dc3545',
                'fontWeight': 'bold'
            }
        ],
        style_header={
            'backgroundColor': 'rgba(0, 102, 204, 0.1)',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        className="metrics-table"
    )
