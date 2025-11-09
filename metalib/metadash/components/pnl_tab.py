"""
PnL Performance Tab Component
"""
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from utils.metrics import calculate_streak_analysis

def render_pnl_tab(merged_deals, account_size):
    """
    Render the PnL performance tab
    """
    # Calculate PnL metrics
    merged_deals["total_profit"] = merged_deals["profit_open"] + merged_deals["profit_close"]
    merged_deals_sorted = merged_deals.sort_values("time_open")
    merged_deals_sorted["cumulative_profit"] = merged_deals_sorted["total_profit"].cumsum()
    
    # Create charts
    equity_fig = create_equity_curve(merged_deals_sorted, account_size)
    drawdown_fig = create_drawdown_chart(merged_deals_sorted, account_size)
    symbol_fig = create_symbol_profit_chart(merged_deals_sorted)
    scatter_fig = create_trade_scatter(merged_deals_sorted)
    streak_fig, streak_stats = create_streak_analysis(merged_deals_sorted)
    
    # Strategy performance table
    strategy_perf = calculate_strategy_performance(merged_deals)
    
    return html.Div([
        html.H3("PnL Performance Analysis", className="section-title"),
        
        # Three main charts in a row
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=equity_fig)
                ], className="graph-container")
            ], width=4),
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=drawdown_fig)
                ], className="graph-container")
            ], width=4),
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=symbol_fig)
                ], className="graph-container")
            ], width=4),
        ], className="mb-4"),
        
        # Trade scatter plot
        html.Div([
            html.H4("Individual Trade Performance", className="section-subtitle"),
            dcc.Graph(figure=scatter_fig)
        ], className="graph-container mb-4"),
        
        # Streak analysis
        html.Div([
            html.H4("Win/Loss Streak Analysis", className="section-subtitle"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=streak_fig)
                ], width=8),
                dbc.Col([
                    html.Div([
                        html.H5("Streak Statistics", className="stats-title"),
                        html.Div(streak_stats, className="streak-stats")
                    ], className="stats-container")
                ], width=4)
            ])
        ], className="graph-container mb-4"),
        
        # Strategy performance table
        html.Div([
            html.H4("Performance by Strategy", className="section-subtitle"),
            create_strategy_table(strategy_perf)
        ], className="table-container")
    ])

def create_equity_curve(equity_data, account_size):
    """
    Create account equity curve chart
    """
    equity_data = equity_data.copy()
    equity_data["equity"] = account_size + equity_data["cumulative_profit"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_data["time_open"],
        y=equity_data["equity"],
        mode="lines",
        name="Account Equity",
        line=dict(color='#0066cc', width=2),
        fill='tonexty',
        fillcolor='rgba(0, 102, 204, 0.1)',
        hovertemplate='Date: %{x|%Y-%m-%d %H:%M}<br>Equity: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add starting balance line
    fig.add_hline(y=account_size, line_dash="dash", line_color="gray", 
                  annotation_text="Starting Balance")
    
    fig.update_layout(
        title="Account Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template='plotly_white',
        height=350,
        showlegend=False
    )
    
    return fig

def create_drawdown_chart(equity_data, account_size):
    """
    Create drawdown chart
    """
    equity_data = equity_data.copy()
    equity_data["equity"] = account_size + equity_data["cumulative_profit"]
    equity_data["running_max"] = equity_data["equity"].cummax()
    equity_data["drawdown"] = (equity_data["equity"] / equity_data["running_max"] - 1) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_data["time_open"],
        y=equity_data["drawdown"],
        mode="lines",
        line=dict(color='#dc3545', width=2),
        fill='tozeroy',
        fillcolor='rgba(220, 53, 69, 0.2)',
        name="Drawdown",
        hovertemplate='Date: %{x|%Y-%m-%d %H:%M}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Drawdown (%)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template='plotly_white',
        height=350,
        showlegend=False
    )
    
    return fig

def create_symbol_profit_chart(merged_deals_sorted):
    """
    Create cumulative profit by symbol chart
    """
    symbols = merged_deals_sorted["symbol_open"].unique()
    fig = go.Figure()
    
    colors = ['#0066cc', '#28a745', '#ffc107', '#dc3545', '#17a2b8', '#6610f2']
    
    for i, symbol in enumerate(symbols):
        symbol_data = merged_deals_sorted[merged_deals_sorted["symbol_open"] == symbol].copy()
        symbol_data = symbol_data.sort_values("time_open")
        symbol_data["cumulative_profit"] = symbol_data["total_profit"].cumsum()
        
        fig.add_trace(go.Scatter(
            x=symbol_data["time_open"],
            y=symbol_data["cumulative_profit"],
            mode='lines',
            name=symbol,
            line=dict(width=2, color=colors[i % len(colors)]),
            hovertemplate='%{y:,.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Cumulative Profit by Symbol",
        xaxis_title="Date",
        yaxis_title="Cumulative Profit ($)",
        template='plotly_white',
        height=350,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.02
        )
    )
    
    return fig

def create_trade_scatter(merged_deals_sorted):
    """
    Create trade scatter plot
    """
    symbols = merged_deals_sorted["symbol_open"].unique()
    fig = go.Figure()
    
    colors = ['#0066cc', '#28a745', '#ffc107', '#dc3545', '#17a2b8', '#6610f2']
    
    for i, symbol in enumerate(symbols):
        symbol_data = merged_deals_sorted[merged_deals_sorted["symbol_open"] == symbol]
        
        # Scale marker size based on profit magnitude
        max_profit = np.abs(symbol_data["total_profit"]).max()
        marker_sizes = np.abs(symbol_data["total_profit"]) / max(1, max_profit) * 20 + 5
        
        fig.add_trace(go.Scatter(
            x=symbol_data["time_open"],
            y=symbol_data["total_profit"],
            mode="markers",
            name=symbol,
            marker=dict(
                size=marker_sizes,
                color=colors[i % len(colors)],
                opacity=0.6,
                line=dict(width=1, color='white')
            ),
            text=symbol_data["comment_open"],
            hovertemplate='Date: %{x|%Y-%m-%d %H:%M}<br>Profit: $%{y:,.2f}<br>Strategy: %{text}<extra></extra>'
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title="Individual Trade Performance",
        xaxis_title="Date",
        yaxis_title="Profit/Loss ($)",
        template='plotly_white',
        height=400
    )
    
    return fig

def create_streak_analysis(merged_deals_sorted):
    """
    Create streak analysis chart and statistics
    """
    streaks = calculate_streak_analysis(merged_deals_sorted)
    
    win_streaks = streaks[streaks["win"]]
    loss_streaks = streaks[~streaks["win"]]
    
    fig = go.Figure()
    
    if not win_streaks.empty:
        fig.add_trace(go.Bar(
            x=win_streaks.index,
            y=win_streaks["streak_length"],
            name="Win Streaks",
            marker_color="#28a745",
            text=win_streaks["total_profit"].round(2),
            hovertemplate='Streak #%{x}<br>Length: %{y}<br>Profit: $%{text}<extra></extra>'
        ))
    
    if not loss_streaks.empty:
        fig.add_trace(go.Bar(
            x=loss_streaks.index,
            y=-loss_streaks["streak_length"],  # Negative for visual effect
            name="Loss Streaks",
            marker_color="#dc3545",
            text=loss_streaks["total_profit"].round(2),
            hovertemplate='Streak #%{x}<br>Length: %{y}<br>Loss: $%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Win/Loss Streaks",
        xaxis_title="Streak ID",
        yaxis_title="Streak Length",
        template='plotly_white',
        height=300,
        barmode='relative'
    )
    
    # Calculate streak statistics
    stats = html.Div([
        html.P([
            html.Strong("Longest Win Streak: "),
            f"{win_streaks['streak_length'].max() if not win_streaks.empty else 0} trades"
        ]),
        html.P([
            html.Strong("Longest Loss Streak: "),
            f"{loss_streaks['streak_length'].max() if not loss_streaks.empty else 0} trades"
        ]),
        html.P([
            html.Strong("Best Streak Profit: "),
            f"${win_streaks['total_profit'].max():.2f}" if not win_streaks.empty else "$0.00"
        ]),
        html.P([
            html.Strong("Worst Streak Loss: "),
            f"${loss_streaks['total_profit'].min():.2f}" if not loss_streaks.empty else "$0.00"
        ])
    ])
    
    return fig, stats

def calculate_strategy_performance(merged_deals):
    """
    Calculate performance metrics by strategy
    """
    merged_deals["total_profit"] = merged_deals["profit_open"] + merged_deals["profit_close"]
    
    strategy_perf = merged_deals.groupby("comment_open").agg({
        "total_profit": ["sum", "mean", "count", "std"]
    }).round(2)
    
    strategy_perf.columns = ["Total Profit", "Average Profit", "Number of Trades", "Std Dev"]
    
    # Calculate win rate
    win_rates = merged_deals.groupby("comment_open")["total_profit"].apply(
        lambda x: 100 * (x > 0).mean()
    ).round(2)
    
    strategy_perf["Win Rate (%)"] = win_rates
    strategy_perf = strategy_perf.reset_index()
    strategy_perf.rename(columns={"comment_open": "Strategy"}, inplace=True)
    
    return strategy_perf

def create_strategy_table(strategy_perf):
    """
    Create strategy performance table
    """
    return dash_table.DataTable(
        data=strategy_perf.to_dict('records'),
        columns=[
            {"name": col, "id": col, "type": "numeric", "format": {"specifier": ".2f"}}
            if col != "Strategy" else {"name": col, "id": col}
            for col in strategy_perf.columns
        ],
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_data_conditional=[
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
            'fontWeight': 'bold'
        },
        sort_action="native"
    )
