"""
Trades Table Tab Component
"""
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

def render_trades_tab(merged_deals):
    """
    Render the trades table tab
    """
    instances = list(merged_deals["comment_open"].unique())
    default_instance = instances[0] if instances else None
    
    return html.Div([
        html.H3("📋 Trades Table", className="section-title"),
        html.P("View individual trades by bot/strategy", className="section-description"),
        
        html.Div([
            html.Label("Select Bot/Strategy:", className="form-label"),
            dcc.Dropdown(
                id='bot-dropdown',
                options=[{'label': i, 'value': i} for i in instances],
                value=default_instance,
                className="bot-selector"
            )
        ], className="mb-4"),
        
        html.Div(id='trades-table-container', className="table-container")
    ])

def create_trades_table(merged_deals, selected_bot):
    """
    Create filtered trades table for selected bot
    """
    if not selected_bot:
        return html.Div("Please select a bot/strategy", className="text-center text-muted")
    
    # Filter and prepare data
    instance_deals = merged_deals[merged_deals["comment_open"] == selected_bot].copy()
    instance_deals["total_profit"] = instance_deals["profit_open"] + instance_deals["profit_close"]
    
    # Select and rename columns
    display_cols = ["symbol_open", "time_open", "time_close", "total_profit", "price_open", "price_close"]
    available_cols = [col for col in display_cols if col in instance_deals.columns]
    
    instance_deals = instance_deals[available_cols]
    
    column_mapping = {
        "symbol_open": "Symbol",
        "time_open": "Open Time",
        "time_close": "Close Time",
        "total_profit": "Total Profit",
        "price_open": "Open Price",
        "price_close": "Close Price"
    }
    
    instance_deals = instance_deals.rename(columns=column_mapping)
    
    # Format datetime columns
    for col in ["Open Time", "Close Time"]:
        if col in instance_deals.columns:
            instance_deals[col] = instance_deals[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create summary statistics
    total_trades = len(instance_deals)
    total_profit = instance_deals["Total Profit"].sum()
    avg_profit = instance_deals["Total Profit"].mean()
    win_rate = (instance_deals["Total Profit"] > 0).mean() * 100
    
    summary = html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6("Total Trades", className="summary-title"),
                    html.H4(f"{total_trades}", className="summary-value")
                ], className="summary-card")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H6("Total Profit", className="summary-title"),
                    html.H4(f"${total_profit:,.2f}", 
                           className=f"summary-value {'text-success' if total_profit > 0 else 'text-danger'}")
                ], className="summary-card")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H6("Average Profit", className="summary-title"),
                    html.H4(f"${avg_profit:,.2f}",
                           className=f"summary-value {'text-success' if avg_profit > 0 else 'text-danger'}")
                ], className="summary-card")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H6("Win Rate", className="summary-title"),
                    html.H4(f"{win_rate:.1f}%",
                           className=f"summary-value {'text-success' if win_rate > 50 else 'text-danger'}")
                ], className="summary-card")
            ], width=3),
        ], className="mb-4")
    ])
    
    # Create data table
    table = dash_table.DataTable(
        data=instance_deals.to_dict('records'),
        columns=[
            {"name": col, "id": col, "type": "numeric", "format": {"specifier": ",.2f"}}
            if col in ['Total Profit', 'Open Price', 'Close Price']
            else {"name": col, "id": col}
            for col in instance_deals.columns
        ],
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{Total Profit} > 0',
                    'column_id': 'Total Profit'
                },
                'backgroundColor': 'rgba(40, 167, 69, 0.1)',
                'color': '#28a745',
                'fontWeight': 'bold'
            },
            {
                'if': {
                    'filter_query': '{Total Profit} < 0',
                    'column_id': 'Total Profit'
                },
                'backgroundColor': 'rgba(220, 53, 69, 0.1)',
                'color': '#dc3545',
                'fontWeight': 'bold'
            }
        ],
        style_header={
            'backgroundColor': 'rgba(0, 102, 204, 0.1)',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={'overflowX': 'auto'}
    )
    
    return html.Div([summary, table])
