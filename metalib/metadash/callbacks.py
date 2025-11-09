"""
Callbacks Module
Handles all Dash callbacks for the MetaDAsh application
"""
from dash import Input, Output, State, html
import dash_bootstrap_components as dbc
from datetime import datetime
import plotly.graph_objects as go

from utils import (
    initialize_mt5,
    get_historical_data,
    process_deals_data,
    get_account_info,
    strategy_metrics,
    calculate_hourly_performance
)

from components import (
    render_overview_tab,
    render_detailed_tab,
    create_detailed_metrics_figure,
    render_pnl_tab,
    render_trades_tab,
    create_trades_table,
    render_raw_tab
)

# Global storage for data
stored_data = {
    'history_orders': None,
    'history_deals': None,
    'merged_deals': None,
    'account_size': 100000
}

def register_callbacks(app):
    """
    Register all callbacks for the application
    """
    
    @app.callback(
        Output('connection-status', 'children'),
        Input('connect-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def connect_mt5(n_clicks):
        """Handle MT5 connection"""
        if n_clicks:
            success, message = initialize_mt5()
            if success:
                return dbc.Alert(message, color="success", className="status-alert")
            else:
                return dbc.Alert(message, color="danger", className="status-alert")
        return ""
    
    @app.callback(
        [Output('data-store', 'data'),
         Output('fetch-status', 'children'),
         Output('account-info-store', 'data')],
        [Input('fetch-btn', 'n_clicks')],
        [State('start-date', 'date'),
         State('end-date', 'date'),
         State('account-size', 'value')],
        prevent_initial_call=True
    )
    def fetch_data(n_clicks, start_date, end_date, account_size):
        """Fetch trading data from MT5"""
        if not n_clicks:
            return None, "", None
        
        # Convert dates
        from_date = datetime.strptime(start_date, '%Y-%m-%d')
        to_date = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
        
        # Get historical data
        history_orders, history_deals, error = get_historical_data(from_date, to_date)
        
        if error:
            return None, dbc.Alert(error, color="danger", className="status-alert"), None
        
        if history_orders is None or history_deals is None:
            return None, dbc.Alert("Failed to retrieve data", color="danger", className="status-alert"), None
        
        # Process deals data
        merged_deals = process_deals_data(history_deals)
        
        if merged_deals is None:
            return None, dbc.Alert("No valid trades found", color="warning", className="status-alert"), None
        
        # Get account info
        account_info = get_account_info()
        
        # Store data globally
        stored_data['history_orders'] = history_orders
        stored_data['history_deals'] = history_deals
        stored_data['merged_deals'] = merged_deals
        stored_data['account_size'] = account_size
        
        message = f"✓ Retrieved {len(history_orders)} orders and {len(history_deals)} deals"
        
        return (
            {'data_available': True},
            dbc.Alert(message, color="success", className="status-alert"),
            account_info
        )
    
    @app.callback(
        Output('tab-content', 'children'),
        [Input('tabs', 'active_tab'),
         Input('data-store', 'data'),
         Input('account-info-store', 'data')]
    )
    def render_tab_content(active_tab, data, account_info):
        """Render content based on selected tab"""
        if data is None or not data.get('data_available', False):
            return html.Div([
                html.Img(src="/assets/logo.png", className="empty-state-img"),
                html.H4("No Data Available", className="text-center text-muted mt-3"),
                html.P("Please connect to MT5 and fetch trading data to view analytics", 
                      className="text-center text-muted")
            ], className="empty-state")
        
        merged_deals = stored_data['merged_deals']
        account_size = stored_data['account_size']
        
        if not account_info:
            account_info = {'balance': 0, 'equity': 0, 'margin': 0}
        
        if active_tab == "overview":
            return render_overview_tab(merged_deals, account_size, account_info)
        elif active_tab == "detailed":
            return render_detailed_tab(merged_deals, account_size)
        elif active_tab == "pnl":
            return render_pnl_tab(merged_deals, account_size)
        elif active_tab == "trades":
            return render_trades_tab(merged_deals)
        elif active_tab == "raw":
            return render_raw_tab(merged_deals, stored_data['history_orders'])
        
        return html.Div()
    
    @app.callback(
        Output('detailed-metrics-graph', 'figure'),
        [Input('metrics-dropdown', 'value')],
        [State('data-store', 'data')]
    )
    def update_detailed_metrics(selected_metrics, data):
        """Update detailed metrics chart based on selection"""
        if not data or not selected_metrics:
            return go.Figure()
        
        merged_deals = stored_data['merged_deals']
        account_size = stored_data['account_size']
        
        if merged_deals is None:
            return go.Figure()
        
        # Calculate metrics
        strategy_metrics_df = merged_deals[
            ["profit_open", "profit_close", "comment_open", "symbol_open", "time_open"]
        ].copy()
        
        grouped_metrics = strategy_metrics_df.groupby(["comment_open", "symbol_open"]).apply(
            lambda x: strategy_metrics(x, account_size)
        )
        
        return create_detailed_metrics_figure(grouped_metrics, selected_metrics)
    
    @app.callback(
        Output('hourly-graph', 'figure'),
        [Input('strategy-dropdown', 'value')],
        [State('data-store', 'data')]
    )
    def update_hourly_graph(selected_strategy, data):
        """Update hourly performance graph"""
        if not data or not selected_strategy:
            return go.Figure()
        
        merged_deals = stored_data['merged_deals']
        account_size = stored_data['account_size']
        
        if merged_deals is None:
            return go.Figure()
        
        # Calculate hourly performance
        hourly_perf = calculate_hourly_performance(merged_deals, selected_strategy)
        
        fig = go.Figure()
        
        if not hourly_perf.empty:
            fig.add_trace(go.Bar(
                x=hourly_perf.index,
                y=hourly_perf["Average Profit by Trade"].fillna(0),
                name='Hourly Avg Profit',
                marker_color='#0066cc',
                hovertemplate='Hour: %{x}<br>Avg Profit: $%{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Hourly Performance for {selected_strategy}",
            xaxis_title="Hour of Day",
            yaxis_title="Average Profit ($)",
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @app.callback(
        Output('trades-table-container', 'children'),
        [Input('bot-dropdown', 'value')],
        [State('data-store', 'data')]
    )
    def update_trades_table(selected_bot, data):
        """Update trades table based on selected bot"""
        if not data or not selected_bot:
            return html.Div("Please select a bot/strategy", className="text-center text-muted")
        
        merged_deals = stored_data['merged_deals']
        
        if merged_deals is None:
            return html.Div("No data available", className="text-center text-muted")
        
        return create_trades_table(merged_deals, selected_bot)

    # Add missing import at the top if needed
    import dash_bootstrap_components as dbc
