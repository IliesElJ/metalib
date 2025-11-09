"""
Raw Data Tab Component
"""
from dash import html, dcc, dash_table
import base64
import pandas as pd

def render_raw_tab(merged_deals, history_orders):
    """
    Render the raw data tab with export functionality
    """
    # Convert orders to DataFrame
    orders_df = pd.DataFrame()
    if history_orders and len(history_orders) > 0:
        orders_df = pd.DataFrame(list(history_orders), columns=history_orders[0]._asdict().keys())
    
    # Prepare download links
    merged_csv = merged_deals.to_csv(index=False)
    merged_b64 = base64.b64encode(merged_csv.encode()).decode()
    
    orders_csv = orders_df.to_csv(index=False) if not orders_df.empty else ""
    orders_b64 = base64.b64encode(orders_csv.encode()).decode() if orders_csv else ""
    
    return html.Div([
        html.H3("💾 Raw Data Export", className="section-title"),
        
        # Merged Deals Section
        html.Div([
            html.H4("Merged Deals Data", className="section-subtitle"),
            html.P(f"Total Records: {len(merged_deals)}", className="text-muted"),
            
            # Data preview
            html.Div([
                create_data_preview_table(merged_deals, "merged-deals-table")
            ], className="table-container mb-3"),
            
            # Download button
            html.A(
                html.Button("📥 Download Merged Deals CSV", className="btn btn-primary"),
                href=f"data:text/csv;base64,{merged_b64}",
                download="merged_deals.csv",
                className="download-link"
            )
        ], className="data-section mb-5"),
        
        # Orders Data Section
        html.Div([
            html.H4("Orders Data", className="section-subtitle"),
            html.P(f"Total Records: {len(orders_df)}", className="text-muted"),
            
            # Data preview
            html.Div([
                create_data_preview_table(orders_df, "orders-table") if not orders_df.empty 
                else html.P("No orders data available", className="text-center text-muted")
            ], className="table-container mb-3"),
            
            # Download button
            html.A(
                html.Button("📥 Download Orders CSV", className="btn btn-primary"),
                href=f"data:text/csv;base64,{orders_b64}",
                download="orders.csv",
                className="download-link"
            ) if not orders_df.empty else html.Div()
        ], className="data-section"),
        
        # Data Statistics
        html.Div([
            html.H4("Data Statistics", className="section-subtitle mt-4"),
            create_data_statistics(merged_deals)
        ], className="stats-section")
    ])

def create_data_preview_table(df, table_id):
    """
    Create a preview table for the data (first 100 rows)
    """
    preview_df = df.head(100).copy()
    
    # Convert datetime columns to string for display
    for col in preview_df.columns:
        if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
            preview_df[col] = preview_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return dash_table.DataTable(
        id=table_id,
        data=preview_df.to_dict('records'),
        columns=[{"name": str(i), "id": str(i)} for i in preview_df.columns],
        style_cell={
            'textAlign': 'left',
            'padding': '8px',
            'maxWidth': '150px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_header={
            'backgroundColor': 'rgba(0, 102, 204, 0.1)',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        page_size=10,
        style_table={'overflowX': 'auto'},
        tooltip_data=[
            {
                column: {'value': str(value), 'type': 'markdown'}
                for column, value in row.items()
            } for row in preview_df.to_dict('records')
        ],
        tooltip_delay=0,
        tooltip_duration=None
    )

def create_data_statistics(df):
    """
    Create data statistics summary
    """
    stats = []
    
    # Basic info
    stats.append(html.Div([
        html.Strong("Dataset Shape: "),
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    ], className="mb-2"))
    
    # Date range
    if 'time_open' in df.columns:
        date_range = f"{df['time_open'].min().strftime('%Y-%m-%d')} to {df['time_open'].max().strftime('%Y-%m-%d')}"
        stats.append(html.Div([
            html.Strong("Date Range: "),
            date_range
        ], className="mb-2"))
    
    # Unique values
    if 'symbol_open' in df.columns:
        unique_symbols = df['symbol_open'].nunique()
        stats.append(html.Div([
            html.Strong("Unique Symbols: "),
            str(unique_symbols)
        ], className="mb-2"))
    
    if 'comment_open' in df.columns:
        unique_strategies = df['comment_open'].nunique()
        stats.append(html.Div([
            html.Strong("Unique Strategies: "),
            str(unique_strategies)
        ], className="mb-2"))
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
    stats.append(html.Div([
        html.Strong("Memory Usage: "),
        f"{memory_usage:.2f} MB"
    ], className="mb-2"))
    
    # Missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        stats.append(html.Div([
            html.Strong("Missing Values: "),
            f"{missing} ({missing / (df.shape[0] * df.shape[1]) * 100:.2f}%)"
        ], className="mb-2 text-warning"))
    
    return html.Div(stats, className="statistics-container")
