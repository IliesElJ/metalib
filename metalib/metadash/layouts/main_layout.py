"""
Main Layout Module
Defines the overall structure of the MetaDAsh application
"""
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import date
from layouts.sidebar import create_sidebar
from layouts.header import create_header
from layouts.tabs import create_tabs_layout

def get_layout():
    """
    Returns the main layout of the application
    """
    return html.Div([
        dcc.Store(id='data-store'),
        dcc.Store(id='account-info-store'),
        dcc.Store(id='metrics-store'),
        
        html.Div([
            # Header
            create_header(),
            
            # Main content
            dbc.Row([
                # Sidebar
                dbc.Col([
                    create_sidebar()
                ], width=3, className="sidebar-column"),
                
                # Content area
                dbc.Col([
                    html.Div([
                        create_tabs_layout(),
                        html.Div(id="tab-content", className="content-container fade-in")
                    ])
                ], width=9, className="content-column")
            ], className="main-row")
        ], className="main-container")
    ])
