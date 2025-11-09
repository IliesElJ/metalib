"""
Sidebar Component
Contains settings and controls for the application
"""
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import date

def create_sidebar():
    """
    Creates the sidebar with settings and controls
    """
    return html.Div([
        html.Div([
            html.H4("Settings", className="sidebar-title")
        ], className="sidebar-header"),
        
        html.Div([
            # Date range selection
            html.Div([
                html.Label("Start Date", className="form-label"),
                dcc.DatePickerSingle(
                    id='start-date',
                    date=date(2020, 1, 1),
                    display_format='YYYY-MM-DD',
                    className="date-picker"
                ),
            ], className="mb-3"),
            
            html.Div([
                html.Label("End Date", className="form-label"),
                dcc.DatePickerSingle(
                    id='end-date',
                    date=date.today(),
                    display_format='YYYY-MM-DD',
                    className="date-picker"
                ),
            ], className="mb-3"),
            
            # Account size input
            html.Div([
                html.Label("Account Size ($)", className="form-label"),
                dbc.Input(
                    id='account-size',
                    type='number',
                    value=100000,
                    min=10000,
                    step=1000,
                    className="form-control"
                ),
            ], className="mb-3"),
            
            # Action buttons
            html.Div([
                dbc.Button(
                    "Connect to MT5",
                    id="connect-btn",
                    className="btn-connect",
                    n_clicks=0
                ),
                dbc.Button(
                    "Fetch Trading Data",
                    id="fetch-btn",
                    className="btn-fetch",
                    n_clicks=0
                ),
            ], className="button-group"),
            
            # Status messages
            html.Div(id="connection-status", className="status-container"),
            html.Div(id="fetch-status", className="status-container"),
            
            # Loading indicator
            dcc.Loading(
                id="loading-indicator",
                type="circle",
                children=html.Div(id="loading-output")
            )
        ], className="sidebar-body")
    ], className="sidebar-container")
