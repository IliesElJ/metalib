"""
Sidebar Component
Contains settings and controls for the application
"""
from datetime import date
from dash import html, dcc
import dash_bootstrap_components as dbc

def create_sidebar():
    """
    Compact, aligned sidebar with full-width controls (no rounded corners).
    """
    return html.Div([
        html.Div([
            html.H4("Settings", className="sidebar-title", style={"margin": 5})
        ], className="sidebar-header"),

        html.Div([
            dbc.Form([
                # Date range (two columns on wide, stacked on narrow)
                dbc.Row([
                    dbc.Col([
                        html.Label("Start Date", className="form-label compact"),
                        dcc.DatePickerSingle(
                            id='start-date',
                            date=date(2025, 1, 1),
                            display_format='YYYY-MM-DD',
                            className="flat-date w-100"
                        ),
                    ], className="mb-2"),
                    dbc.Col([
                        html.Label("End Date", className="form-label compact"),
                        dcc.DatePickerSingle(
                            id='end-date',
                            date=date.today(),
                            display_format='YYYY-MM-DD',
                            className="flat-date w-100"
                        ),
                    ], className="mb-2"),
                ], className="g-2"),

                # Account size
                html.Div([
                    html.Label("Account Size ($)", className="form-label compact"),
                    dbc.InputGroup([
                        dbc.Input(
                            id='account-size',
                            type='number',
                            value=100000,
                            min=10000,
                            step=1000,
                            size="sm",
                            className="form-control flat-input w-100"
                        ),
                    ], className="flat-ig w-100"),
                ], className="mb-2"),

                # Buttons (side-by-side on wide, stacked on narrow)
                dbc.Row([
                    dbc.Col(dbc.Button(
                        "Connect to MT5",
                        id="connect-btn",
                        n_clicks=0,
                        className="btn-connect btn-sm w-100"
                    ), xs=12, md=6, className="mb-2"),
                    dbc.Col(dbc.Button(
                        "Fetch Trading Data",
                        id="fetch-btn",
                        n_clicks=0,
                        className="btn-fetch btn-sm w-100"
                    ), xs=12, md=6, className="mb-2"),
                ], className="g-2"),

                # Status
                html.Div(id="connection-status", className="status-container mb-1"),
                html.Div(id="fetch-status", className="status-container mb-1"),

                # Loading
                dcc.Loading(
                    id="loading-indicator",
                    type="circle",
                    color="#00712D",
                    children=html.Div(id="loading-output"),
                ),
            ], className="sidebar-form sidebar-compact")
        ], className="sidebar-body")
    ], className="sidebar-container")
