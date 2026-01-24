"""
Main Layout Module
Defines the overall structure of the MetaDAsh application
"""
from dash import dcc, html
import dash_bootstrap_components as dbc
from layouts.header import create_header
from layouts.tabs import create_tabs_layout

def get_layout():
    """
    Returns the main layout of the application.
    Sidebar has been removed - data is loaded automatically on startup.
    """
    return html.Div([
        dcc.Store(id='data-store'),
        dcc.Store(id='account-info-store'),
        dcc.Store(id='metrics-store'),

        # Interval component for initial data load
        dcc.Interval(
            id='startup-trigger',
            interval=500,  # 500ms after load
            n_intervals=0,
            max_intervals=1,  # Only fire once
        ),

        html.Div([
            # Header
            create_header(),

            # Main content (full width)
            dbc.Container([
                html.Div([
                    create_tabs_layout(),

                    # Connection status (small, top-right)
                    html.Div(
                        id="connection-status",
                        style={
                            "position": "fixed",
                            "top": "70px",
                            "right": "20px",
                            "zIndex": "1000",
                            "maxWidth": "300px",
                        }
                    ),

                    html.Div(id="tab-content", className="content-container fade-in")
                ])
            ], fluid=True, className="px-4 py-3")
        ], className="main-container")
    ])
