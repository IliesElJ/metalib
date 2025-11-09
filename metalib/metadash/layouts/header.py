"""
Header Component
"""
from dash import html

def create_header():
    """
    Creates the application header
    """
    return html.Div([
        html.H1("MetaDAsh", className="app-title"),
        html.P("Analytics Dashboard", className="subtitle")
    ], className="app-header")
