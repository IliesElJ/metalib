"""
MetaDAsh - Main Application Entry Point
"""
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import warnings

warnings.filterwarnings("ignore")

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.title = "MetaDAsh"
server = app.server

# Import layout and callbacks after app initialization
from layouts.main_layout import get_layout
from callbacks import register_callbacks

# Set the layout
app.layout = get_layout()

# Register all callbacks
register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True, port=8050, host='0.0.0.0')
