"""
MetaDAsh - Main Application Entry Point
"""

import dash
import dash_bootstrap_components as dbc
import warnings

warnings.filterwarnings("ignore")

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&family=Inter:wght@400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
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

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8501)
