from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

def render_log_tab():

    return html.Div([
        html.H3("Detailed Analysis", className="section-title"),

        # Metrics selector
        html.Div([
            html.Label("Select Metrics to Display:", className="form-label"),
            dcc.Dropdown(
                id='metrics-dropdown',
                options=[
                    {'label': 'Total Profit', 'value': 'Total Profit'},
                    {'label': 'Win Rate (%)', 'value': 'Win Rate (%)'},
                    {'label': 'Sharpe Ratio', 'value': 'Sharpe Ratio'},
                    {'label': 'Max Drawdown (%)', 'value': 'Max Drawdown (%)'},
                    {'label': 'Profit Factor', 'value': 'Profit Factor'},
                    {'label': 'Average Profit', 'value': 'Average Profit by Trade'},
                ],
                value=["Total Profit", "Win Rate (%)", "Sharpe Ratio"],
                multi=True,
                className="metrics-selector"
            )
        ], className="mb-4"),
