"""
Tabs Layout Component
"""

import dash_bootstrap_components as dbc


def create_tabs_layout():
    """
    Creates the tabs navigation layout
    """
    return dbc.Tabs(
        [
            dbc.Tab(label="Accueil", tab_id="welcome"),
            dbc.Tab(label="Status Monitor", tab_id="status"),
            dbc.Tab(label="Overview", tab_id="overview"),
            dbc.Tab(label="Strategy Types", tab_id="strategy_types"),
            dbc.Tab(label="Detailed Analysis", tab_id="detailed"),
            dbc.Tab(label="PnL Performance", tab_id="pnl"),
            dbc.Tab(label="Trades Table", tab_id="trades"),
            dbc.Tab(label="Instance Trades", tab_id="instance_trades"),
            dbc.Tab(label="Weight Calibration", tab_id="calibration"),
            dbc.Tab(label="View Logs", tab_id="logs"),
            dbc.Tab(label="Raw Data", tab_id="raw"),
        ],
        id="tabs",
        active_tab="overview",
        className="nav-tabs",
    )
