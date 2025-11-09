"""
Tabs Layout Component
"""
import dash_bootstrap_components as dbc

def create_tabs_layout():
    """
    Creates the tabs navigation layout
    """
    return dbc.Tabs([
        dbc.Tab(label="📊 Overview", tab_id="overview"),
        dbc.Tab(label="📈 Detailed Analysis", tab_id="detailed"),
        dbc.Tab(label="💰 PnL Performance", tab_id="pnl"),
        dbc.Tab(label="📋 Trades Table", tab_id="trades"),
        dbc.Tab(label="💾 Raw Data", tab_id="raw"),
    ], id="tabs", active_tab="overview", className="nav-tabs")
