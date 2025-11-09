"""
Components Module
Contains all tab components for the MetaDAsh application
"""
from .overview_tab import render_overview_tab
from .detailed_tab import render_detailed_tab, create_detailed_metrics_figure
from .pnl_tab import render_pnl_tab
from .trades_tab import render_trades_tab, create_trades_table
from .raw_tab import render_raw_tab

__all__ = [
    'render_overview_tab',
    'render_detailed_tab',
    'create_detailed_metrics_figure',
    'render_pnl_tab',
    'render_trades_tab',
    'create_trades_table',
    'render_raw_tab'
]
