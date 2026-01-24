"""
Components Module
Contains all tab components for the MetaDAsh application
"""
from .overview_tab import render_overview_tab
from .detailed_tab import render_detailed_tab, create_detailed_metrics_figure, create_hourly_chart
from .pnl_tab import render_pnl_tab
from .trades_tab import render_trades_tab, create_trades_table
from .raw_tab import render_raw_tab
from .strategy_type_tab import render_strategy_type_tab
from .log_tab import render_log_tab, create_log_stats_display, format_log_content
from .status_tab import render_status_tab, create_status_summary, create_status_table
from .welcome_tab import render_welcome_tab

__all__ = [
    'render_overview_tab',
    'render_detailed_tab',
    'create_detailed_metrics_figure',
    'create_hourly_chart',
    'render_pnl_tab',
    'render_trades_tab',
    'create_trades_table',
    'render_raw_tab',
    'render_strategy_type_tab',
    'render_log_tab',
    'create_log_stats_display',
    'format_log_content',
    'render_status_tab',
    'create_status_summary',
    'create_status_table',
    'render_welcome_tab',
]
