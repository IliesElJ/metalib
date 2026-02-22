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
from .status_tab import render_status_tab, create_status_summary, create_status_table, create_pm2_process_table
from .welcome_tab import render_welcome_tab
from .instance_trades_tab import (
    render_instance_trades_tab,
    get_filtered_strategy_instances,
    get_dates_for_instance,
    create_instance_trades_grid,
    create_instance_trades_stats,
    create_trade_candlestick_chart,
)
from .calibration_tab import (
    render_calibration_tab,
    create_results_table,
    create_results_chart,
    create_asset_matrices,
    DEFAULT_STRATEGY_PARAMS,
)
from .indicators_tab import (
    render_indicators_tab,
    create_indicator_chart,
    create_indicator_info_cards,
    pull_price_for_indicators,
)

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
    'create_pm2_process_table',
    'render_welcome_tab',
    'render_instance_trades_tab',
    'get_filtered_strategy_instances',
    'get_dates_for_instance',
    'create_instance_trades_grid',
    'create_instance_trades_stats',
    'create_trade_candlestick_chart',
    'render_calibration_tab',
    'create_results_table',
    'create_results_chart',
    'create_asset_matrices',
    'DEFAULT_STRATEGY_PARAMS',
    'render_indicators_tab',
    'create_indicator_chart',
    'create_indicator_info_cards',
    'pull_price_for_indicators',
]
