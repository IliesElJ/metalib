"""
Utils Module
Contains utility functions for MT5 connection and metrics calculation
"""
from .mt5_utils import (
    initialize_mt5,
    get_historical_data,
    process_deals_data,
    get_account_info
)
from .metrics import (
    calculate_additional_metrics,
    strategy_metrics,
    calculate_daily_performance,
    calculate_hourly_performance,
    calculate_streak_analysis,
    extract_strategy_type,
    calculate_strategy_type_metrics,
    calculate_strategy_type_cumulative
)
from .log_utils import (
    get_logs_directory,
    get_available_log_files,
    parse_log_filename,
    get_strategy_instances,
    get_dates_for_strategy,
    read_log_file,
    parse_log_content,
    get_log_statistics
)

__all__ = [
    'initialize_mt5',
    'get_historical_data',
    'process_deals_data',
    'get_account_info',
    'calculate_additional_metrics',
    'strategy_metrics',
    'calculate_daily_performance',
    'calculate_hourly_performance',
    'calculate_streak_analysis',
    'extract_strategy_type',
    'calculate_strategy_type_metrics',
    'calculate_strategy_type_cumulative',
    'get_logs_directory',
    'get_available_log_files',
    'parse_log_filename',
    'get_strategy_instances',
    'get_dates_for_strategy',
    'read_log_file',
    'parse_log_content',
    'get_log_statistics'
]
