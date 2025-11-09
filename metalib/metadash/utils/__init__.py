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
    calculate_streak_analysis
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
    'calculate_streak_analysis'
]
