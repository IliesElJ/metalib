"""
MT5 Utilities Module
Handles MetaTrader 5 connection and data processing
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime


def initialize_mt5():
    """
    Initialize MT5 connection
    """
    if not mt5.initialize():
        return False, f"MT5 initialization failed! Error code: {mt5.last_error()}"
    return True, "Connected to MT5 successfully pelo!"


def get_historical_data(from_date, to_date):
    """
    Get historical data from MT5
    """
    history_orders = mt5.history_orders_get(from_date, to_date)
    history_deals = mt5.history_deals_get(from_date, to_date)

    if history_orders is None:
        return None, None, f"No history orders, error code={mt5.last_error()}"

    return history_orders, history_deals, None


def process_deals_data(history_deals):
    """
    Process raw deals data from MT5.
    Matches open/close deals using position_id and the entry field.
    """
    if not history_deals or len(history_deals) == 0:
        return None

    # Convert to DataFrame
    df_deals = pd.DataFrame(
        list(history_deals), columns=history_deals[0]._asdict().keys()
    )

    # Keep only actual trades (BUY=0, SELL=1), exclude balance/credit/etc.
    df_deals = df_deals[df_deals["type"].isin([0, 1])]

    # Opens: ENTRY_IN (0) with our bot identifier in comment
    df_deals_opens = df_deals[
        (df_deals["entry"] == 0) & (df_deals["comment"].str.contains("meta", na=False))
    ]
    # Closes: ENTRY_OUT (1) — all exits, regardless of comment
    df_deals_closes = df_deals[df_deals["entry"] == 1]

    if df_deals_opens.empty or df_deals_closes.empty:
        return None

    # Merge open and close deals on position_id
    merged_deals = df_deals_closes.merge(
        df_deals_opens, on="position_id", suffixes=("_close", "_open")
    )

    # Process time columns
    merged_deals["time_open"] = pd.to_datetime(merged_deals["time_open"], unit="s")
    if "time_close" in merged_deals.columns:
        merged_deals["time_close"] = pd.to_datetime(
            merged_deals["time_close"], unit="s"
        )

    return merged_deals


def get_account_info():
    """
    Get MT5 account information
    """
    account_info = mt5.account_info()
    if account_info:
        return account_info._asdict()
    return {"balance": 0, "equity": 0, "margin": 0, "profit": 0, "credit": 0}


def get_candles_for_trade(symbol, time_open, time_close, buffer_minutes=30):
    """
    Get OHLC candle data for a trade with buffer before and after.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'DE40')
        time_open: Trade open time (datetime)
        time_close: Trade close time (datetime), can be None
        buffer_minutes: Minutes of buffer before and after the trade

    Returns:
        DataFrame with columns: time, open, high, low, close, volume
        Returns None if no data available
    """
    from datetime import timedelta

    # Always use M1 timeframe for detailed view
    timeframe = mt5.TIMEFRAME_M1

    # Calculate date range with buffer
    buffer = timedelta(minutes=buffer_minutes)
    start_time = time_open - buffer
    end_time = (time_close if time_close else time_open) + buffer

    # Fetch candles from MT5
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)

    if rates is None or len(rates) == 0:
        # Try with a different symbol format (some brokers add suffixes)
        # Try common variations
        for suffix in ["", ".a", ".b", ".pro", "_SB"]:
            alt_symbol = symbol + suffix if suffix else symbol
            rates = mt5.copy_rates_range(alt_symbol, timeframe, start_time, end_time)
            if rates is not None and len(rates) > 0:
                break

    if rates is None or len(rates) == 0:
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Rename columns to match expected format
    df = df.rename(columns={"tick_volume": "volume"})

    return df[["time", "open", "high", "low", "close", "volume"]]
