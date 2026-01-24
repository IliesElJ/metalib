"""
Metrics Calculation Module
Contains functions for calculating trading metrics
"""

import pandas as pd
import numpy as np


def calculate_additional_metrics(profit_df, account_size=100000):
    """
    Calculate additional trading metrics
    """
    returns = profit_df["profit_open"] / account_size

    sharpe_ratio = 0
    max_drawdown = 0
    max_drawdown_pct = 0

    if len(returns) > 1:
        # Sharpe ratio calculation
        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        )

        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown_pct = drawdown.min() * 100
        max_drawdown = (max_drawdown_pct / 100) * account_size

    # Profit factor calculation
    total_gains = profit_df[profit_df["profit_open"] > 0]["profit_open"].sum()
    total_losses = abs(profit_df[profit_df["profit_open"] < 0]["profit_open"].sum())
    profit_factor = total_gains / total_losses if total_losses != 0 else float("inf")

    # Win/Loss ratio
    avg_win = (
        profit_df[profit_df["profit_open"] > 0]["profit_open"].mean()
        if len(profit_df[profit_df["profit_open"] > 0]) > 0
        else 0
    )
    avg_loss = (
        abs(profit_df[profit_df["profit_open"] < 0]["profit_open"].mean())
        if len(profit_df[profit_df["profit_open"] < 0]) > 0
        else 0
    )
    win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else float("inf")

    return pd.Series(
        {
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Max Drawdown (%)": max_drawdown_pct,
            "Profit Factor": profit_factor,
            "RRR": win_loss_ratio,
            "Account Roll (%)": profit_df["profit_open"].sum() / account_size * 100,
        }
    )


def strategy_metrics(profit_df, account_size=100000):
    """
    Calculate strategy-level metrics
    """
    profit_df = profit_df.copy()
    profit_df["profit_open"] = profit_df["profit_open"] + profit_df["profit_close"]

    base_metrics = pd.Series(
        {
            "Number of Trades": len(profit_df),
            "Total Profit": profit_df["profit_open"].sum(),
            "Average Profit by Trade": profit_df["profit_open"].mean(),
            "Win Rate (%)": 100 * (profit_df["profit_open"] > 0).mean(),
            "Loss Rate (%)": 100 * (profit_df["profit_open"] < 0).mean(),
        }
    )

    additional_metrics = calculate_additional_metrics(profit_df, account_size)

    return pd.concat([base_metrics, additional_metrics])


def calculate_daily_performance(merged_deals):
    """
    Calculate daily performance metrics
    """
    overall_daily = merged_deals.copy()
    overall_daily["date"] = (
        overall_daily["time_close"].dt.date
        if "time_close" in overall_daily.columns
        else overall_daily["time_open"].dt.date
    )
    overall_daily["profit"] = (
        overall_daily["profit_open"] + overall_daily["profit_close"]
    )
    daily_profit = overall_daily.groupby("date")["profit"].sum().reset_index()
    return daily_profit


def calculate_hourly_performance(merged_deals, strategy):
    """
    Calculate hourly performance for a specific strategy
    """
    filtered_deals = merged_deals[merged_deals["comment_open"] == strategy].copy()
    filtered_deals["hour"] = filtered_deals["time_open"].dt.hour
    filtered_deals = filtered_deals[
        ["profit_open", "profit_close", "comment_open", "symbol_open", "hour"]
    ].copy()

    hourly_perf = filtered_deals.groupby(["hour"]).apply(lambda x: strategy_metrics(x))

    return hourly_perf


def calculate_streak_analysis(merged_deals):
    """
    Analyze winning and losing streaks
    """
    merged_deals_sorted = merged_deals.sort_values("time_open").copy()
    merged_deals_sorted["total_profit"] = (
        merged_deals_sorted["profit_open"] + merged_deals_sorted["profit_close"]
    )
    merged_deals_sorted["win"] = merged_deals_sorted["total_profit"] > 0
    merged_deals_sorted["streak_change"] = merged_deals_sorted["win"].ne(
        merged_deals_sorted["win"].shift()
    )
    merged_deals_sorted["streak_id"] = merged_deals_sorted["streak_change"].cumsum()

    streaks = (
        merged_deals_sorted.groupby("streak_id")
        .agg(
            {
                "win": "first",
                "total_profit": "sum",
                "time_open": "first",
                "symbol_open": "first",
                "comment_open": "first",
                "streak_id": "size",
            }
        )
        .rename(columns={"streak_id": "streak_length"})
    )

    return streaks


def extract_strategy_type(comment):
    """
    Extract strategy type from comment/tag by taking the first part before underscore.

    Args:
        comment: Strategy comment/tag string

    Returns:
        Strategy type (first part before '_') or the full comment if no underscore exists
    """
    if pd.isna(comment) or not isinstance(comment, str):
        return "unknown"

    parts = comment.split("_")
    return parts[0] if parts else comment


def calculate_strategy_type_metrics(merged_deals, account_size=100000):
    """
    Calculate aggregated metrics by strategy type.

    Args:
        merged_deals: DataFrame with merged trading deals
        account_size: Account size for calculations

    Returns:
        DataFrame with aggregated metrics by strategy type
    """
    deals_copy = merged_deals.copy()

    # Extract strategy type from comment
    deals_copy["strategy_type"] = deals_copy["comment_open"].apply(
        extract_strategy_type
    )

    # Calculate total profit
    deals_copy["total_profit"] = deals_copy["profit_open"] + deals_copy["profit_close"]

    # Determine the datetime column to use for last trade
    if "time_close" in deals_copy.columns:
        datetime_col = "time_close"
    else:
        datetime_col = "time_open"

    # Group by strategy type and calculate metrics
    grouped = deals_copy.groupby("strategy_type")

    metrics_list = []
    for strategy_type, group in grouped:
        profit_df = group[
            ["profit_open", "profit_close", "total_profit", "time_open"]
        ].copy()
        profit_df["profit_open"] = profit_df[
            "total_profit"
        ]  # Use total profit for metrics

        # Get last trade datetime
        last_trade = group[datetime_col].max()

        base_metrics = pd.Series(
            {
                "Strategy Type": strategy_type,
                "Number of Trades": len(group),
                "Total Profit": group["total_profit"].sum(),
                "Average Profit per Trade": group["total_profit"].mean(),
                "Win Rate (%)": 100 * (group["total_profit"] > 0).mean(),
                "Loss Rate (%)": 100 * (group["total_profit"] < 0).mean(),
                "Last Trade": last_trade,
            }
        )

        additional_metrics = calculate_additional_metrics(profit_df, account_size)
        combined = pd.concat([base_metrics, additional_metrics])
        metrics_list.append(combined)

    result_df = pd.DataFrame(metrics_list)
    result_df = result_df.set_index("Strategy Type")

    return result_df


def calculate_strategy_type_cumulative(merged_deals):
    """
    Calculate cumulative performance over time by strategy type.

    Args:
        merged_deals: DataFrame with merged trading deals

    Returns:
        DataFrame with cumulative profit by date and strategy type
    """
    deals_copy = merged_deals.copy()

    # Extract strategy type
    deals_copy["strategy_type"] = deals_copy["comment_open"].apply(
        extract_strategy_type
    )

    # Calculate total profit
    deals_copy["total_profit"] = deals_copy["profit_open"] + deals_copy["profit_close"]

    # Use close time if available, otherwise open time
    if "time_close" in deals_copy.columns:
        deals_copy["datetime"] = pd.to_datetime(deals_copy["time_close"])
    else:
        deals_copy["datetime"] = pd.to_datetime(deals_copy["time_open"])

    # Sort by datetime
    deals_copy = deals_copy.sort_values("datetime")

    # Calculate cumulative profit by strategy type
    cumulative_data = []
    for strategy_type in deals_copy["strategy_type"].unique():
        strategy_deals = deals_copy[deals_copy["strategy_type"] == strategy_type].copy()
        strategy_deals["cumulative_profit"] = strategy_deals["total_profit"].cumsum()

        cumulative_data.append(
            strategy_deals[
                ["datetime", "strategy_type", "cumulative_profit", "total_profit"]
            ]
        )

    if cumulative_data:
        result_df = pd.concat(cumulative_data, ignore_index=True)
        return result_df.sort_values("datetime")

    return pd.DataFrame()
