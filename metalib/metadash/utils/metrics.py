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
    returns = profit_df['profit_open'] / account_size
    
    sharpe_ratio = 0
    max_drawdown = 0
    max_drawdown_pct = 0
    
    if len(returns) > 1:
        # Sharpe ratio calculation
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        
        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown_pct = drawdown.min() * 100
        max_drawdown = (max_drawdown_pct / 100) * account_size
    
    # Profit factor calculation
    total_gains = profit_df[profit_df['profit_open'] > 0]['profit_open'].sum()
    total_losses = abs(profit_df[profit_df['profit_open'] < 0]['profit_open'].sum())
    profit_factor = total_gains / total_losses if total_losses != 0 else float('inf')
    
    # Win/Loss ratio
    avg_win = profit_df[profit_df['profit_open'] > 0]['profit_open'].mean() if len(profit_df[profit_df['profit_open'] > 0]) > 0 else 0
    avg_loss = abs(profit_df[profit_df['profit_open'] < 0]['profit_open'].mean()) if len(profit_df[profit_df['profit_open'] < 0]) > 0 else 0
    win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')
    
    return pd.Series({
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Max Drawdown (%)": max_drawdown_pct,
        "Profit Factor": profit_factor,
        "RRR": win_loss_ratio,
        "Account Roll (%)": profit_df["profit_open"].sum() / account_size * 100,
    })

def strategy_metrics(profit_df, account_size=100000):
    """
    Calculate strategy-level metrics
    """
    profit_df = profit_df.copy()
    profit_df["profit_open"] = profit_df["profit_open"] + profit_df["profit_close"]
    
    base_metrics = pd.Series({
        "Number of Trades": len(profit_df),
        "Total Profit": profit_df["profit_open"].sum(),
        "Average Profit by Trade": profit_df["profit_open"].mean(),
        "Win Rate (%)": 100 * (profit_df["profit_open"] > 0).mean(),
        "Loss Rate (%)": 100 * (profit_df["profit_open"] < 0).mean(),
    })
    
    additional_metrics = calculate_additional_metrics(profit_df, account_size)
    
    return pd.concat([base_metrics, additional_metrics])

def calculate_daily_performance(merged_deals):
    """
    Calculate daily performance metrics
    """
    overall_daily = merged_deals.copy()
    overall_daily['date'] = overall_daily['time_close'].dt.date if 'time_close' in overall_daily.columns else overall_daily['time_open'].dt.date
    overall_daily["profit"] = overall_daily["profit_open"] + overall_daily["profit_close"]
    daily_profit = overall_daily.groupby('date')['profit'].sum().reset_index()
    return daily_profit

def calculate_hourly_performance(merged_deals, strategy):
    """
    Calculate hourly performance for a specific strategy
    """
    filtered_deals = merged_deals[merged_deals['comment_open'] == strategy].copy()
    filtered_deals['hour'] = filtered_deals['time_open'].dt.hour
    filtered_deals = filtered_deals[
        ["profit_open", "profit_close", "comment_open", "symbol_open", "hour"]
    ].copy()
    
    hourly_perf = filtered_deals.groupby(["hour"]).apply(
        lambda x: strategy_metrics(x)
    )
    
    return hourly_perf

def calculate_streak_analysis(merged_deals):
    """
    Analyze winning and losing streaks
    """
    merged_deals_sorted = merged_deals.sort_values("time_open").copy()
    merged_deals_sorted["total_profit"] = merged_deals_sorted["profit_open"] + merged_deals_sorted["profit_close"]
    merged_deals_sorted["win"] = merged_deals_sorted["total_profit"] > 0
    merged_deals_sorted["streak_change"] = merged_deals_sorted["win"].ne(merged_deals_sorted["win"].shift())
    merged_deals_sorted["streak_id"] = merged_deals_sorted["streak_change"].cumsum()
    
    streaks = merged_deals_sorted.groupby("streak_id").agg({
        "win": "first",
        "total_profit": "sum",
        "time_open": "first",
        "symbol_open": "first",
        "comment_open": "first",
        "streak_id": "size"
    }).rename(columns={"streak_id": "streak_length"})
    
    return streaks
