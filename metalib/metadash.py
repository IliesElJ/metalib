import streamlit as st
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
import plotly.graph_objects as go
import os
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="MetaDAsh")

# Initialize MT5 connection
@st.cache_resource
def initialize_mt5():
    if not mt5.initialize():
        st.error(f"MT5 initialization failed! Error code: {mt5.last_error()}")
        return False
    return True


# Function to get historical data
def get_historical_data(from_date, to_date):
    history_orders = mt5.history_orders_get(from_date, to_date)
    history_deals = mt5.history_deals_get(from_date, to_date)

    if history_orders is None:
        st.error(f"No history orders, error code={mt5.last_error()}")
        return None, None

    return history_orders, history_deals

# Save new deals and add the old ones
def save_and_retrieve_historical_deals(new_merged_deals):
    if os.path.exists("data/historical_merged_deals.pkl"):
        old_merged_deals = pd.read_pickle("data/historical_merged_deals.pkl")
    else:
        old_merged_deals = pd.DataFrame()

    merged_deals = pd.concat([old_merged_deals, new_merged_deals])
    merged_deals = merged_deals.drop_duplicates(subset=["symbol_open", "time_open", "position_id"], keep="first")
    merged_deals.to_pickle("historical_merged_deals.pkl")
    return merged_deals

# Calculate additional metrics
def calculate_additional_metrics(profit_df, account_size=100000):
    # Calculate metrics per trade (no resampling)
    returns = profit_df['profit_open'] / account_size

    # Calculate metrics
    sharpe_ratio = 0
    max_drawdown = 0
    max_drawdown_pct = 0

    if len(returns) > 1:
        # Simple approximation of Sharpe ratio using trade returns
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0

        # Calculate drawdown on trade-by-trade basis
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown_pct = drawdown.min() * 100  # Convert to percentage
        max_drawdown = (max_drawdown_pct / 100) * account_size

    # Calculate profit factor
    total_gains = profit_df[profit_df['profit_open'] > 0]['profit_open'].sum()
    total_losses = abs(profit_df[profit_df['profit_open'] < 0]['profit_open'].sum())
    profit_factor = total_gains / total_losses if total_losses != 0 else float('inf')

    # Calculate average win/loss ratio
    avg_win = profit_df[profit_df['profit_open'] > 0]['profit_open'].mean() if len(
        profit_df[profit_df['profit_open'] > 0]) > 0 else 0
    avg_loss = abs(profit_df[profit_df['profit_open'] < 0]['profit_open'].mean()) if len(
        profit_df[profit_df['profit_open'] < 0]) > 0 else 0
    win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')

    return pd.Series({
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Max Drawdown (%)": max_drawdown_pct,
        "Profit Factor": profit_factor,
        "RRR": win_loss_ratio,
        "Account Roll (%)": profit_df["profit_open"].sum() / account_size * 100,
    })


# Strategy metrics function
def strategy_metrics(profit_df, account_size=100000):
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

# Main dashboard
def main():
    st.title("MetaDAsh")

    # Sidebar for settings and controls
    st.sidebar.header("Settings")

    # Date range selection
    default_start_date = datetime(2020, 1, 1)
    default_end_date = datetime.now()

    start_date = st.sidebar.date_input("Start Date", default_start_date)
    end_date = st.sidebar.date_input("End Date", default_end_date)

    # Convert to datetime with time
    from_date = datetime.combine(start_date, datetime.min.time())
    to_date = datetime.combine(end_date, datetime.max.time())

    # Account size for calculations
    account_size = st.sidebar.number_input("Account Size ($)", value=100000, min_value=10000, step=1000)

    # Connect button
    if st.sidebar.button("Connect to MT5"):
        if initialize_mt5():
            st.sidebar.success("Connected to MT5 successfully pelo!")
        else:
            st.sidebar.error("Failed to connect to MT5 pelo")
            return

    # Fetch data button
    if st.sidebar.button("Fetch Trading Data"):
        with st.spinner("Fetching data from MT5..."):
            history_orders, history_deals = get_historical_data(from_date, to_date)

            if history_orders is None or history_deals is None:
                st.error("Failed to retrieve data")
                return

            st.success(f"Retrieved {len(history_orders)} orders and {len(history_deals)} deals")

            if len(history_orders) > 0:
                st.write(f"history_orders_get({from_date}, {to_date}) = {len(history_orders)}")

                # Process data
                df_deals = pd.DataFrame(list(history_deals), columns=history_deals[0]._asdict().keys())

                # Create tabs for different analyses
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Detailed Analysis", "PnL Performance", "Trades Table", "Raw Data"])
                main_container = st.container()

                with main_container:

                    with tab1:
                        st.header("Trading Overview")

                        # Filter deals
                        df_deals_opens = df_deals[df_deals["comment"].str.contains("meta")]
                        df_deals_closes = df_deals[
                            (df_deals["comment"].str.contains("sl")) |
                            (df_deals["comment"].str.contains("tp")) |
                            (df_deals["comment"].str.contains("Close"))
                            ]

                        # Merge open and close deals
                        merged_deals = df_deals_closes.merge(
                            df_deals_opens, on="position_id", suffixes=("_close", "_open")
                        )

                        # Process time columns
                        merged_deals["time_open"] = pd.to_datetime(merged_deals["time_open"], unit='s')
                        if "time_close" in merged_deals.columns:
                            merged_deals["time_close"] = pd.to_datetime(merged_deals["time_close"], unit='s')

                        merged_deals = save_and_retrieve_historical_deals(merged_deals)

                        # Calculate strategy metrics - now using symbol_open and comment_open for grouping
                        strategy_metrics_df = merged_deals[
                            ["profit_open", "profit_close", "comment_open", "symbol_open", "time_open"]].copy()

                        # Group by strategy and symbol (using open values)
                        grouped_metrics = strategy_metrics_df.groupby(["comment_open", "symbol_open"]).apply(
                            lambda x: strategy_metrics(x, account_size)
                        )

                        # Overall account info box
                        st.subheader("📈 Account Overview Pelo")
                        account_info = mt5.account_info()._asdict()
                        overall_daily = merged_deals.copy()
                        overall_daily['date'] = overall_daily['time_close'].dt.date
                        overall_daily["profit"] = overall_daily["profit_open"] + overall_daily["profit_close"]
                        daily_profit = overall_daily.groupby('date')['profit'].sum().reset_index()

                        st.markdown("### MetaTrader Account Info")
                        st.container()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Balance", f"${account_info['balance']:.2f}")
                        with col2:
                            st.metric("Equity", f"${account_info['equity']:.2f}")
                        with col3:
                            st.metric("Margin", f"${account_info['margin']:.2f}")

                        daily_fig = go.Figure()
                        daily_fig.add_trace(go.Scatter(
                            x=daily_profit['date'],
                            y=daily_profit['profit'],
                            mode='lines+markers',
                            name='Daily Profit',
                            line=dict(color='royalblue', width=2),
                            marker=dict(size=6)
                        ))
                        daily_fig.update_layout(
                            title="Overall Daily Performance",
                            xaxis_title="Date",
                            yaxis_title="Profit",
                            xaxis=dict(
                                tickformat='%Y-%m-%d'
                            )
                        )
                        st.plotly_chart(daily_fig, use_container_width=True)

                        # Display metrics
                        st.subheader("Strategy Performance Metrics Pelo")
                        st.dataframe(grouped_metrics.style.format(precision=2).background_gradient(axis=0, cmap="YlOrRd"), use_container_width=True)

                        # Prepare data for plotting - convert to long format for plotly
                        plot_metrics = ["Total Profit", "Average Profit by Trade", "Win Rate (%)", "Loss Rate (%)"]
                        plot_data = grouped_metrics.reset_index()

                        # Create a figure manually instead of using px.bar with multiple columns
                        fig = go.Figure()

                        # Create a proper label for x-axis (using open values)
                        plot_data['strategy_symbol'] = plot_data['comment_open'] + ' - ' + plot_data['symbol_open']

                        # Add bars for each metric
                        for metric in plot_metrics:
                            fig.add_trace(go.Bar(
                                x=plot_data['strategy_symbol'],
                                y=plot_data[metric],
                                name=metric
                            ))

                        fig.update_layout(
                            title="Strategy Performance Comparison",
                            xaxis_title="Strategy - Symbol",
                            yaxis_title="Value",
                            barmode='group'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        st.header("Detailed Analysis Pelo")

                        # Select which metrics to display
                        metrics_to_plot = st.multiselect(
                            "Select metrics to display:",
                            options=["Total Profit", "Win Rate (%)", "Sharpe Ratio", "Max Drawdown (%)", "Profit Factor"],
                            default=["Total Profit", "Win Rate (%)", "Sharpe Ratio"]
                        )

                        if metrics_to_plot:
                            # Create a figure manually for selected metrics
                            fig = go.Figure()

                            for metric in metrics_to_plot:
                                fig.add_trace(go.Bar(
                                    x=plot_data['strategy_symbol'],
                                    y=plot_data[metric],
                                    name=metric
                                ))

                            fig.update_layout(
                                title="Detailed Strategy Metrics",
                                xaxis_title="Strategy - Symbol",
                                yaxis_title="Value",
                                barmode='group'
                            )

                            st.plotly_chart(fig, use_container_width=True)

                        # Trade duration analysis
                        if "time_close" in merged_deals.columns:
                            merged_deals["duration"] = (merged_deals["time_close"] - merged_deals[
                                "time_open"]).dt.total_seconds() / 3600  # hours

                            st.subheader("Trade Duration Analysis")

                            # Use go.Figure directly instead of px.histogram
                            duration_fig = go.Figure()

                            # Group by symbol_open instead of symbol_close

                            for symbol in merged_deals["symbol_open"].unique():
                                symbol_data = merged_deals[merged_deals["symbol_open"] == symbol]

                                duration_fig.add_trace(go.Histogram(
                                    x=symbol_data["duration"],
                                    name=symbol,
                                    opacity=0.7,
                                    nbinsx=50
                                ))

                            duration_fig.update_layout(
                                title="Trade Duration Distribution (hours)",
                                xaxis_title="Duration (hours)",
                                yaxis_title="Count",
                                barmode='overlay'
                            )

                            st.plotly_chart(duration_fig, use_container_width=True)

                        # Hourly performance for selected strategy/symbol
                        st.subheader("Hourly Average Performance")
                        strategy_choice = st.selectbox("Select Strategy/Symbol:", merged_deals['comment_open'].unique())

                        filtered_deals = merged_deals[merged_deals['comment_open'] == strategy_choice].copy()
                        filtered_deals['hour'] = filtered_deals['time_open'].dt.hour
                        filtered_deals = filtered_deals[
                            ["profit_open", "profit_close", "comment_open", "symbol_open", "hour"]].copy()

                        # Group by strategy and symbol (using open values)
                        hourly_perf = filtered_deals.groupby(["hour"]).apply(
                            lambda x: strategy_metrics(x, account_size)
                        )

                        hourly_fig = go.Figure()

                        for col in hourly_perf.columns:
                            hourly_perf[col] = hourly_perf[col].fillna(0)

                            hourly_fig.add_trace(go.Bar(
                                x=hourly_perf.index,
                                y=hourly_perf[col],
                                name='Hourly Avg Profit'
                            ))

                        hourly_fig.update_layout(
                            title=f"Hourly Performance for {strategy_choice}",
                            xaxis_title="Hour of Day",
                            yaxis_title="Average Profit"
                        )
                        st.plotly_chart(hourly_fig, use_container_width=True)

                    with tab3:
                        st.header("PnL Performance Pelo")

                        # Calculate trade-by-trade PnL
                        merged_deals["total_profit"] = merged_deals["profit_open"] + merged_deals["profit_close"]
                        merged_deals_sorted = merged_deals.sort_values("time_open")
                        merged_deals_sorted["cumulative_profit"] = merged_deals_sorted["total_profit"].cumsum()

                        # Create three columns for side-by-side charts
                        col1, col2, col3 = st.columns(3)

                        # Account equity chart - trade by trade
                        equity = merged_deals_sorted.copy()
                        equity["equity"] = account_size + equity["cumulative_profit"]

                        equity_fig = go.Figure()
                        equity_fig.add_trace(go.Scatter(
                            x=equity["time_open"],
                            y=equity["equity"],
                            mode="lines",
                            name="Account Equity"
                        ))
                        equity_fig.update_layout(
                            title="Account Equity Curve",
                            xaxis_title="Date",
                            yaxis_title="Equity ($)",
                            height=400
                        )

                        with col1:
                            st.plotly_chart(equity_fig, use_container_width=True)

                        # Drawdown chart
                        equity["running_max"] = equity["equity"].cummax()
                        equity["drawdown"] = (equity["equity"] / equity["running_max"] - 1) * 100

                        drawdown_fig = go.Figure()
                        drawdown_fig.add_trace(go.Scatter(
                            x=equity["time_open"],
                            y=equity["drawdown"],
                            mode="lines",
                            line=dict(color='red'),
                            name="Drawdown"
                        ))
                        drawdown_fig.update_layout(
                            title="Drawdown (%)",
                            xaxis_title="Date",
                            yaxis_title="Drawdown (%)",
                            height=400
                        )

                        with col2:
                            st.plotly_chart(drawdown_fig, use_container_width=True)

                        # Profit by symbol chart
                        symbols = merged_deals_sorted["symbol_open"].unique()
                        symbol_fig = go.Figure()

                        for symbol in symbols:
                            symbol_data = merged_deals_sorted[merged_deals_sorted["symbol_open"] == symbol]
                            symbol_data = symbol_data.sort_values("time_open")
                            symbol_data["cumulative_profit"] = symbol_data["total_profit"].cumsum()

                            symbol_fig.add_trace(go.Scatter(
                                x=symbol_data["time_open"],
                                y=symbol_data["cumulative_profit"],
                                mode='lines',
                                name=symbol
                            ))

                        symbol_fig.update_layout(
                            title="Cumulative Profit by Symbol",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Profit ($)",
                            height=400
                        )

                        with col3:
                            st.plotly_chart(symbol_fig, use_container_width=True)

                        # Trade scatter plot - profit vs time (using symbol_open)
                        st.subheader("Individual Trade Performance")

                        scatter_fig = go.Figure()

                        for symbol in symbols:
                            symbol_data = merged_deals_sorted[merged_deals_sorted["symbol_open"] == symbol]

                            scatter_fig.add_trace(go.Scatter(
                                x=symbol_data["time_open"],
                                y=symbol_data["total_profit"],
                                mode="markers",
                                name=symbol,
                                marker=dict(
                                    size=np.abs(symbol_data["total_profit"]) / max(1, np.abs(
                                        symbol_data["total_profit"]).max()) * 20 + 5
                                ),
                                text=symbol_data["comment_open"]  # Using comment_open for hover info
                            ))

                        scatter_fig.update_layout(
                            title="Individual Trade Performance",
                            xaxis_title="Date",
                            yaxis_title="Profit/Loss ($)"
                        )

                        st.plotly_chart(scatter_fig, use_container_width=True)

                        # Trade streak analysis (no change in grouping needed)
                        st.subheader("Win/Loss Streaks")

                        merged_deals_sorted["win"] = merged_deals_sorted["total_profit"] > 0

                        # Create a new column that identifies when the win/loss streak changes
                        merged_deals_sorted["streak_change"] = merged_deals_sorted["win"].ne(
                            merged_deals_sorted["win"].shift())

                        # Assign a streak ID to each streak
                        merged_deals_sorted["streak_id"] = merged_deals_sorted["streak_change"].cumsum()

                        # Group by streak ID to calculate streak length and total profit
                        # Now using symbol_open instead of symbol_close
                        streaks = merged_deals_sorted.groupby("streak_id").agg({
                            "win": "first",
                            "total_profit": "sum",
                            "time_open": "first",
                            "symbol_open": "first",  # Changed to symbol_open
                            "comment_open": "first",  # Added comment_open
                            "streak_id": "size"
                        }).rename(columns={"streak_id": "streak_length"})

                        # Plot streak length
                        streak_fig = go.Figure()

                        # Create separate traces for wins and losses for better coloring
                        win_streaks = streaks[streaks["win"]]
                        loss_streaks = streaks[~streaks["win"]]

                        streak_fig.add_trace(go.Bar(
                            x=win_streaks.index,
                            y=win_streaks["streak_length"],
                            name="Win Streaks",
                            marker_color="green",
                            text=win_streaks["total_profit"].round(2),
                            hovertemplate="Streak ID: %{x}<br>Length: %{y}<br>Profit: $%{text}<br>Symbol: %{customdata[0]}<br>Strategy: %{customdata[1]}",
                            customdata=np.stack((win_streaks["symbol_open"], win_streaks["comment_open"]), axis=1)
                        ))

                        streak_fig.add_trace(go.Bar(
                            x=loss_streaks.index,
                            y=loss_streaks["streak_length"],
                            name="Loss Streaks",
                            marker_color="red",
                            text=loss_streaks["total_profit"].round(2),
                            hovertemplate="Streak ID: %{x}<br>Length: %{y}<br>Loss: $%{text}<br>Symbol: %{customdata[0]}<br>Strategy: %{customdata[1]}",
                            customdata=np.stack((loss_streaks["symbol_open"], loss_streaks["comment_open"]), axis=1)
                        ))

                        streak_fig.update_layout(
                            title="Win/Loss Streak Lengths",
                            xaxis_title="Streak ID",
                            yaxis_title="Streak Length"
                        )

                        st.plotly_chart(streak_fig, use_container_width=True)

                        # Add analysis by strategy (comment_open)
                        st.subheader("Performance by Strategy")

                        strategy_perf = merged_deals.groupby("comment_open")["total_profit"].agg(
                            ["sum", "mean", "count", "std"]
                        ).reset_index()
                        strategy_perf.columns = ["Strategy", "Total Profit", "Average Profit", "Number of Trades",
                                                 "Std Dev"]
                        strategy_perf["Win Rate (%)"] = merged_deals.groupby("comment_open")["total_profit"].apply(
                            lambda x: 100 * (x > 0).mean()
                        ).values

                        st.dataframe(strategy_perf, use_container_width=True)

                    with tab4:
                        st.header("Trades table")
                        st.subheader("Voila les trades par bot pelo.")

                        instances = list(merged_deals.loc[:, "comment_open"].unique())

                        selected_instance = st.selectbox(
                            "Choose the bot mec",
                            instances,
                        )

                        instance_deals = merged_deals.loc[merged_deals["comment_open"] == selected_instance]
                        instance_deals = instance_deals[["symbol_open", "time_open", "time_close", "total_profit", "price_open", "price_close"]]
                        instance_deals.columns = ["Symbol", "Open Time", "Close Time", "Total Profit", "Open Price", "Close Price"]

                        st.dataframe(instance_deals, use_container_width=True)

                    with tab5:
                        st.header("Raw Data")
                        st.subheader("Merged Deals Data")
                        st.dataframe(merged_deals, use_container_width=True)

                        # Add download button for the data
                        @st.cache_data
                        def convert_df_to_csv(df):
                            return df.to_csv(index=False).encode('utf-8')

                        csv = convert_df_to_csv(merged_deals)
                        st.download_button(
                            "Download Merged Deals CSV",
                            csv,
                            "merged_deals.csv",
                            "text/csv",
                            key='download-csv'
                        )

                        st.subheader("Orders Data")
                        orders_df = pd.DataFrame(list(history_orders), columns=history_orders[0]._asdict().keys())
                        st.dataframe(orders_df, use_container_width=True)

                        # Add download button for orders data
                        csv_orders = convert_df_to_csv(orders_df)
                        st.download_button(
                            "Download Orders CSV",
                            csv_orders,
                            "orders.csv",
                            "text/csv",
                            key='download-orders-csv'
                        )


if __name__ == "__main__":
    main()