import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, date
import plotly.graph_objects as go
import os
import warnings
import pickle
import base64
import io

warnings.filterwarnings("ignore")

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "MetaDAsh"

# Global variables to store data
stored_data = {
    'history_orders': None,
    'history_deals': None,
    'merged_deals': None,
    'account_size': 100000
}


# Initialize MT5 connection
def initialize_mt5():
    if not mt5.initialize():
        return False, f"MT5 initialization failed! Error code: {mt5.last_error()}"
    return True, "Connected to MT5 successfully pelo!"


# Function to get historical data
def get_historical_data(from_date, to_date):
    history_orders = mt5.history_orders_get(from_date, to_date)
    history_deals = mt5.history_deals_get(from_date, to_date)

    if history_orders is None:
        return None, None, f"No history orders, error code={mt5.last_error()}"

    return history_orders, history_deals, None


# Save new deals and add the old ones
def save_and_retrieve_historical_deals(new_merged_deals):
    if os.path.exists("historical_merged_deals.pkl"):
        with open("historical_merged_deals.pkl", "rb") as f:
            old_merged_deals = pickle.load(f)
    else:
        old_merged_deals = pd.DataFrame()

    merged_deals = pd.concat([old_merged_deals, new_merged_deals])
    merged_deals = merged_deals.drop_duplicates(subset=["symbol_open", "time_open", "position_id"], keep="first")

    with open("historical_merged_deals.pkl", "wb") as f:
        pickle.dump(merged_deals, f)

    return merged_deals


# Calculate additional metrics
def calculate_additional_metrics(profit_df, account_size=100000):
    returns = profit_df['profit_open'] / account_size

    sharpe_ratio = 0
    max_drawdown = 0
    max_drawdown_pct = 0

    if len(returns) > 1:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0

        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown_pct = drawdown.min() * 100
        max_drawdown = (max_drawdown_pct / 100) * account_size

    total_gains = profit_df[profit_df['profit_open'] > 0]['profit_open'].sum()
    total_losses = abs(profit_df[profit_df['profit_open'] < 0]['profit_open'].sum())
    profit_factor = total_gains / total_losses if total_losses != 0 else float('inf')

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


# Layout
app.layout = dbc.Container([
    dcc.Store(id='data-store'),

    dbc.Row([
        dbc.Col([
            html.H1("MetaDAsh", className="text-center mb-4"),
        ])
    ]),

    dbc.Row([
        # Sidebar
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Settings", className="mb-3"),

                    html.Label("Start Date"),
                    dcc.DatePickerSingle(
                        id='start-date',
                        date=date(2020, 1, 1),
                        display_format='YYYY-MM-DD',
                        className="mb-3"
                    ),

                    html.Label("End Date"),
                    dcc.DatePickerSingle(
                        id='end-date',
                        date=date.today(),
                        display_format='YYYY-MM-DD',
                        className="mb-3"
                    ),

                    html.Label("Account Size ($)"),
                    dbc.Input(
                        id='account-size',
                        type='number',
                        value=100000,
                        min=10000,
                        step=1000,
                        className="mb-3"
                    ),

                    dbc.Button("Connect to MT5", id="connect-btn", color="primary", className="mb-2 w-100"),
                    dbc.Button("Fetch Trading Data", id="fetch-btn", color="success", className="mb-2 w-100"),

                    html.Div(id="connection-status", className="mt-3"),
                    html.Div(id="fetch-status", className="mt-3"),
                ])
            ])
        ], width=3),

        # Main content area
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Overview", tab_id="overview"),
                dbc.Tab(label="Detailed Analysis", tab_id="detailed"),
                dbc.Tab(label="PnL Performance", tab_id="pnl"),
                dbc.Tab(label="Trades Table", tab_id="trades"),
                dbc.Tab(label="Raw Data", tab_id="raw"),
            ], id="tabs", active_tab="overview"),

            html.Div(id="tab-content", className="mt-4")
        ], width=9)
    ])
], fluid=True)


# Callbacks
@app.callback(
    Output('connection-status', 'children'),
    Input('connect-btn', 'n_clicks'),
    prevent_initial_call=True
)
def connect_mt5(n_clicks):
    if n_clicks:
        success, message = initialize_mt5()
        if success:
            return dbc.Alert(message, color="success")
        else:
            return dbc.Alert(message, color="danger")
    return ""


@app.callback(
    [Output('data-store', 'data'),
     Output('fetch-status', 'children')],
    [Input('fetch-btn', 'n_clicks')],
    [State('start-date', 'date'),
     State('end-date', 'date'),
     State('account-size', 'value')],
    prevent_initial_call=True
)
def fetch_data(n_clicks, start_date, end_date, account_size):
    if n_clicks:
        from_date = datetime.strptime(start_date, '%Y-%m-%d')
        to_date = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)

        history_orders, history_deals, error = get_historical_data(from_date, to_date)

        if error:
            return None, dbc.Alert(error, color="danger")

        if history_orders is None or history_deals is None:
            return None, dbc.Alert("Failed to retrieve data", color="danger")

        message = f"Retrieved {len(history_orders)} orders and {len(history_deals)} deals"

        # Process data
        if len(history_orders) > 0 and len(history_deals) > 0:
            df_deals = pd.DataFrame(list(history_deals), columns=history_deals[0]._asdict().keys())

            # Filter deals
            df_deals_opens = df_deals[df_deals["comment"].str.contains("meta", na=False)]
            df_deals_closes = df_deals[
                (df_deals["comment"].str.contains("sl", na=False)) |
                (df_deals["comment"].str.contains("tp", na=False)) |
                (df_deals["comment"].str.contains("Close", na=False))
                ]

            if not df_deals_opens.empty and not df_deals_closes.empty:
                # Merge open and close deals
                merged_deals = df_deals_closes.merge(
                    df_deals_opens, on="position_id", suffixes=("_close", "_open")
                )

                # Process time columns
                merged_deals["time_open"] = pd.to_datetime(merged_deals["time_open"], unit='s')
                if "time_close" in merged_deals.columns:
                    merged_deals["time_close"] = pd.to_datetime(merged_deals["time_close"], unit='s')

                merged_deals = save_and_retrieve_historical_deals(merged_deals)

                # Store data
                stored_data['history_orders'] = history_orders
                stored_data['history_deals'] = history_deals
                stored_data['merged_deals'] = merged_deals
                stored_data['account_size'] = account_size

                return {'data_available': True}, dbc.Alert(message, color="success")

        return None, dbc.Alert("No valid trades found", color="warning")

    return None, ""


@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab'),
     Input('data-store', 'data')]
)
def render_tab_content(active_tab, data):
    if data is None or not data.get('data_available', False):
        return html.Div("Please fetch data first", className="text-center mt-5")

    merged_deals = stored_data['merged_deals']
    account_size = stored_data['account_size']

    if active_tab == "overview":
        return render_overview_tab(merged_deals, account_size)
    elif active_tab == "detailed":
        return render_detailed_tab(merged_deals, account_size)
    elif active_tab == "pnl":
        return render_pnl_tab(merged_deals, account_size)
    elif active_tab == "trades":
        return render_trades_tab(merged_deals)
    elif active_tab == "raw":
        return render_raw_tab(merged_deals, stored_data['history_orders'])

    return html.Div()


def render_overview_tab(merged_deals, account_size):
    # Calculate strategy metrics
    strategy_metrics_df = merged_deals[
        ["profit_open", "profit_close", "comment_open", "symbol_open", "time_open"]].copy()

    # Group by strategy and symbol
    grouped_metrics = strategy_metrics_df.groupby(["comment_open", "symbol_open"]).apply(
        lambda x: strategy_metrics(x, account_size)
    )

    # Account info
    account_info = mt5.account_info()
    if account_info:
        account_info = account_info._asdict()
    else:
        account_info = {'balance': 0, 'equity': 0, 'margin': 0}

    # Overall daily performance
    overall_daily = merged_deals.copy()
    overall_daily['date'] = overall_daily['time_close'].dt.date if 'time_close' in overall_daily.columns else \
    overall_daily['time_open'].dt.date
    overall_daily["profit"] = overall_daily["profit_open"] + overall_daily["profit_close"]
    daily_profit = overall_daily.groupby('date')['profit'].sum().reset_index()

    # Create daily performance figure
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
        xaxis=dict(tickformat='%Y-%m-%d')
    )

    # Prepare data for plotting
    plot_metrics = ["Total Profit", "Average Profit by Trade", "Win Rate (%)", "Loss Rate (%)"]
    plot_data = grouped_metrics.reset_index()
    plot_data['strategy_symbol'] = plot_data['comment_open'] + ' - ' + plot_data['symbol_open']

    # Create comparison figure
    comparison_fig = go.Figure()
    for metric in plot_metrics:
        comparison_fig.add_trace(go.Bar(
            x=plot_data['strategy_symbol'],
            y=plot_data[metric],
            name=metric
        ))

    comparison_fig.update_layout(
        title="Strategy Performance Comparison",
        xaxis_title="Strategy - Symbol",
        yaxis_title="Value",
        barmode='group'
    )

    return html.Div([
        html.H3("📈 Account Overview Pelo"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Balance"),
                        html.H4(f"${account_info['balance']:.2f}")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Equity"),
                        html.H4(f"${account_info['equity']:.2f}")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Margin"),
                        html.H4(f"${account_info['margin']:.2f}")
                    ])
                ])
            ], width=4),
        ], className="mb-4"),

        dcc.Graph(figure=daily_fig),

        html.H3("Strategy Performance Metrics Pelo", className="mt-4"),
        dash_table.DataTable(
            data=grouped_metrics.reset_index().to_dict('records'),
            columns=[{"name": i, "id": i, "type": "numeric", "format": {"specifier": ".2f"}}
                     if i not in ['comment_open', 'symbol_open'] else {"name": i, "id": i}
                     for i in grouped_metrics.reset_index().columns],
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            }
        ),

        dcc.Graph(figure=comparison_fig, className="mt-4")
    ])


def render_detailed_tab(merged_deals, account_size):
    # Calculate metrics
    strategy_metrics_df = merged_deals[
        ["profit_open", "profit_close", "comment_open", "symbol_open", "time_open"]].copy()

    grouped_metrics = strategy_metrics_df.groupby(["comment_open", "symbol_open"]).apply(
        lambda x: strategy_metrics(x, account_size)
    )

    plot_data = grouped_metrics.reset_index()
    plot_data['strategy_symbol'] = plot_data['comment_open'] + ' - ' + plot_data['symbol_open']

    # Create detailed metrics figure
    metrics_to_plot = ["Total Profit", "Win Rate (%)", "Sharpe Ratio"]

    detailed_fig = go.Figure()
    for metric in metrics_to_plot:
        if metric in plot_data.columns:
            detailed_fig.add_trace(go.Bar(
                x=plot_data['strategy_symbol'],
                y=plot_data[metric],
                name=metric
            ))

    detailed_fig.update_layout(
        title="Detailed Strategy Metrics",
        xaxis_title="Strategy - Symbol",
        yaxis_title="Value",
        barmode='group'
    )

    # Trade duration analysis
    duration_content = []
    if "time_close" in merged_deals.columns:
        merged_deals["duration"] = (merged_deals["time_close"] - merged_deals["time_open"]).dt.total_seconds() / 3600

        duration_fig = go.Figure()
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

        duration_content = [
            html.H4("Trade Duration Analysis", className="mt-4"),
            dcc.Graph(figure=duration_fig)
        ]

    # Hourly performance
    strategies = merged_deals['comment_open'].unique()
    default_strategy = strategies[0] if len(strategies) > 0 else None

    return html.Div([
        html.H3("Detailed Analysis Pelo"),

        html.Div([
            html.Label("Select metrics to display:"),
            dcc.Dropdown(
                id='metrics-dropdown',
                options=[
                    {'label': 'Total Profit', 'value': 'Total Profit'},
                    {'label': 'Win Rate (%)', 'value': 'Win Rate (%)'},
                    {'label': 'Sharpe Ratio', 'value': 'Sharpe Ratio'},
                    {'label': 'Max Drawdown (%)', 'value': 'Max Drawdown (%)'},
                    {'label': 'Profit Factor', 'value': 'Profit Factor'},
                ],
                value=["Total Profit", "Win Rate (%)", "Sharpe Ratio"],
                multi=True,
                className="mb-3"
            )
        ]),

        dcc.Graph(id='detailed-metrics-graph', figure=detailed_fig),

        *duration_content,

        html.H4("Hourly Average Performance", className="mt-4"),
        dcc.Dropdown(
            id='strategy-dropdown',
            options=[{'label': s, 'value': s} for s in strategies],
            value=default_strategy,
            className="mb-3"
        ),

        dcc.Graph(id='hourly-graph')
    ])


def render_pnl_tab(merged_deals, account_size):
    # Calculate PnL
    merged_deals["total_profit"] = merged_deals["profit_open"] + merged_deals["profit_close"]
    merged_deals_sorted = merged_deals.sort_values("time_open")
    merged_deals_sorted["cumulative_profit"] = merged_deals_sorted["total_profit"].cumsum()

    # Account equity
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
        yaxis_title="Equity ($)"
    )

    # Drawdown
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
        yaxis_title="Drawdown (%)"
    )

    # Profit by symbol
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
        yaxis_title="Cumulative Profit ($)"
    )

    # Trade scatter plot
    scatter_fig = go.Figure()

    for symbol in symbols:
        symbol_data = merged_deals_sorted[merged_deals_sorted["symbol_open"] == symbol]

        scatter_fig.add_trace(go.Scatter(
            x=symbol_data["time_open"],
            y=symbol_data["total_profit"],
            mode="markers",
            name=symbol,
            marker=dict(
                size=np.abs(symbol_data["total_profit"]) / max(1, np.abs(symbol_data["total_profit"]).max()) * 20 + 5
            ),
            text=symbol_data["comment_open"]
        ))

    scatter_fig.update_layout(
        title="Individual Trade Performance",
        xaxis_title="Date",
        yaxis_title="Profit/Loss ($)"
    )

    # Win/Loss Streaks
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

    win_streaks = streaks[streaks["win"]]
    loss_streaks = streaks[~streaks["win"]]

    streak_fig = go.Figure()

    if not win_streaks.empty:
        streak_fig.add_trace(go.Bar(
            x=win_streaks.index,
            y=win_streaks["streak_length"],
            name="Win Streaks",
            marker_color="green",
            text=win_streaks["total_profit"].round(2)
        ))

    if not loss_streaks.empty:
        streak_fig.add_trace(go.Bar(
            x=loss_streaks.index,
            y=loss_streaks["streak_length"],
            name="Loss Streaks",
            marker_color="red",
            text=loss_streaks["total_profit"].round(2)
        ))

    streak_fig.update_layout(
        title="Win/Loss Streak Lengths",
        xaxis_title="Streak ID",
        yaxis_title="Streak Length"
    )

    # Performance by strategy
    strategy_perf = merged_deals.groupby("comment_open")["total_profit"].agg(
        ["sum", "mean", "count", "std"]
    ).reset_index()
    strategy_perf.columns = ["Strategy", "Total Profit", "Average Profit", "Number of Trades", "Std Dev"]
    strategy_perf["Win Rate (%)"] = merged_deals.groupby("comment_open")["total_profit"].apply(
        lambda x: 100 * (x > 0).mean()
    ).values

    return html.Div([
        html.H3("PnL Performance Pelo"),

        dbc.Row([
            dbc.Col([dcc.Graph(figure=equity_fig)], width=4),
            dbc.Col([dcc.Graph(figure=drawdown_fig)], width=4),
            dbc.Col([dcc.Graph(figure=symbol_fig)], width=4),
        ], className="mb-4"),

        html.H4("Individual Trade Performance", className="mt-4"),
        dcc.Graph(figure=scatter_fig),

        html.H4("Win/Loss Streaks", className="mt-4"),
        dcc.Graph(figure=streak_fig),

        html.H4("Performance by Strategy", className="mt-4"),
        dash_table.DataTable(
            data=strategy_perf.to_dict('records'),
            columns=[{"name": i, "id": i, "type": "numeric", "format": {"specifier": ".2f"}}
                     if i != 'Strategy' else {"name": i, "id": i}
                     for i in strategy_perf.columns],
            style_cell={'textAlign': 'left'}
        )
    ])


def render_trades_tab(merged_deals):
    instances = list(merged_deals["comment_open"].unique())
    default_instance = instances[0] if instances else None

    return html.Div([
        html.H3("Trades table"),
        html.P("Voila les trades par bot pelo."),

        dcc.Dropdown(
            id='bot-dropdown',
            options=[{'label': i, 'value': i} for i in instances],
            value=default_instance,
            className="mb-3"
        ),

        html.Div(id='trades-table-container')
    ])


def render_raw_tab(merged_deals, history_orders):
    # Convert orders to DataFrame
    orders_df = pd.DataFrame(list(history_orders),
                             columns=history_orders[0]._asdict().keys()) if history_orders else pd.DataFrame()

    # Prepare download links
    merged_csv = merged_deals.to_csv(index=False)
    merged_b64 = base64.b64encode(merged_csv.encode()).decode()

    orders_csv = orders_df.to_csv(index=False) if not orders_df.empty else ""
    orders_b64 = base64.b64encode(orders_csv.encode()).decode()

    return html.Div([
        html.H3("Raw Data"),

        html.H4("Merged Deals Data"),
        dash_table.DataTable(
            data=merged_deals.head(100).to_dict('records'),  # Limit to 100 rows for performance
            columns=[{"name": i, "id": i} for i in merged_deals.columns],
            style_cell={'textAlign': 'left'},
            page_size=20,
            style_table={'overflowX': 'auto'}
        ),
        html.A(
            "Download Merged Deals CSV",
            href=f"data:text/csv;base64,{merged_b64}",
            download="merged_deals.csv",
            className="btn btn-primary mt-3"
        ),

        html.H4("Orders Data", className="mt-4"),
        dash_table.DataTable(
            data=orders_df.head(100).to_dict('records') if not orders_df.empty else [],
            columns=[{"name": i, "id": i} for i in orders_df.columns] if not orders_df.empty else [],
            style_cell={'textAlign': 'left'},
            page_size=20,
            style_table={'overflowX': 'auto'}
        ),
        html.A(
            "Download Orders CSV",
            href=f"data:text/csv;base64,{orders_b64}",
            download="orders.csv",
            className="btn btn-primary mt-3"
        ) if not orders_df.empty else html.Div()
    ])


# Additional callbacks for interactive components
@app.callback(
    Output('detailed-metrics-graph', 'figure'),
    [Input('metrics-dropdown', 'value')],
    [State('data-store', 'data')]
)
def update_detailed_metrics(selected_metrics, data):
    if not data or not selected_metrics:
        return go.Figure()

    merged_deals = stored_data['merged_deals']
    account_size = stored_data['account_size']

    strategy_metrics_df = merged_deals[
        ["profit_open", "profit_close", "comment_open", "symbol_open", "time_open"]].copy()

    grouped_metrics = strategy_metrics_df.groupby(["comment_open", "symbol_open"]).apply(
        lambda x: strategy_metrics(x, account_size)
    )

    plot_data = grouped_metrics.reset_index()
    plot_data['strategy_symbol'] = plot_data['comment_open'] + ' - ' + plot_data['symbol_open']

    fig = go.Figure()
    for metric in selected_metrics:
        if metric in plot_data.columns:
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

    return fig


@app.callback(
    Output('hourly-graph', 'figure'),
    [Input('strategy-dropdown', 'value')],
    [State('data-store', 'data')]
)
def update_hourly_graph(selected_strategy, data):
    if not data or not selected_strategy:
        return go.Figure()

    merged_deals = stored_data['merged_deals']
    account_size = stored_data['account_size']

    filtered_deals = merged_deals[merged_deals['comment_open'] == selected_strategy].copy()
    filtered_deals['hour'] = filtered_deals['time_open'].dt.hour
    filtered_deals = filtered_deals[
        ["profit_open", "profit_close", "comment_open", "symbol_open", "hour"]].copy()

    hourly_perf = filtered_deals.groupby(["hour"]).apply(
        lambda x: strategy_metrics(x, account_size)
    )

    fig = go.Figure()

    if not hourly_perf.empty:
        fig.add_trace(go.Bar(
            x=hourly_perf.index,
            y=hourly_perf["Average Profit by Trade"].fillna(0),
            name='Hourly Avg Profit'
        ))

    fig.update_layout(
        title=f"Hourly Performance for {selected_strategy}",
        xaxis_title="Hour of Day",
        yaxis_title="Average Profit"
    )

    return fig


@app.callback(
    Output('trades-table-container', 'children'),
    [Input('bot-dropdown', 'value')],
    [State('data-store', 'data')]
)
def update_trades_table(selected_bot, data):
    if not data or not selected_bot:
        return html.Div()

    merged_deals = stored_data['merged_deals']

    instance_deals = merged_deals[merged_deals["comment_open"] == selected_bot].copy()
    instance_deals["total_profit"] = instance_deals["profit_open"] + instance_deals["profit_close"]

    display_cols = ["symbol_open", "time_open", "time_close", "total_profit", "price_open", "price_close"]
    available_cols = [col for col in display_cols if col in instance_deals.columns]

    instance_deals = instance_deals[available_cols]

    # Rename columns for display
    column_mapping = {
        "symbol_open": "Symbol",
        "time_open": "Open Time",
        "time_close": "Close Time",
        "total_profit": "Total Profit",
        "price_open": "Open Price",
        "price_close": "Close Price"
    }

    instance_deals = instance_deals.rename(columns=column_mapping)

    return dash_table.DataTable(
        data=instance_deals.to_dict('records'),
        columns=[{"name": i, "id": i, "type": "numeric", "format": {"specifier": ".2f"}}
                 if i in ['Total Profit', 'Open Price', 'Close Price'] else {"name": i, "id": i}
                 for i in instance_deals.columns],
        style_cell={'textAlign': 'left'},
        page_size=20
    )


if __name__ == '__main__':
    app.run(debug=True, port=8050)