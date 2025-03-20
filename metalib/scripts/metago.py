from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
import pytz as pytz
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

from metalib.indicators import *
from metalib.metastrategy import MetaStrategy


class MetaGO(MetaStrategy):
    def __init__(self, symbols, timeframe, tag, active_hours, risk_factor=1, lookahead=24, hist_length=10000,):
        super().__init__(symbols, timeframe, tag, active_hours)
        self.indicators         = None
        self.quantile           = None
        self.model              = None
        self.indicators_std     = None
        self.indicators_mean    = None
        self.state              = None
        self.risk_factor        = risk_factor
        self.lookahead          = lookahead
        self.hist_length        = hist_length
        self.telegram           = True
        self.logger             = logging.getLogger(__name__)
        logging.basicConfig(filename=f'{self.tag}.log', encoding='utf-8', level=logging.DEBUG)

    def signals(self):
        ohlc        = self.data[self.symbols[0]]
        indicators  = self.retrieve_indicators(ohlc)
        close       = ohlc['close']

        self.indicators = indicators

        # Demean Indicators
        indicators = indicators.tail(3)
        close      = close.tail(3)
        indicators = (indicators - self.indicators_mean) / self.indicators_std

        uptrend     = indicators['uptrend']
        downtrend   = indicators['downtrend']
        true_open_yearly = indicators['true_open_monthly']

        mask_uptrend_below_yearly   = ( uptrend > 0 ) & (close < true_open_yearly)
        mask_downtrend_above_yearly = ( downtrend > 0 ) & (close > true_open_yearly)
        mask_combined               = mask_uptrend_below_yearly | mask_downtrend_above_yearly

        y_hat = self.model.predict_proba(indicators)[:, 1]
        vote = mask_combined.iloc[-1]

        if y_hat[-1] < 0.3 and self.are_positions_with_tag_open(position_type="buy"):
            self.state = -2
        elif y_hat[-1] > 0.7 and self.are_positions_with_tag_open(position_type="sell"):
            self.state = -2
        elif vote and y_hat[-1] > 0.9:
            self.state = 1
        elif vote and y_hat[-1] < 0.1:
            self.state = -1
        else:
            self.state = 0

        print(f"{self.tag}::: Open positions for strategy: {self.tag}: {self.are_positions_with_tag_open()}")
        print(f"{self.tag}::: Vote of indicators: {vote}, and last 3 predicted probabilities: {y_hat}")

        signal_line = indicators.iloc[[-1]]
        signal_line.loc[:, 'vote'] = vote
        signal_line.loc[:, 'predicted_proba'] = y_hat[-1]

        self.signals_data = signal_line

    def check_conditions(self):
        volume = self.position_sizing(
            percentage  = 0.05,
            symbol      = self.symbols[0]
        )
        mean_entry_price, num_positions = self.get_positions_info()
        mean_entry_price                = round(mean_entry_price, 4)
        if self.state == 0:
            pass
        elif self.state == 1:
            self.execute(symbol=self.symbols[0], volume=volume, short=False)
            # Send a message when an order is entered
            self.send_telegram_message(
                f"Entered BUY order for {self.symbols[0]} with volume: {volume} et pelo sa achete! Mean Entry Price: {mean_entry_price}, Number of Positions: {num_positions}")
        elif self.state == -1:
            self.execute(symbol=self.symbols[0], volume=volume, short=True)
            # Send a message when an order is entered
            self.send_telegram_message(
                f"Entered SELL order for {self.symbols[0]} with volume: {volume}et pelo ca vend: Mean Entry Price: {mean_entry_price}, Number of Positions: {num_positions}")
        elif self.state == -2:
            self.close_all_positions()
            # Send a message when positions are closed
            self.send_telegram_message(f"Closed all positions for {self.symbols[0]}")

    def fit(self, data=None):
        # Define the UTC timezone
        utc = pytz.timezone('UTC')
        # Get the current time in UTC
        end_time    = datetime.now(utc)
        start_time  = end_time - timedelta(days=60)
        # Set the time components to 0 (midnight) and maintain the timezone
        end_time    = end_time.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(utc)
        start_time  = start_time.astimezone(utc)
        # Grab lookahead parameter
        lookahead   = self.lookahead
        hist_length = self.hist_length

        # Pulling last days of data
        if data is None:
            self.loadData(start_time, end_time)
            data = self.data[self.symbols[0]]
        else:
            data = data[self.symbols[0]]

        returns     = data.loc[:, 'close'].apply(np.log) - data.loc[:, 'open'].apply(np.log)

        # Compute rolling next returns series
        T = returns.shape[0]
        next_returns    = [np.sum(returns[i + 1: i + lookahead]) for i in range(T)]
        next_returns    = pd.Series(next_returns, index=returns.index)

        # Indicators
        indicators      = self.retrieve_indicators(ohlc_df=data).dropna()
        self.indicators = indicators
        close           = data['close']

        # Retrieve history
        hist_indicators     = indicators[:hist_length]
        hist_next_returns   = next_returns.loc[hist_indicators.index]
        # indicators          = indicators.loc[indicators.index.difference(hist_indicators.index)]
        # next_returns        = next_returns.loc[indicators.index]

        # Demean from history
        indicators     = (hist_indicators - hist_indicators.mean()) / hist_indicators.std()
        next_returns   = (hist_next_returns - hist_next_returns.mean()) / hist_next_returns.std()

        print(f"Before filtering: {indicators.shape[0]}")
        # Mask for uptrend and close below true yearly open
        uptrend = indicators['uptrend']
        downtrend = indicators['downtrend']
        true_open_monthly = indicators['true_open_monthly']

        # Ensure all series are aligned to the same index
        aligned_close, aligned_true_open_monthly = close.align(true_open_monthly, join='inner')
        aligned_uptrend, aligned_close = uptrend.align(aligned_close, join='inner')
        align_downtrend, aligned_close = downtrend.align(aligned_close, join='inner')

        # Perform the comparison with aligned indices
        mask_uptrend_below_yearly = ( aligned_uptrend > 0 ) & (aligned_close < aligned_true_open_monthly)
        mask_downtrend_above_yearly = ( align_downtrend > 0 ) & (aligned_close > aligned_true_open_monthly)

        # Combine masks with all other times as False
        mask_combined = mask_uptrend_below_yearly | mask_downtrend_above_yearly

        indicators = indicators[mask_combined]
        print(f"After filtering: {indicators.shape[0]}")
        dummy_next_five_returns = next_returns.loc[indicators.index].apply(assign_cat)

        X, y = indicators.ffill(), dummy_next_five_returns

        # Fit DecisionTreeClassifier
        tree_dummy = DecisionTreeClassifier(max_depth=5).fit(X, y)
        print(f"{self.tag}::: DecisionTree Model trained from {X.index[0]} to {X.index[-1]}.")

        # Add line to log file to signal training completion
        self.logger.info(f"DecisionTree Model trained from {X.index[0]} to {X.index[-1]}.")

        # Save model, 1st and 2nd indicator moments
        self.model = tree_dummy
        self.indicators_mean = hist_indicators.mean()
        self.indicators_std = hist_indicators.std()

        # Save the 90% quantile of the closes
        self.quantile = data['close'].quantile(0.9)

        # Plot and save the decision tree
        plt.figure(figsize=(20, 10))
        plot_tree(tree_dummy, feature_names=X.columns, filled=True, rounded=True)
        ## plt.savefig("decision_tree.png")
        plt.close()

        print(f"{self.tag}::: DecisionTree Model and first/second moments saved.")


    def position_sizing(self, percentage, symbol, account_balance=None):
        # Retrieve account balance if not provided
        if account_balance is None:
            account_info = mt5.account_info()
            if account_info is None:
                print("Failed to get account balance, error code =", mt5.last_error())
                mt5.shutdown()
                return
            account_balance = account_info.balance

        # Calculate the position size
        position_size = abs(account_balance * percentage)

        # Get the symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Failed to get symbol info for {symbol}, error code =", mt5.last_error())
            mt5.shutdown()
            return

        # Calculate the number of lots
        contract_size   = symbol_info.trade_contract_size  # Use trade_contract_size instead of lot_size
        price           = mt5.symbol_info_tick(symbol).ask
        lots            = position_size / (contract_size * price)

        return round(self.risk_factor * 5 * lots, 2)

    def retrieve_indicators(self, ohlc_df):
        ohlc = ohlc_df.copy()
        ohlc_daily = ohlc.resample('1D').agg({'open': 'first',
                                              'high': 'max',
                                              'low': 'min',
                                              'close': 'last',
                                              })
        closes  = ohlc.loc[:, 'close']
        returns = closes.apply(np.log).diff().dropna()

        short_sma   = closes.rolling(12).median()
        long_sma    = closes.rolling(24).median()

        uptrend     = (closes > short_sma) & (short_sma > long_sma)
        downtrend   = (closes < short_sma) & (short_sma < long_sma)
        sideways    = ~(uptrend | downtrend)
        print(f"{self.tag}::: Computed Trend indicators")

        true_open_weekly    = get_last_monday_6pm_open_ffill(ohlc, ohlc.index)
        true_open_monthly   = get_second_monday_open_ffill(ohlc_daily, ohlc.index)
        true_open_yearly    = get_first_monday_of_april_open_ffill(ohlc_daily, ohlc.index)
        print(f"{self.tag}::: Computed True opens (weekly, monthly) ")

        # Compute differences between various true open prices
        crossed_diff_monthly_weekly     = true_open_monthly - true_open_weekly
        above_true_open_monthly_diff    = ohlc['close'] - true_open_monthly
        above_true_open_weekly_diff     = ohlc['close'] - true_open_weekly
        print(f"{self.tag}::: Computed True opens diffs (weekly, monthly) ")

        # Merge Features
        indicators =  [ uptrend, downtrend, sideways,
                        true_open_weekly, true_open_monthly,
                        crossed_diff_monthly_weekly,
                        above_true_open_monthly_diff, above_true_open_weekly_diff,
                        closes
                        ]

        indicators          = pd.concat(indicators, axis=1).iloc[1:]
        indicators.columns  = [ 'uptrend', 'downtrend', 'sideways',
                                'true_open_weekly', 'true_open_monthly',
                                'crossed_diff_monthly_weekly',
                                'above_true_open_monthly_diff', 'above_true_open_weekly_diff',
                                'close']
        indicators.index    = pd.to_datetime(indicators.index)
        indicators.columns  = indicators.columns.astype(str)
        print(f"{self.tag}::: Merged indicators")

        return indicators

    def get_positions_info(self):
        # Ensure connected to MT5
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            return None, None

        # Retrieve all positions
        positions = mt5.positions_get()
        if positions is None:
            print("No positions found, error code =", mt5.last_error())
            return None, None

        # Filter positions based on the comment
        filtered_positions = [pos for pos in positions if pos.comment == self.tag]

        # Calculate mean entry price and count positions
        if filtered_positions:
            total_volume        = sum(pos.volume for pos in filtered_positions)
            mean_entry_price    = sum(pos.price_open * pos.volume for pos in filtered_positions) / total_volume
            num_positions       = len(filtered_positions)
        else:
            mean_entry_price    = 0
            num_positions       = 0

        # Return the mean entry price and number of positions
        return mean_entry_price, num_positions


def assign_cat(val):
    if val < 0.:
        return 0
    else:
        return 1
