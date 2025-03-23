from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
import pytz as pytz
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

from metalib.indicators import *
from metalib.metastrategy import MetaStrategy

class MetaGO(MetaStrategy):
    def __init__(self,
                 symbols,
                 timeframe,
                 tag,
                 active_hours,
                 risk_factor=1,
                 lookahead=24,
                 hist_length=10000,
                 ):
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
        self.vol                = None
        self.logger             = logging.getLogger(__name__)
        logging.basicConfig(filename=f'{self.tag}.log', encoding='utf-8', level=logging.DEBUG)

    def signals(self):
        ohlc        = self.data[self.symbols[0]]
        indicators  = self.retrieve_indicators(ohlc)
        close       = ohlc['close']

        self.vol        = close.pct_change().std()*np.sqrt(48)
        self.indicators = indicators

        uptrend     = indicators['uptrend'].iloc[-1]
        downtrend   = indicators['downtrend'].iloc[-1]
        true_open_yearly = indicators['true_open_monthly'].iloc[-1]

        mask_uptrend_below_yearly   = (uptrend > 0) & (close < true_open_yearly)
        mask_downtrend_above_yearly = (downtrend > 0) & (close > true_open_yearly)

        close_last_4 = close.tail(3).head(2)  # Extract last 3 values, then take first 2 candles
        close_positive_condition = np.all(close_last_4 > 0)
        close_negative_condition = np.all(close_last_4 < 0)

        long_signal = close_positive_condition & mask_uptrend_below_yearly.iloc[-1]
        short_signal = close_negative_condition & mask_downtrend_above_yearly.iloc[-1]

        if long_signal and not self.are_positions_with_tag_open(position_type="buy"):
            self.state = 1
        elif short_signal and not self.are_positions_with_tag_open(position_type="sell"):
            self.state = -1
        else:
            self.state = 0

        # Detailed Logging
        print(f"{self.tag}::: Open positions for strategy: {self.tag}: {self.are_positions_with_tag_open()}")
        print(f"{self.tag}::: Long Signal Components:")
        print(f"    - All of last 4 (except last) are > 0: {close_positive_condition}")
        print(f"    - Uptrend > 0: {uptrend > 0}")
        print(f"    - Close < True Open Yearly: {close.iloc[-1] < true_open_yearly}")
        print(f"    - Mask Uptrend Below Yearly: {mask_uptrend_below_yearly.iloc[-1]}")
        print(f"    => Final Long Signal: {long_signal}")

        print(f"{self.tag}::: Short Signal Components:")
        print(f"    - All of last 4 (except last) are < 0: {close_negative_condition}")
        print(f"    - Downtrend > 0: {downtrend > 0}")
        print(f"    - Close > True Open Yearly: {close.iloc[-1] > true_open_yearly}")
        print(f"    - Mask Downtrend Above Yearly: {mask_downtrend_above_yearly.iloc[-1]}")
        print(f"    => Final Short Signal: {short_signal}")

        print(f"{self.tag}::: Saved Vol which is currently: {self.vol}%")

        signal_line = indicators.iloc[[-1]]
        self.signals_data = signal_line
        
        # Generate and save the chart
        self.plot_signals(indicators.iloc[-24*5:], long_signal, short_signal)
        
        # Ensure indicators is a DataFrame
        if isinstance(indicators, pd.DataFrame) and not indicators.empty:
            file_path = f"../indicators/indicators_{self.symbols[0]}.xlsx"
            try:
                indicators.to_excel(file_path, engine="openpyxl")
                print(f"Indicators saved successfully at {file_path}")
            except Exception as e:
                print(f"Error saving file: {e}")
        else:
            print("Indicators DataFrame is empty or invalid!")
            
    def plot_signals(self, indicators, long_signal, short_signal):
        plt.figure(figsize=(12, 6))
        
        plt.plot(indicators.index, indicators['close'], label='Close Price', color='black', linewidth=1.5)
        plt.plot(indicators.index, indicators['true_open_monthly'], label='True Open Monthly', linestyle='dashed', color='blue')
        plt.plot(indicators.index, indicators['true_open_weekly'], label='True Open Weekly', linestyle='dashed', color='purple')
        plt.plot(indicators.index, indicators['long_sma'], label='Long SMA', linestyle='solid', color='green')
        plt.plot(indicators.index, indicators['short_sma'], label='Short SMA', linestyle='solid', color='green')

        # Mark long and short signals
        if long_signal:
            plt.scatter(indicators.index[-1], indicators['close'].iloc[-1], color='green', s=100, label='Long Signal', edgecolors='black')
        if short_signal:
            plt.scatter(indicators.index[-1], indicators['close'].iloc[-1], color='red', s=100, label='Short Signal', edgecolors='black')

        plt.title('Market Context and Trading Signals')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.savefig(f"../charts/trading_signals_{self.symbols[0]}.png")
        plt.close()
        print("Trading signals chart saved as 'trading_signals.png'")
        
    def check_conditions(self):
        if not self.vol:
            print(f"Warning: self.vol is None or 0, skipping trade execution.")
            return

        volume = self.position_sizing(
            percentage=self.risk_factor, 
            symbol=self.symbols[0]
        )
        
        mean_entry_price, num_positions = self.get_positions_info()
        mean_entry_price = round(mean_entry_price, 4)
        
        symbol = self.symbols[0]
        symbol_info = mt5.symbol_info(symbol)
        
        if symbol_info is None:
            print(f"Failed to get symbol info for {symbol}")
            return
        
        tick_size = symbol_info.point
        digits = symbol_info.digits  # Use symbol_info.digits for proper rounding

        # Take-Profit and Stop-Loss calculations (keeping your SL formula)
        tp = mean_entry_price * (1 + self.state * self.vol * self.risk_factor)
        sl = mean_entry_price * (1 - self.state * self.vol)

        # Proper rounding
        tp, sl = round(tp, digits), round(sl, digits)

        print(f"Mean Entry Price: {mean_entry_price}, Positions: {num_positions}, Vol: {self.vol}%, State: {self.state}, TP: {tp}, SL: {sl}")

        if self.state in [1, -1]:  # If a trade signal is active
            try:
                self.execute(symbol=symbol, volume=volume, short=(self.state == -1), sl=sl, tp=tp)
                trade_type = "BUY" if self.state == 1 else "SELL"
                self.send_telegram_message(
                    f"Entered {trade_type} order for {symbol} with volume: {volume}. Mean Entry Price: {mean_entry_price}, Positions: {num_positions}."
                )
            except Exception as e:
                print(f"Execution failed for {symbol}: {str(e)}")

        elif self.state == -2:
            if num_positions > 0:  # Avoid unnecessary API calls
                self.close_all_positions()
                self.send_telegram_message(f"Closed all positions for {symbol}")
            else:
                print(f"No positions to close for {symbol}")

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
        # plt.savefig("decision_tree.png")
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

        # Ensure it meets the broker's minimum lot size requirement
        return max(round(self.risk_factor * 5 * lots, 2), symbol_info.volume_min)


    def retrieve_indicators(self, ohlc_df):
        ohlc = ohlc_df.copy()
        ohlc_daily = ohlc.resample('1D').agg({'open': 'first',
                                              'high': 'max',
                                              'low': 'min',
                                              'close': 'last',
                                              })
        closes  = ohlc.loc[:, 'close']
        returns = closes.apply(np.log).diff().dropna()

        short_sma   = closes.rolling(12).mean()
        long_sma    = closes.rolling(24).mean()

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
                        short_sma, long_sma,
                        closes
                        ]

        indicators          = pd.concat(indicators, axis=1).iloc[1:]
        indicators.columns  = [ 'uptrend', 'downtrend', 'sideways',
                                'true_open_weekly', 'true_open_monthly',
                                'crossed_diff_monthly_weekly',
                                'above_true_open_monthly_diff', 'above_true_open_weekly_diff',
                                'short_sma', 'long_sma',
                                'close' ]
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
            tick_info = mt5.symbol_info_tick(self.symbols[0])
            mean_entry_price = tick_info.last if tick_info.last else (tick_info.bid + tick_info.ask) / 2
            num_positions       = 0

        # Return the mean entry price and number of positions
        return mean_entry_price, num_positions


def assign_cat(val):
    if val < 0.:
        return 0
    else:
        return 1
