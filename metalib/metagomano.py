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

    def signals(self):
        ohlc = self.data[self.symbols[0]]
        indicators = self.retrieve_indicators(ohlc)
        close = ohlc['close']
        del ohlc

        self.vol = close.pct_change().std() * np.sqrt(48)
        self.indicators = indicators

        uptrend = indicators['uptrend'].iloc[-1]
        downtrend = indicators['downtrend'].iloc[-1]
        true_open_monthly = indicators['true_open_monthly'].iloc[-1]

        mask_uptrend_below_yearly = (uptrend > 0) & (close < true_open_monthly)
        mask_downtrend_above_yearly = (downtrend > 0) & (close > true_open_monthly)

        close_last_4 = close.tail(3).head(2)  # Extract last 3 values, then take first 2 candles
        close_positive_condition = np.all(close_last_4 > 0)
        close_negative_condition = np.all(close_last_4 < 0)

        long_signal = close_positive_condition & mask_uptrend_below_yearly.iloc[-1]
        short_signal = close_negative_condition & mask_downtrend_above_yearly.iloc[-1]

        sl = close.iloc[-1] - 6 * indicators['atr'].iloc[-1] if short_signal else close.iloc[-1] + 6 * \
                                                                                       indicators['atr'].iloc[-1]
        self.sl = float(sl)
        self.tp = float(true_open_monthly)

        if long_signal and not self.are_positions_with_tag_open(position_type="buy"):
            self.state = 1
        elif short_signal and not self.are_positions_with_tag_open(position_type="sell"):
            self.state = -1
        elif downtrend and self.are_positions_with_tag_open(position_type="buy"):
            self.state = -2
        elif uptrend and self.are_positions_with_tag_open(position_type="sell"):
            self.state = -2
        else:
            self.state = 0

        # Detailed Logging
        print(f"{self.tag}::: Open positions for strategy: {self.tag}: {self.are_positions_with_tag_open()}")
        print(f"{self.tag}::: Long Signal Components:")
        print(f"    - All of last 4 (except last) are > 0: {close_positive_condition}")
        print(f"    - Uptrend > 0: {uptrend > 0}")
        print(f"    - Close < True Open Monthly: {close.iloc[-1] < true_open_monthly}")
        print(f"    - Mask Uptrend Below Monthly: {mask_uptrend_below_yearly.iloc[-1]}")
        print(f"    => Final Long Signal: {long_signal}")

        print(f"{self.tag}::: Short Signal Components:")
        print(f"    - All of last 4 (except last) are < 0: {close_negative_condition}")
        print(f"    - Downtrend > 0: {downtrend > 0}")
        print(f"    - Close > True Open Monthly: {close.iloc[-1] > true_open_monthly}")
        print(f"    - Mask Downtrend Above Yearly: {mask_downtrend_above_yearly.iloc[-1]}")
        print(f"    => Final Short Signal: {short_signal}")

        print(f"{self.tag}::: Saved Vol which is currently: {self.vol}%")

        signal_line = indicators.reset_index(drop=True).iloc[[-1]]
        signal_line["timestamp"] = indicators.index[-1]
        self.signalData = signal_line


            
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

        symbol = self.symbols[0]
        symbol_info = mt5.symbol_info(symbol)
        digits = symbol_info.digits + 1  # Use symbol_info.digits for proper rounding + add one because its after the decimal
        tick_size = symbol_info.point

        if symbol_info is None:
            print(f"Failed to get symbol info for {symbol}")
            return

        print(f"Rounding for {symbol}: {digits}")
        print(f"Strategy Risk factor: {self.risk_factor}")
        print(f"Vol times Risk factor: {self.vol * self.risk_factor}")

        positions_mean_entry_price, num_positions = self.get_positions_info()

        # Take-Profit and Stop-Loss calculations (keeping your SL formula)
        price_mid = (symbol_info.ask + symbol_info.bid) / 2

        # Proper rounding
        tp, sl = round(self.tp, digits), round(self.sl, digits)

        print(
            f"Mid Price: {price_mid}, Positions: {num_positions}, Vol: {self.vol}%, State: {self.state}, TP: {tp}, SL: {sl}")

        if self.state in [1, -1]:  # If a trade signal is active
            try:
                self.execute(symbol=symbol, volume=volume, short=(self.state == -1), sl=sl, tp=tp)
                trade_type = "BUY" if self.state == 1 else "SELL"
                self.send_telegram_message(
                    f"Entered {trade_type} order for {symbol} with volume: {volume}. Mean Entry Price: {positions_mean_entry_price}, Positions: {num_positions}."
                )
            except Exception as e:
                print(f"Execution failed for {symbol}: {str(e)}")

        elif self.state == -2: # If we have to close positions.
            if num_positions > 0:
                self.close_all_positions()
                self.send_telegram_message(f"Closed all positions for {symbol}")
            else:
                print(f"No positions to close for {symbol}")

    def fit(self, data=None):
        print(f"No model to fit for MetaGO(mano) pelo!")
        return

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
     #    return max(round(self.risk_factor * 5 * lots, 2), symbol_info.volume_min)
        return symbol_info.volume_min

    def retrieve_indicators(self, ohlc_df):
        ohlc = ohlc_df.copy()
        ohlc_daily = ohlc.resample('1D').agg({'open': 'first',
                                              'high': 'max',
                                              'low': 'min',
                                              'close': 'last',
                                              }, closed="right", label="right")
        closes = ohlc.loc[:, 'close']
        short_sma = closes.rolling(12).mean()
        long_sma = closes.rolling(24).mean()

        uptrend = (closes > short_sma) & (short_sma > long_sma)
        downtrend = (closes < short_sma) & (short_sma < long_sma)
        sideways = ~(uptrend | downtrend)  #
        print(f"{self.tag}::: Computed Trend indicators")

        true_open_weekly = get_last_monday_6pm_open_ffill(ohlc, ohlc.index)
        true_open_monthly = get_second_monday_open_ffill(ohlc_daily, ohlc.index)
        true_open_yearly = get_first_monday_of_april_open_ffill(ohlc_daily, ohlc.index)
        print(f"{self.tag}::: Computed True opens (weekly, monthly) ")
        del ohlc
        del ohlc_daily

        # Compute differences between various true open prices
        crossed_diff_monthly_weekly = true_open_monthly - true_open_weekly
        above_true_open_monthly_diff = closes - true_open_monthly
        above_true_open_weekly_diff = closes - true_open_weekly
        print(f"{self.tag}::: Computed True opens diffs (weekly, monthly) ")

        # Compute ATR
        highs = ohlc_df['high']
        lows = ohlc_df['low']
        prev_closes = ohlc_df['close'].shift(1)
        tr = pd.concat([
            highs - lows,
            (highs - prev_closes).abs(),
            (lows - prev_closes).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        print(f"{self.tag}::: Computed ATR")

        # Merge Features
        indicators = [uptrend, downtrend, sideways,
                      true_open_weekly, true_open_monthly,
                      crossed_diff_monthly_weekly,
                      above_true_open_monthly_diff, above_true_open_weekly_diff,
                      short_sma, long_sma,
                      closes, atr
                      ]

        indicators = pd.concat(indicators, axis=1).iloc[1:]
        indicators.columns = ['uptrend', 'downtrend', 'sideways',
                              'true_open_weekly', 'true_open_monthly',
                              'crossed_diff_monthly_weekly',
                              'above_true_open_monthly_diff', 'above_true_open_weekly_diff',
                              'short_sma', 'long_sma',
                              'close', 'atr']

        indicators.index = pd.to_datetime(indicators.index)
        indicators.columns = indicators.columns.astype(str)
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
            total_volume    = sum(pos.volume for pos in filtered_positions)
            mid_price       = sum(pos.price_open * pos.volume for pos in filtered_positions) / total_volume
            num_positions   = len(filtered_positions)
        else:
            tick_info       = mt5.symbol_info_tick(self.symbols[0])
            mid_price       = tick_info.last if tick_info.last else (tick_info.bid + tick_info.ask) / 2
            num_positions   = 0

        # Return the mean entry price and number of positions
        return mid_price, num_positions


def assign_cat(val):
    if val < 0.:
        return 0
    else:
        return 1
