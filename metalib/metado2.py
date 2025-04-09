import logging
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import vectorbt as vbt

from metalib.indicators import *
from metalib.metastrategy import MetaStrategy

class MetaDO(MetaStrategy):
    def __init__(self,
                 symbols,
                 timeframe,
                 tag,
                 active_hours,
                 mode="mean_rev",
                 risk_factor=1,
                 lookahead=24,
                 ):
        super().__init__(symbols, timeframe, tag, active_hours)
        self.indicators         = None
        self.quantile           = None
        self.state              = None
        self.mode               = mode
        self.risk_factor        = risk_factor
        self.lookahead          = lookahead
        self.telegram           = True
        self.vol                = None
        self.logger             = logging.getLogger(__name__)
        logging.basicConfig(filename=f'../logs/{self.tag}.log', encoding='utf-8', level=logging.DEBUG)

    def signals(self):
        ohlc        = self.data[self.symbols[0]]
        close       = ohlc['close']

        self.vol        = close.pct_change().std()*np.sqrt(48)

        donchian = ohlc.resample('15min', label="right", closed="right").agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})

        upper_donchian = donchian.high.rolling(8).max()
        lower_donchian = donchian.low.rolling(8).min()
        channels = pd.concat({'upper': upper_donchian, 'lower': lower_donchian}, axis=1)

        channels.index = pd.to_datetime(channels.index)  # Ensure the index is datetime
        resampled_channels = channels.resample("min", label="right", closed="right").ffill()

        resampled_channels = resampled_channels.reindex(ohlc.index).ffill()
        crossed_above_upper = close.vbt.crossed_above(resampled_channels.upper)
        crossed_below_lower = close.vbt.crossed_below(resampled_channels.lower)

        if self.mode == "mean_rev":
            long_signal     = crossed_below_lower.iloc[-1]
            short_signal    = crossed_above_upper.iloc[-1]
        elif self.mode == "trend_follow":
            long_signal     = crossed_above_upper.iloc[-1]
            short_signal    = crossed_below_lower.iloc[-1]
        else:
            print("Invalid mode: %s" % self.mode)
            raise NotImplementedError

        if long_signal and not self.are_positions_with_tag_open(position_type="buy"):
            self.state = 1
        elif short_signal and not self.are_positions_with_tag_open(position_type="sell"):
            self.state = -1
        else:
            self.state = 0

        # === Logging Aligned with Donchian Logic ===
        print(f"{self.tag}::: Open positions for strategy: {self.tag}: {self.are_positions_with_tag_open()}")

        print(f"{self.tag}::: Donchian Signal Check:")
        print(f"    - Current Close: {close.iloc[-1]:.2f}")
        print(f"    - Resampled Upper Band: {resampled_channels.upper.iloc[-1]}")
        print(f"    - Resampled Lower Band: {resampled_channels.lower.iloc[-1]}")
        print(f"    - Crossed Above Upper (Short Signal): {crossed_above_upper.iloc[-1]}")
        print(f"    - Crossed Below Lower (Long Signal): {crossed_below_lower.iloc[-1]}")
        print(f"    => Final Long Signal: {long_signal}")
        print(f"    => Final Short Signal: {short_signal}")
        print(f"    => Final State: {self.state}")
        print(f"{self.tag}::: Saved Vol which is currently: {self.vol}%")

            
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
        digits = symbol_info.digits +1  # Use symbol_info.digits for proper rounding + add one because its after the decimal
        tick_size = symbol_info.point

        if symbol_info is None:
            print(f"Failed to get symbol info for {symbol}")
            return
        
        print(f"Rounding for {symbol}: {digits}")
        print(f"Vol Risk factor: {self.vol * self.risk_factor}")

        mean_entry_price, num_positions = self.get_positions_info()
        mean_entry_price = round(mean_entry_price, digits)

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
        print(f"{self.tag}::: No fitted content to save.")
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
        return max(round(self.risk_factor * 5 * lots, 2), symbol_info.volume_min)


    def retrieve_indicators(self, ohlc_df):
        print(f"{self.tag}::: No indicators to compute indicators")
        return