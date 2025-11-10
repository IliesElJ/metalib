from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from metalib.metastrategy import MetaStrategy


class MetaOB(MetaStrategy):
    def __init__(self,
                 symbols,
                 timeframe,
                 tag,
                 size_position,
                 active_hours=None,
                     pivot_window=40,
                 breakout_lookback=3,
                 sma_short_hours=192,  # 8 days
                 sma_long_hours=1200,  # 50 days
                 atr_period=14,
                 sl_atr_mult=2.0,
                 tp_atr_mult=6.0,
                 ):
        super().__init__(symbols, timeframe, tag, size_position, active_hours, size_position)

        # Strategy parameters
        self.pivot_window = pivot_window
        self.breakout_lookback = breakout_lookback
        self.sma_short_hours = sma_short_hours
        self.sma_long_hours = sma_long_hours
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult

        # State variables
        self.state = 0
        self.sl = None
        self.tp = None
        self.signalData = None
        self.telegram = True
        self.logger = logging.getLogger(__name__)

    def signals(self):
        """Generate trading signals based on order blocks and pivot breakouts"""
        ohlc = self.data[self.symbols[0]]

        # Calculate indicators
        indicators = self.calculate_indicators(ohlc)

        # Detect signals
        long_signal = self.detect_long_signal(ohlc, indicators)
        short_signal = self.detect_short_signal(ohlc, indicators)

        # Calculate stops
        if long_signal or short_signal:
            self.calculate_stops(ohlc, indicators, long_signal)

        # Set state
        if long_signal and not self.are_positions_with_tag_open(position_type="buy"):
            self.state = 1
        elif short_signal and not self.are_positions_with_tag_open(position_type="sell"):
            self.state = -1
        else:
            self.state = 0

        # Logging
        self.log_signals(ohlc, indicators, long_signal, short_signal)

        # Save signal data
        signal_line = indicators.iloc[-1].copy()
        signal_line["timestamp"] = ohlc.index[-1]
        signal_line["long_signal"] = long_signal
        signal_line["short_signal"] = short_signal
        self.signalData = signal_line

    def calculate_indicators(self, ohlc):
        """Calculate all technical indicators"""
        indicators = pd.DataFrame(index=ohlc.index)

        # Price data
        o, h, l, c = ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close']
        indicators['close'] = c

        # SMAs
        indicators['sma_short'] = c.rolling(self.sma_short_hours).mean()
        indicators['sma_long'] = c.rolling(self.sma_long_hours).mean()
        indicators['uptrend'] = indicators['sma_short'] > indicators['sma_long']

        # Pivot points
        indicators['pivot_low'] = l.rolling(self.pivot_window).min().shift(1)
        indicators['pivot_high'] = h.rolling(self.pivot_window).max().shift(1)

        # ATR
        prev_c = c.shift(1)
        tr = pd.concat([
            h - l,
            (h - prev_c).abs(),
            (l - prev_c).abs()
        ], axis=1).max(axis=1)
        indicators['atr'] = tr.rolling(window=self.atr_period).mean()

        # Order blocks
        o_1, h_1, l_1, c_1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
        indicators['bull_ob'] = (c > o) & (o_1 > c_1) & (h > h_1)
        indicators['bear_ob'] = (c < o) & (o_1 < c_1) & (l < l_1)

        # Pivot crosses
        indicators['cross_below_pivot'] = l < indicators['pivot_low']
        indicators['cross_above_pivot'] = h > indicators['pivot_high']

        return indicators

    def detect_long_signal(self, ohlc, indicators):
        """Detect long entry signal"""
        # Check if pivot low was crossed recently
        cross_pivot = indicators['cross_below_pivot'].rolling(
            self.breakout_lookback
        ).apply(lambda x: np.any(x), raw=True).astype(bool)

        # Order block pattern
        bull_ob = indicators['bull_ob'].iloc[-1]

        # Trend filter
        uptrend = indicators['uptrend'].iloc[-1]

        # Combine conditions
        return cross_pivot.iloc[-1] and bull_ob and uptrend

    def detect_short_signal(self, ohlc, indicators):
        """Detect short entry signal"""
        # Check if pivot high was crossed recently
        cross_pivot = indicators['cross_above_pivot'].rolling(
            self.breakout_lookback
        ).apply(lambda x: np.any(x), raw=True).astype(bool)

        # Order block pattern
        bear_ob = indicators['bear_ob'].iloc[-1]

        # Trend filter
        downtrend = not indicators['uptrend'].iloc[-1]

        # Combine conditions
        return cross_pivot.iloc[-1] and bear_ob and downtrend

    def calculate_stops(self, ohlc, indicators, is_long):
        """Calculate stop loss and take profit levels"""
        atr = indicators['atr'].iloc[-1]
        close = indicators['close'].iloc[-1]

        if is_long:
            self.sl = close - atr * self.sl_atr_mult
            self.tp = close + atr * self.tp_atr_mult
        else:
            self.sl = close + atr * self.sl_atr_mult
            self.tp = close - atr * self.tp_atr_mult

    def check_conditions(self):
        """Execute trades based on state"""
        if self.state == 0:
            return

        symbol = self.symbols[0]
        symbol_info = mt5.symbol_info(symbol)

        if symbol_info is None:
            print(f"Failed to get symbol info for {symbol}")
            return

        digits = symbol_info.digits + 1
        tp = round(self.tp, digits)
        sl = round(self.sl, digits)

        positions_mean_entry, num_positions = self.get_positions_info()

        print(f"{self.tag}:: State: {self.state}, TP: {tp}, SL: {sl}, Positions: {num_positions}")

        if self.state in [1, -1]:
            try:
                self.execute(symbol=symbol, short=(self.state == -1), sl=sl, tp=tp)
                trade_type = "SELL" if self.state == -1 else "BUY"
                print(f"{self.tag}:: Entered {trade_type} for {symbol}, SL: {sl}, TP: {tp}")
            except Exception as e:
                print(f"{self.tag}:: Execution failed: {str(e)}")

    def log_signals(self, ohlc, indicators, long_signal, short_signal):
        """Log signal details for debugging"""
        print(f"\n{self.tag}:: --- Signal Analysis ---")
        print(f"Symbol: {self.symbols[0]}")
        print(f"Time: {ohlc.index[-1]}")
        print(f"Close: {indicators['close'].iloc[-1]:.5f}")
        print(f"SMA Short: {indicators['sma_short'].iloc[-1]:.5f}")
        print(f"SMA Long: {indicators['sma_long'].iloc[-1]:.5f}")
        print(f"Uptrend: {indicators['uptrend'].iloc[-1]}")
        print(f"Bull OB: {indicators['bull_ob'].iloc[-1]}")
        print(f"Bear OB: {indicators['bear_ob'].iloc[-1]}")
        print(f"ATR: {indicators['atr'].iloc[-1]:.5f}")
        print(f"Long Signal: {long_signal}")
        print(f"Short Signal: {short_signal}")
        if self.sl and self.tp:
            print(f"SL: {self.sl:.5f}, TP: {self.tp:.5f}")
        print(f"--- End Analysis ---\n")

    def fit(self, data=None):
        """No fitting required for this strategy"""
        print(f"{self.tag}:: No model to fit for MetaOrderBlock")
        return

    def get_positions_info(self):
        """Get current position information"""
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            return None, None

        positions = mt5.positions_get()
        if positions is None:
            return None, 0

        filtered_positions = [pos for pos in positions if pos.comment == self.tag]

        if filtered_positions:
            total_volume = sum(pos.volume for pos in filtered_positions)
            mid_price = sum(pos.price_open * pos.volume for pos in filtered_positions) / total_volume
            num_positions = len(filtered_positions)
        else:
            tick_info = mt5.symbol_info_tick(self.symbols[0])
            mid_price = tick_info.last if tick_info else (tick_info.bid + tick_info.ask) / 2
            num_positions = 0

        return mid_price, num_positions