from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
import pandas as pd
import pytz
import numpy as np
from metalib.metastrategy import MetaStrategy
from metalib.indicators import (
    rolling_mean_nb,
    pct_change_nb,
    rolling_sharpe_nb,
    rolling_min_shift1_nb,
    rolling_max_shift1_nb,
    true_range_nb,
    order_blocks_nb,
    crosses_nb,
)


class MetaOB(MetaStrategy):
    """
    Order Block trading strategy that uses pivot breakouts and order blocks for entry signals.

    This strategy combines technical analysis concepts including:
    - Order blocks (bullish/bearish patterns)
    - Pivot point breakouts
    - Trend filtering using rolling Sharpe ratio
    - ATR-based stop loss and take profit levels
    """

    def __init__(
        self,
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
        super().__init__(
            symbols, timeframe, tag, size_position, active_hours
        )

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
        self.sharpe_threshold_long = None
        self.sharpe_threshold_short = None
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
        if (
            long_signal
        ):  # and not self.are_positions_with_tag_open(position_type="buy"):
            self.state = 1
        elif (
            short_signal
        ):  # and not self.are_positions_with_tag_open(position_type="sell" ):
            self.state = -1
        else:
            self.state = 0

        # Logging
        self.log_signals(ohlc, indicators, long_signal, short_signal)

        # Save signal data
        signal_line = indicators.iloc[-1].copy()
        signal_line["timestamp"] = ohlc.index[-1]
        signal_line["state"] = self.state
        signal_line["symbol"] = self.symbols[0]
        signal_line["long_signal"] = long_signal
        signal_line["short_signal"] = short_signal
        self.signalData = signal_line

    def calculate_indicators(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators needed for the strategy.

        Args:
            ohlc: DataFrame with OHLC price data

        Returns:
            DataFrame with calculated indicators including SMAs, Sharpe ratio, pivots, ATR, and order blocks
        """
        indicators = pd.DataFrame(index=ohlc.index)

        o = ohlc["open"].to_numpy(dtype=np.float64)
        h = ohlc["high"].to_numpy(dtype=np.float64)
        l = ohlc["low"].to_numpy(dtype=np.float64)
        c = ohlc["close"].to_numpy(dtype=np.float64)

        indicators["close"] = c

        # SMAs (sequential rolling sum)
        indicators["sma_short"] = rolling_mean_nb(c, int(self.sma_short_hours))
        indicators["sma_long"] = rolling_mean_nb(c, int(self.sma_long_hours))

        # Rolling Sharpe (sequential, but returns computed in parallel)
        rets = pct_change_nb(c)
        rolling_sh = rolling_sharpe_nb(rets, int(self.sma_long_hours))
        indicators["rolling_sharpe"] = rolling_sh

        indicators["uptrend"] = rolling_sh > float(self.sharpe_threshold_long)
        indicators["downtrend"] = rolling_sh < float(self.sharpe_threshold_short)

        # Pivot points (parallel, good candidate)
        w = int(self.pivot_window)
        pivot_low = rolling_min_shift1_nb(l, w)
        pivot_high = rolling_max_shift1_nb(h, w)
        indicators["pivot_low"] = pivot_low
        indicators["pivot_high"] = pivot_high

        # ATR: TR parallel, rolling mean sequential
        tr = true_range_nb(h, l, c)
        indicators["atr"] = rolling_mean_nb(tr, int(self.atr_period))

        # Order blocks (parallel)
        bull_ob, bear_ob = order_blocks_nb(o, h, l, c)
        indicators["bull_ob"] = bull_ob
        indicators["bear_ob"] = bear_ob

        # Crosses (parallel)
        cross_below, cross_above = crosses_nb(l, h, pivot_low, pivot_high)
        indicators["cross_below_pivot"] = cross_below
        indicators["cross_above_pivot"] = cross_above

        return indicators

    def detect_long_signal(self, ohlc, indicators):
        """Detect long entry signal"""
        # Check if pivot low was crossed recently
        cross_pivot = (
            indicators["cross_below_pivot"]
            .rolling(self.breakout_lookback)
            .apply(lambda x: np.max(x), raw=True)
            .astype(bool)
        )

        # Order block pattern
        bull_ob = indicators["bull_ob"].iloc[-1]

        # Trend filter
        uptrend = indicators["uptrend"].iloc[-1]

        # Combine conditions
        return cross_pivot.iloc[-1] and bull_ob and uptrend

    def detect_short_signal(self, ohlc, indicators):
        """Detect short entry signal"""
        # Check if pivot high was crossed recently
        cross_pivot = (
            indicators["cross_above_pivot"]
            .rolling(self.breakout_lookback)
            .apply(lambda x: np.max(x), raw=True)
            .astype(bool)
        )

        # Order block pattern
        bear_ob = indicators["bear_ob"].iloc[-1]

        # Trend filter
        downtrend = indicators["downtrend"].iloc[-1]

        # Combine conditions
        return cross_pivot.iloc[-1] and bear_ob and downtrend

    def calculate_stops(self, ohlc, indicators, is_long):
        """Calculate stop loss and take profit levels"""
        atr = indicators["atr"].iloc[-1]
        close = indicators["close"].iloc[-1]

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

        print(
            f"{self.tag}:: State: {self.state}, TP: {tp}, SL: {sl}, Positions: {num_positions}"
        )

        if self.state in [1, -1]:
            try:
                self.execute(symbol=symbol, short=(self.state == -1), sl=sl, tp=tp)
                trade_type = "SELL" if self.state == -1 else "BUY"
                print(
                    f"{self.tag}:: Entered {trade_type} for {symbol}, SL: {sl}, TP: {tp}"
                )
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
        print(
            f"Sharpe Ratio: {indicators['rolling_sharpe'].iloc[-1]:.5f} vs Threshold: {self.sharpe_threshold_long:.5f}/{self.sharpe_threshold_short:.5f}"
        )
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
        # Define the UTC timezone
        utc = pytz.timezone("UTC")
        # Get the current time in UTC
        end_time = datetime.now(utc)
        start_time = end_time - timedelta(days=66)
        # Set the time components to 0 (midnight) and maintain the timezone
        end_time = end_time.replace(
            hour=0, minute=0, second=0, microsecond=0
        ).astimezone(utc)
        start_time = start_time.astimezone(utc)

        # Pulling last days of data
        self.loadData(start_time, end_time)
        data = self.data[self.symbols[0]]
        close = data["close"]
        rolling_sharpe = (
            close.pct_change()
            .rolling(self.sma_long_hours)
            .apply(lambda x: np.mean(x) / np.std(x))
        )
        self.sharpe_threshold_long = rolling_sharpe.quantile(0.75)
        self.sharpe_threshold_short = rolling_sharpe.quantile(0.25)
        print(
            f"{self.tag}:: Sharpe Uptrend quantile is: {round(self.sharpe_threshold_long, 4)}"
        )
        print(
            f"{self.tag}:: Sharpe Downtrend quantile is: {round(self.sharpe_threshold_short, 4)}"
        )
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
            mid_price = (
                sum(pos.price_open * pos.volume for pos in filtered_positions)
                / total_volume
            )
            num_positions = len(filtered_positions)
        else:
            tick_info = mt5.symbol_info_tick(self.symbols[0])
            mid_price = (
                tick_info.last if tick_info else (tick_info.bid + tick_info.ask) / 2
            )
            num_positions = 0

        return mid_price, num_positions
