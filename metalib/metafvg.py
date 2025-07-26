import MetaTrader5 as mt5
import pytz
from datetime import timedelta, datetime
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

from metalib.metastrategy import MetaStrategy
from metalib.fastfinance import atr

class MetaFVG(MetaStrategy):
    """MetaTrader FVG (Fair Value Gap) Trading Strategy"""

    def __init__(self,
                 symbols,
                 timeframe,
                 size_position,
                 tag,
                 limit_number_position=1):
        """
        Initialize the MetaFvg strategy

        Args:
            symbols: List of trading symbols
            timeframe: Trading timeframe
            size_position: Position size
            sl: Stop loss sensitivity
            tag: Strategy tag/identifier
            tp: Take profit level
            limit_number_position: Maximum number of positions
        """
        super().__init__(symbols, timeframe, tag)
        print(f"Initializing MetaFvg strategy with symbols: {symbols}, timeframe: {timeframe}")

        self.volume = size_position
        self.state = None
        self.entry = None
        self.sl = None
        self.tp = None
        self.limit_number_position = limit_number_position
        self.htf_fvg = None
        self.risk_reward = 3
        self.atr_sensitivity = 2

    def detect_fvg_momentum_tres_strong(self, ohlc_df: pd.DataFrame) -> List[Dict]:
        """
        Detect FVG Pattern 1: Momentum très strong
        - Strong momentum with significant gap
        - Target hit fast
        """
        patterns = []

        for i in range(len(ohlc_df) - 2):
            candle_i = ohlc_df.iloc[i]
            candle_i1 = ohlc_df.iloc[i + 1]  # Gap candle
            candle_i2 = ohlc_df.iloc[i + 2]

            # Check for bullish FVG: gap between high of i and low of i+2
            if candle_i['high'] < candle_i2['low']:
                # Additional momentum checks for "très strong"
                body_i1 = abs(candle_i1['close'] - candle_i1['open'])
                range_i1 = candle_i1['high'] - candle_i1['low']

                # Strong momentum conditions
                if (candle_i1['close'] > candle_i1['open'] and  # Bullish candle
                        body_i1 / range_i1 > 0.7): # Strong body ratio

                    pattern = {
                        'timestamp': candle_i1.name,  # i+1 candle timestamp
                        'gap_low': candle_i2['low'],  # Low of i+2 candle
                        'gap_high': candle_i['high'],  # High of i candle
                        'pattern_type': 'MOMENTUM_FVG',
                        'candle_indices': [i, i + 1, i + 2]
                    }
                    patterns.append(pattern)

        return patterns

    def detect_fvg_htf(self, ohlc_df):
        """
        Detect FVG Pattern 1: Momentum très strong
        - Strong momentum with significant gap
        - Target hit fast
        """
        patterns = []
        print(f"{self.tag}::    Started Momentum FVG detection... pelo")

        for i in tqdm(range(len(ohlc_df) - 2)):
            candle_i = ohlc_df.iloc[i]
            candle_i1 = ohlc_df.iloc[i + 1]  # Gap candle
            candle_i2 = ohlc_df.iloc[i + 2]

            # Check for bullish FVG: gap between high of i and low of i+2
            if candle_i['high'] < candle_i2['low']:
                # Additional momentum checks for "très strong"
                body_i1 = abs(candle_i1['close'] - candle_i1['open'])
                range_i1 = candle_i1['high'] - candle_i1['low']
                # Strong momentum conditions
                if (candle_i1['close'] > candle_i1['open']):
                    pattern = {
                        'timestamp': candle_i1.name,  # i+1 candle timestamp
                        'gap_low': candle_i2['low'],  # Low of i+2 candle
                        'gap_high': candle_i['high'],  # High of i candle
                        'pattern_type': 'FVG_HTF',
                        'candle_indices': [i, i + 1, i + 2]
                    }
                    patterns.append(pattern)

        return patterns

    def check_conditions(self):
        """Check trading conditions and execute trades based on current state"""
        print(f"Checking conditions with state: {self.state}")

        if self.state == 0:
            print("State is 0, no action required")
            pass
        elif self.state == 1:
            print(f"Long setup valid - SL: {self.sl},ENTRY : {self.entry},TP : {self.tp}, VOLUME: {self.volume}")
            self.execute(symbol=self.symbols[0], volume=self.volume, sl=self.sl, tp=self.tp,
                         entry = self.entry, is_limit = True, is_eod=True)
        else:
            print(f"Unknown state: {self.state}")

    def signals(self):
        """Generate trading signals based on FVG patterns"""
        print(f"{self.tag}::    Generating trading signals")

        _, current_open_position = self.get_positions_info()
        print(f"{self.tag}::    Current open positions: {current_open_position}")

        if current_open_position >= self.limit_number_position:
            print(f"{self.tag}::    Maximum open position reached, number of positions : {current_open_position}")
            return

        ohlc_ltf        = self.data[self.symbols[0]]
        ohlc_ltf_vals   = ohlc_ltf.values
        print(f"{self.tag}::    Pulled data for symbol: {self.symbols[0]}")

        last_price = ohlc_ltf['close'][-1]
        momentum_fvg_patterns = self.detect_fvg_momentum_tres_strong(ohlc_ltf.iloc[-4:-1])

        if len(momentum_fvg_patterns) == 0:
            print(f"{self.tag}::    No fvg patterns detected, no action required")
            return

        current_momentum    = momentum_fvg_patterns[0]
        higher_band         = current_momentum["gap_low"]

        htf_fvgs = self.htf_fvg
        in_htf_fvg = htf_fvgs.apply(lambda x: (last_price < x["gap_low"]) and (last_price > x["gap_high"]), axis=1)

        if not in_htf_fvg.any():
            print(f"{self.tag}::    Price not in any FVG H4, no action required")
            return
        print(f"{self.tag}:: We are in FV H4 pello !!")

        atr_value   = atr(ohlc_ltf_vals[:, 0], ohlc_ltf_vals[:, 1], ohlc_ltf_vals[:, 2], 14)[-1]

        self.entry = higher_band
        self.sl = atr_value * self.atr_sensitivity - self.entry
        self.tp = self.risk_reward
        self.state  = 1


    def fit(self):
        print(f"{self.tag}::     Starting the FVG Detection pelo!!")

        # Define the UTC timezone
        utc = pytz.timezone('UTC')
        # Get the current time in UTC
        end_time = datetime.now(utc)
        start_time = end_time - timedelta(days=14)
        # Set the time components to 0 (midnight) and maintain the timezone
        end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(utc)
        start_time = start_time.astimezone(utc)

        # Pulling last days of data
        self.loadData(start_time, end_time)
        ohlc            = self.data[self.symbols[0]]
        ohlc_resampled_df = ohlc.resample('4h').agg({'open': 'first',
                             'high': 'max',
                             'low': 'min',
                             'close': 'last'},
                           label="right",
                           closed="right")

        htf_fvg_pattern = self.detect_fvg_htf(ohlc_resampled_df)

        if len(htf_fvg_pattern) == 0:
            print(f"No FVG patterns found in the last 14 days on H4")
        else:

            print(f"Found {len(htf_fvg_pattern)} FVG patterns in the last 14 days on H4.")

        self.htf_fvg = pd.DataFrame(htf_fvg_pattern)

        print(f"{self.tag}::     Finished FVG detection and fitting pelo!!")
        return

    def check_positions_and_orders(self):
        """Check if there are open positions and orders for the symbol"""
        print(f"Checking positions and orders for {self.symbols[0]}")

        try:
            open_positions = mt5.positions_get(symbol=self.symbols[0])
            if open_positions is None:
                print(f"Failed to get open positions for symbol: {self.symbols[0]}")
                return False

            filtered_open_positions = [position for position in open_positions if position.comment == self.tag]
            open_positions_count = len(filtered_open_positions)
            print(f"Open positions with tag '{self.tag}': {open_positions_count}")

            orders = mt5.orders_get(symbol=self.symbols[0])
            if orders is None:
                print(f"Failed to get orders for symbol: {self.symbols[0]}")
                return False

            filtered_orders = [order for order in orders if order.comment == self.tag]
            pending_orders_count = len(filtered_orders)
            print(f"Pending orders with tag '{self.tag}': {pending_orders_count}")

            result = open_positions_count == 0 and pending_orders_count <= 1
            print(f"Position check result: {result}")
            return result

        except Exception as e:
            print(f"Error in check_positions_and_orders: {e}")
            return False

