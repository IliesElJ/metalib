import MetaTrader5 as mt5
import pytz
from datetime import timedelta, datetime
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
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
        print(f"{self.tag}::    Initializing MetaFVG strategy with {symbols[0]}")

        self.volume = size_position
        self.state = None
        self.entry = None
        self.sl = None
        self.tp = None
        self.limit_number_position = limit_number_position
        self.htf_fvg = None
        self.risk_reward = 2
        self.atr_sensitivity = 4
        self.htf_fill_pct = 1
        # The following argument will be multiplied by 2 when filtering.
        self.max_htf_number_crossings = 3

    def detect_fvg_momentum_tres_strong(self, ohlc_df: pd.DataFrame, direction) -> List[Dict]:
        """
        Detect FVG Pattern 1: Momentum très strong
        - Strong momentum with significant gap
        - Target hit fast
        """
        patterns = []

        for i in range(len(ohlc_df) - 2):
            candle_i        = ohlc_df.iloc[i]
            candle_i1       = ohlc_df.iloc[i + 1]  # Gap candle
            candle_i2       = ohlc_df.iloc[i + 2]

            body_i1         = abs(candle_i1['close'] - candle_i1['open'])
            range_i1        = abs(candle_i1['high'] - candle_i1['low'])
            momentum_cond   = body_i1 / range_i1 > 0.7

            # Check for bullish FVG: gap between high of i and low of i+2
            if direction:
                if candle_i['high'] < candle_i2['low']:
                    # Strong momentum conditions
                    if (candle_i1['close'] > candle_i1['open'] and  # Bullish candle
                        momentum_cond): # Strong body ratio

                        pattern = {
                            'timestamp': candle_i1.name,  # i+1 candle timestamp
                            'gap_low': candle_i2['low'],  # Low of i+2 candle
                            'gap_high': candle_i['high'],  # High of i candle
                            'pattern_type': 'MOMENTUM_FVG',
                            'candle_indices': [i, i + 1, i + 2],
                            'direction': 'bullish'
                        }
                        patterns.append(pattern)
            else:
                # Check for bearish FVG: gap between low of i and high of i+2
                if candle_i['low'] > candle_i2['high']:
                    if (candle_i1['open'] > candle_i1['close'] and  # Bearish candle
                        momentum_cond): # Strong body ratio

                        pattern = {
                            'timestamp': candle_i1.name,  # i+1 candle timestamp
                            'gap_low': candle_i2['high'],  # Low of i+2 candle
                            'gap_high': candle_i['low'],  # High of i candle
                            'pattern_type': 'MOMENTUM_FVG',
                            'candle_indices': [i, i + 1, i + 2],
                            'direction': 'bearish'
                        }
                        patterns.append(pattern)

        return patterns

    def detect_fvg_htf(self, ohlc_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect FVG Pattern 1: Momentum très strong
        - Strong momentum with significant gap
        - Target hit fast
        """
        bullish_patterns = []
        bearish_patterns = []
        print(f"{self.tag}::    Started Momentum FVG detection... pelo")

        for i in tqdm(range(len(ohlc_df) - 2)):
            candle_i = ohlc_df.iloc[i]
            candle_i1 = ohlc_df.iloc[i + 1]  # Gap candle
            candle_i2 = ohlc_df.iloc[i + 2]

            # Check for bullish FVG: gap between high of i and low of i+2
            if candle_i['high'] < candle_i2['low']:
                # Strong momentum conditions
                if candle_i1['close'] > candle_i1['open']:
                    pattern = {
                        'timestamp': candle_i1.name,  # i+1 candle timestamp
                        'gap_low': candle_i2['low'],  # Low of i+2 candle
                        'gap_high': candle_i['high'],  # High of i candle
                        'pattern_type': 'FVG_HTF',
                        'candle_indices': [i, i + 1, i + 2],
                        'direction': 'bullish'
                    }
                    bullish_patterns.append(pattern)

            # Check for bearish FVG: gap between low of i and high of i+2
            if candle_i['low'] > candle_i2['high']:
                # Strong momentum conditions
                if candle_i1['close'] < candle_i1['open']:
                    pattern = {
                        'timestamp': candle_i1.name,  # i+1 candle timestamp
                        'gap_low': candle_i['low'],  # Low of i candle
                        'gap_high': candle_i2['high'],  # High of i+2 candle
                        'pattern_type': 'FVG_HTF',
                        'candle_indices': [i, i + 1, i + 2],
                        'direction': 'bearish'
                    }
                    bearish_patterns.append(pattern)

        return bullish_patterns, bearish_patterns

    def check_conditions(self):
        """Check trading conditions and execute trades based on current state"""
        print(f"Checking conditions with state: {self.state}")

        if self.state == 0:
            print(f"{self.tag}::    State is 0, no action required")
            return

        symbol = self.symbols[0]
        symbol_info = mt5.symbol_info(symbol)
        digits = symbol_info.digits + 1  # Use symbol_info.digits for proper rounding + add one because its after the decimal

        if symbol_info is None:
            print(f"Failed to get symbol info for {symbol}")
            return

        print(f"{self.tag}::    Rounding for {symbol}:  {digits}")
        print(f"{self.tag}::    Strategy Volume:        {self.volume}")
        print(f"{self.tag}::    Strategy RRR:           {self.risk_reward}")

        # Proper rounding
        tp, sl, entry = round(self.tp, digits), round(self.sl, digits), round(self.entry, digits)


        if self.state == 1:
            print(f"{self.tag}::    Long setup is valid - SL: ${sl}, ENTRY : ${entry},TP : ${tp}, VOLUME: {self.volume}")
            self.execute(symbol =   self.symbols[0],
                         volume =   self.volume,
                         sl =       sl,
                         tp =       tp,
                         entry =    entry,
                         is_limit = True,
                         is_eod =   True
                         )
        else:
            print(f"{self.tag}::    Unknown state: {self.state}")

    def signals(self):
        """Generate trading signals based on FVG patterns"""
        self.state = 0
        print(f"{self.tag}::    Generating trading signals")

        _, current_open_position = self.get_positions_info()
        print(f"{self.tag}::    Current open positions: {current_open_position}")

        if current_open_position >= self.limit_number_position:
            print(f"{self.tag}::    Maximum open position reached, number of positions : {current_open_position}")
            return

        ohlc_ltf        = self.data[self.symbols[0]]
        ohlc_ltf_vals   = ohlc_ltf.values
        last_price = ohlc_ltf['close'][-1]

        print(f"{self.tag}::    Pulled data for symbol: {self.symbols[0]}")
        print(f"{self.tag}::    Last Price for symbol:  ${last_price}")

        if len(self.filt_bearish_htf_fvg_patterns) == 0 and len(self.filt_bearish_htf_fvg_patterns) == 0:
            print(f"{self.tag}::    No HTF FVG patterns detected, no action required")
            return

        bullish_htf_fvgs = self.filt_bullish_htf_fvg_patterns
        in_bullish_htf_fvg = bullish_htf_fvgs.apply(lambda x: (last_price < x["gap_low"]) and (last_price > x["gap_high"]), axis=1)

        bearish_htf_fvgs = self.filt_bearish_htf_fvg_patterns
        in_bearish_htf_fvg = bearish_htf_fvgs.apply(lambda x: (last_price < x["gap_low"]) and (last_price > x["gap_high"]), axis=1)

        if not in_bullish_htf_fvg.any() and not in_bearish_htf_fvg.any():
            print(f"{self.tag}::    Price not in any Bullish OR Bearish FVG H4, no action required")
            return
        else:
            if in_bearish_htf_fvg.any() and in_bearish_htf_fvg.any():
                last_fvg_bull = in_bullish_htf_fvg[in_bullish_htf_fvg].iloc[-1]["timestamp"]
                last_fvg_bear = in_bearish_htf_fvg[in_bearish_htf_fvg].iloc[-1]["timestamp"]
                direction = 1 if last_fvg_bull > last_fvg_bear else 0
                print(f"{self.tag}::    Price in both bearish and bullish FVG H4, direction: {direction}")
                print(f"{self.tag}::    Time of last bullish FVG H4: {last_fvg_bull}")
                print(f"{self.tag}::    Time of last bearish FVG H4: {last_fvg_bear}")
            if in_bullish_htf_fvg.any():
                print(f"{self.tag}::    Price in Bullish FVG H4, looking for long setups")
                direction = 1
            elif in_bearish_htf_fvg.any():
                print(f"{self.tag}::    Price in Bearish FVG H4, looking for short setups")
                direction = 0

        momentum_fvg_patterns = self.detect_fvg_momentum_tres_strong(ohlc_ltf.iloc[-4:-1], direction)

        if len(momentum_fvg_patterns) == 0:
            print(f"{self.tag}::    No LTF FVG patterns detected, no action required")
            return

        current_momentum    = momentum_fvg_patterns[0]

        if (current_momentum['direction'] != 'bullish') == direction:
            print(f"{self.tag}::    Price not in the correct direction for FVG H4, no action required")
            return

        higher_band         = current_momentum["gap_low"]
        lower_band          = current_momentum["gap_high"]
        future_state        = 2 * direction - 1 # If direction is 1, state is 1 (long) else it's -1 (short)

        print(f"{self.tag}:: We are in FVG H4 pello !!")

        atr_value   = atr(ohlc_ltf_vals[:, 0], ohlc_ltf_vals[:, 1], ohlc_ltf_vals[:, 2], 14)[-1]
        print(f"{self.tag}:: ATR value: {round(atr_value, 2)}$")

        self.entry  = higher_band if direction else lower_band
        self.sl     = self.entry - atr_value * self.atr_sensitivity * future_state
        self.tp     = self.entry + atr_value * self.atr_sensitivity * self.risk_reward * future_state
        self.state  = future_state

    def retrieve_fvg_crosses(self, fvg_pattern, price_ts, fill_pct):
        # On calcule le niveau sur lequel on veut verifier les croisements
        fvg_level = fill_pct*fvg_pattern["gap_low"] + (1 - fill_pct) * fvg_pattern["gap_high"]
        # On filtre la time series pour trouver uniquement les croisements formes apres la fvg
        idx_fvg = fvg_pattern["candle_indices"][-1]

        # On calcule le nombre de croisements
        filt_price_ts = price_ts.iloc[idx_fvg:].values
        filt_price_ts = filt_price_ts - fvg_level
        filt_price_ts = filt_price_ts[1:] * filt_price_ts[:-1] < 0

        return np.sum(filt_price_ts)

    def fit(self):
        print(f"{self.tag}::     Starting the FVG Detection pelo!!")        # Define the UTC timezone
        utc = pytz.timezone('UTC')
        # Get the current time in UTC
        end_time = datetime.now(utc)
        start_time = end_time - timedelta(days=7)
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

        bullish_htf_fvg_patterns, bearish_htf_fvg_patterns = self.detect_fvg_htf(ohlc_resampled_df)

        if len(bearish_htf_fvg_patterns) == 0 and len(bullish_htf_fvg_patterns) == 0:
            print(f"{self.tag}::     No FVG patterns found in the last 14 days on H4")
            return

        compute_crossing_bullish_fvg = lambda fvg_pattern: self.retrieve_fvg_crosses(fvg_pattern,
                                                                                     ohlc_resampled_df['low'],
                                                                                     self.htf_fill_pct)

        compute_crossing_bearish_fvg = lambda fvg_pattern: self.retrieve_fvg_crosses(fvg_pattern,
                                                                                     ohlc_resampled_df['high'],
                                                                                     1 - self.htf_fill_pct)

        # Filter Bullish FVG patterns:
        print(f"{self.tag}::     Found {len(bullish_htf_fvg_patterns)} Bullish FVG patterns in the last 14 days on H4.")
        bullish_htf_fvg_patterns = pd.DataFrame(bullish_htf_fvg_patterns)

        bullish_htf_fvg_patterns.loc[:, "crossings"] = bullish_htf_fvg_patterns.apply(compute_crossing_bullish_fvg, axis=1)
        filt_htf_fvg_patterns               = bullish_htf_fvg_patterns[bullish_htf_fvg_patterns["crossings"] < self.max_htf_number_crossings * 2]
        self.all_bullish_htf_fvg_patterns   = bullish_htf_fvg_patterns
        self.filt_bullish_htf_fvg_patterns   = filt_htf_fvg_patterns

        print(f"{self.tag}::     Finished Bulish FVG detection and fitting pelo!!")
        print(f"{self.tag}::     Number of Bulish FVG patterns before crossing filtering:  {self.all_bullish_htf_fvg_patterns.shape[0]}")
        print(f"{self.tag}::     Numver of Bulish FVG patterns after crossing filtering:   {self.filt_bullish_htf_fvg_patterns.shape[0]}")



        # Filter Bearish FVG patterns:
        print(f"{self.tag}::     Found {len(bearish_htf_fvg_patterns)} Bearish FVG patterns in the last 14 days on H4.")
        bearish_htf_fvg_patterns = pd.DataFrame(bearish_htf_fvg_patterns)

        bearish_htf_fvg_patterns.loc[:, "crossings"] = bearish_htf_fvg_patterns.apply(compute_crossing_bearish_fvg, axis=1)
        filt_htf_fvg_patterns             = bearish_htf_fvg_patterns[bearish_htf_fvg_patterns["crossings"] < self.max_htf_number_crossings * 2]
        self.all_bearish_htf_fvg_patterns = bearish_htf_fvg_patterns
        self.filt_bearish_htf_fvg_patterns = filt_htf_fvg_patterns

        print(f"{self.tag}::     Finished Bearish FVG detection and fitting pelo!!")
        print(f"{self.tag}::     Number of Bearish FVG patterns before crossing filtering:  {self.all_bearish_htf_fvg_patterns.shape[0]}")
        print(f"{self.tag}::     Numver of Bearish FVG patterns after crossing filtering:   {self.filt_bearish_htf_fvg_patterns.shape[0]}")

        return

