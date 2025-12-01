import MetaTrader5 as mt5
import pytz
from dataclasses import dataclass, field
from datetime import timedelta, datetime
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

from metalib.metastrategy import MetaStrategy
from metalib.fastfinance import atr
from metalib.indicators import retrieve_high_pivot_point, retrieve_low_pivot_point


@dataclass
class FVGPattern:
    """Single FVG pattern data."""

    timestamp: datetime
    gap_low: float
    gap_high: float
    pattern_type: str
    candle_indices: List[int]
    direction: str  # "bullish" or "bearish"
    crossings: Optional[int] = None

    def contains_price(self, price: float) -> bool:
        """Check if price is within the FVG gap."""
        return self.gap_high < price < self.gap_low


@dataclass
class FVGPatternCollection:
    """Container for FVG pattern detection results."""

    patterns: List[FVGPattern] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.patterns)

    def is_empty(self) -> bool:
        return self.count == 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert patterns to DataFrame for compatibility with existing code."""
        if self.is_empty():
            return pd.DataFrame()
        return pd.DataFrame([vars(p) for p in self.patterns])

    def filter_by_max_crossings(self, max_crossings: int) -> "FVGPatternCollection":
        """Return new collection with patterns below crossing threshold."""
        filtered = [
            p
            for p in self.patterns
            if p.crossings is not None and p.crossings < max_crossings
        ]
        return FVGPatternCollection(patterns=filtered)

    def get_patterns_containing_price(self, price: float) -> List[FVGPattern]:
        """Return patterns where price is within the gap."""
        return [p for p in self.patterns if p.contains_price(price)]

    def get_latest_pattern(self) -> Optional[FVGPattern]:
        """Return the most recent pattern by timestamp."""
        if self.is_empty():
            return None
        return max(self.patterns, key=lambda p: p.timestamp)


@dataclass
class HTFFVGResult:
    """Results from HTF FVG detection and filtering."""

    all_patterns: FVGPatternCollection
    filtered_patterns: FVGPatternCollection

    @property
    def count_all(self) -> int:
        return self.all_patterns.count

    @property
    def count_filtered(self) -> int:
        return self.filtered_patterns.count


class MetaFVG(MetaStrategy):
    """MetaTrader FVG (Fair Value Gap) Trading Strategy"""

    # Configuration constants
    DEFAULT_HTF_TIMEFRAME = "4h"
    DEFAULT_LOOKBACK_DAYS = 7
    PIVOT_WINDOW = 7
    ATR_PERIOD = 14
    MOMENTUM_BODY_RATIO_THRESHOLD = 0.7

    def __init__(
        self,
        symbols: List[str],
        timeframe,
        size_position: float,
        tag: str,
        limit_number_position: int = 1,
    ):
        """
        Initialize the MetaFVG strategy.

        Args:
            symbols: List of trading symbols
            timeframe: Trading timeframe
            size_position: Position size
            tag: Strategy tag/identifier
            limit_number_position: Maximum number of positions
        """
        super().__init__(symbols, timeframe, tag, size_position)
        self._log(f"Initializing MetaFVG strategy with {symbols[0]}")

        # Trading state
        self.state: int = 0
        self.entry: Optional[float] = None
        self.sl: Optional[float] = None
        self.tp: Optional[float] = None

        # Configuration
        self.limit_number_position = limit_number_position
        self.risk_reward = 2
        self.atr_sensitivity = 4
        self.htf_fill_pct = 1
        # Crossing threshold multiplier (applied as * 2 during filtering)
        self.max_htf_number_crossings = 3

        # HTF FVG results (populated by fit())
        self.bullish_htf_result: Optional[HTFFVGResult] = None
        self.bearish_htf_result: Optional[HTFFVGResult] = None

    # =========================================================================
    # Logging
    # =========================================================================

    def _log(self, message: str) -> None:
        """Log a message with the strategy tag."""
        print(f"{self.tag}::    {message}")

    # =========================================================================
    # Data Loading and Preparation
    # =========================================================================

    def _load_recent_data(self, days: int) -> pd.DataFrame:
        """Load OHLC data for the specified number of days."""
        utc = pytz.UTC
        end_time = datetime.now(utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(days=days)

        self.loadData(start_time, end_time)
        return self.data[self.symbols[0]]

    def _resample_to_htf(
        self,
        ohlc: pd.DataFrame,
        timeframe: str = DEFAULT_HTF_TIMEFRAME,
    ) -> pd.DataFrame:
        """Resample OHLC data to higher timeframe."""
        return ohlc.resample(timeframe).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }
        )

    # =========================================================================
    # FVG Detection
    # =========================================================================

    def detect_fvg_htf(
        self, ohlc_df: pd.DataFrame
    ) -> Tuple[FVGPatternCollection, FVGPatternCollection]:
        """
        Detect HTF FVG patterns in OHLC data.

        Returns:
            Tuple of (bullish_patterns, bearish_patterns) as FVGPatternCollections
        """
        bullish_patterns: List[FVGPattern] = []
        bearish_patterns: List[FVGPattern] = []
        self._log("Started HTF FVG detection...")

        for i in tqdm(range(len(ohlc_df) - 2)):
            candle_i = ohlc_df.iloc[i]
            candle_i1 = ohlc_df.iloc[i + 1]  # Gap candle
            candle_i2 = ohlc_df.iloc[i + 2]

            # Check for bullish FVG: gap between high of i and low of i+2
            if candle_i["high"] < candle_i2["low"]:
                if candle_i1["close"] > candle_i1["open"]:  # Bullish candle
                    bullish_patterns.append(
                        FVGPattern(
                            timestamp=candle_i1.name,
                            gap_low=candle_i2["low"],
                            gap_high=candle_i["high"],
                            pattern_type="FVG_HTF",
                            candle_indices=[i, i + 1, i + 2],
                            direction="bullish",
                        )
                    )

            # Check for bearish FVG: gap between low of i and high of i+2
            if candle_i["low"] > candle_i2["high"]:
                if candle_i1["close"] < candle_i1["open"]:  # Bearish candle
                    bearish_patterns.append(
                        FVGPattern(
                            timestamp=candle_i1.name,
                            gap_low=candle_i["low"],
                            gap_high=candle_i2["high"],
                            pattern_type="FVG_HTF",
                            candle_indices=[i, i + 1, i + 2],
                            direction="bearish",
                        )
                    )

        return (
            FVGPatternCollection(patterns=bullish_patterns),
            FVGPatternCollection(patterns=bearish_patterns),
        )

    def detect_fvg_momentum_tres_strong(
        self, ohlc_df: pd.DataFrame, is_bullish: bool
    ) -> List[FVGPattern]:
        """
        Detect LTF Momentum FVG patterns.

        Args:
            ohlc_df: OHLC DataFrame to scan
            is_bullish: True for bullish patterns, False for bearish

        Returns:
            List of detected FVG patterns
        """
        patterns: List[FVGPattern] = []

        for i in range(len(ohlc_df) - 2):
            candle_i = ohlc_df.iloc[i]
            candle_i1 = ohlc_df.iloc[i + 1]  # Gap candle
            candle_i2 = ohlc_df.iloc[i + 2]

            # Momentum condition: strong body relative to range
            body_i1 = abs(candle_i1["close"] - candle_i1["open"])
            range_i1 = abs(candle_i1["high"] - candle_i1["low"])
            if range_i1 == 0:
                continue
            has_strong_momentum = (
                body_i1 / range_i1
            ) > self.MOMENTUM_BODY_RATIO_THRESHOLD

            if is_bullish:
                # Bullish FVG: gap between high of i and low of i+2
                has_gap = candle_i["high"] < candle_i2["low"]
                is_bullish_candle = candle_i1["close"] > candle_i1["open"]

                if has_gap and is_bullish_candle and has_strong_momentum:
                    patterns.append(
                        FVGPattern(
                            timestamp=candle_i1.name,
                            gap_low=candle_i2["low"],
                            gap_high=candle_i["high"],
                            pattern_type="MOMENTUM_FVG",
                            candle_indices=[i, i + 1, i + 2],
                            direction="bullish",
                        )
                    )
            else:
                # Bearish FVG: gap between low of i and high of i+2
                has_gap = candle_i["low"] > candle_i2["high"]
                is_bearish_candle = candle_i1["open"] > candle_i1["close"]

                if has_gap and is_bearish_candle and has_strong_momentum:
                    patterns.append(
                        FVGPattern(
                            timestamp=candle_i1.name,
                            gap_low=candle_i2["high"],
                            gap_high=candle_i["low"],
                            pattern_type="MOMENTUM_FVG",
                            candle_indices=[i, i + 1, i + 2],
                            direction="bearish",
                        )
                    )

        return patterns

    # =========================================================================
    # FVG Crossing Analysis
    # =========================================================================

    def _compute_crossing_count(
        self,
        pattern: FVGPattern,
        price_series: pd.Series,
        fill_pct: float,
    ) -> int:
        """
        Count how many times price crossed the FVG level after pattern formation.

        Args:
            pattern: The FVG pattern to analyze
            price_series: Price series (typically high or low)
            fill_pct: Fill percentage to determine the crossing level

        Returns:
            Number of crossings
        """
        fvg_level = fill_pct * pattern.gap_low + (1 - fill_pct) * pattern.gap_high
        idx_fvg = pattern.candle_indices[-1]

        # Get prices after FVG formation and compute crossings
        prices_after_fvg = price_series.iloc[idx_fvg:].values - fvg_level
        if len(prices_after_fvg) < 2:
            return 0

        # Count sign changes (crossings)
        sign_changes = prices_after_fvg[1:] * prices_after_fvg[:-1] < 0
        return int(np.sum(sign_changes))

    def _compute_crossings_for_collection(
        self,
        collection: FVGPatternCollection,
        price_series: pd.Series,
        fill_pct: float,
    ) -> FVGPatternCollection:
        """Compute crossing counts for all patterns in a collection."""
        for pattern in collection.patterns:
            pattern.crossings = self._compute_crossing_count(
                pattern, price_series, fill_pct
            )
        return collection

    # =========================================================================
    # HTF FVG Processing
    # =========================================================================

    def _process_htf_fvg_patterns(
        self,
        patterns: FVGPatternCollection,
        price_series: pd.Series,
        fill_pct: float,
        direction: str,
    ) -> HTFFVGResult:
        """
        Process HTF FVG patterns: compute crossings and filter.

        Args:
            patterns: Collection of detected patterns
            price_series: Price series for crossing calculation
            fill_pct: Fill percentage for crossing level
            direction: "Bullish" or "Bearish" for logging

        Returns:
            HTFFVGResult with all and filtered patterns
        """
        self._log(f"Found {patterns.count} {direction} FVG patterns on HTF")

        if patterns.is_empty():
            empty = FVGPatternCollection()
            return HTFFVGResult(all_patterns=empty, filtered_patterns=empty)

        # Compute crossings for all patterns
        self._compute_crossings_for_collection(patterns, price_series, fill_pct)

        # Filter by crossing threshold
        max_crossings = self.max_htf_number_crossings * 2
        filtered = patterns.filter_by_max_crossings(max_crossings)

        return HTFFVGResult(all_patterns=patterns, filtered_patterns=filtered)

    def _log_htf_results(self) -> None:
        """Log HTF FVG detection results."""
        for direction, result in [
            ("Bullish", self.bullish_htf_result),
            ("Bearish", self.bearish_htf_result),
        ]:
            if result is None:
                continue
            self._log(f"Finished {direction} FVG detection")
            self._log(f"{direction} patterns before filtering: {result.count_all}")
            self._log(f"{direction} patterns after filtering: {result.count_filtered}")

    # =========================================================================
    # Pivot Point Detection
    # =========================================================================

    def _retrieve_last_pivots(self, ohlc: pd.DataFrame) -> Tuple[float, float]:
        """
        Retrieve the most recent pivot low and pivot high.

        Returns:
            Tuple of (last_pivot_low, last_pivot_high)
        """
        last_pivot_low = (
            ohlc["low"]
            .rolling(self.PIVOT_WINDOW)
            .apply(retrieve_low_pivot_point, engine="numba", raw=True)
            .dropna()
            .iloc[-1]
        )

        last_pivot_high = (
            ohlc["high"]
            .rolling(self.PIVOT_WINDOW)
            .apply(retrieve_high_pivot_point, engine="numba", raw=True)
            .dropna()
            .iloc[-1]
        )

        return last_pivot_low, last_pivot_high

    # =========================================================================
    # Main Strategy Methods
    # =========================================================================

    def fit(self) -> None:
        """Detect and filter HTF FVG patterns from recent market data."""
        self._log("Starting FVG Detection")

        # Load and prepare data
        ohlc = self._load_recent_data(days=self.DEFAULT_LOOKBACK_DAYS)
        ohlc_htf = self._resample_to_htf(ohlc, self.DEFAULT_HTF_TIMEFRAME)

        # Detect patterns
        bullish_patterns, bearish_patterns = self.detect_fvg_htf(ohlc_htf)

        if bullish_patterns.is_empty() and bearish_patterns.is_empty():
            self._log(
                f"No FVG patterns found in the last {self.DEFAULT_LOOKBACK_DAYS} days on HTF"
            )
            # Initialize empty results
            empty = FVGPatternCollection()
            self.bullish_htf_result = HTFFVGResult(
                all_patterns=empty, filtered_patterns=empty
            )
            self.bearish_htf_result = HTFFVGResult(
                all_patterns=empty, filtered_patterns=empty
            )
            return

        # Process bullish patterns
        self.bullish_htf_result = self._process_htf_fvg_patterns(
            patterns=bullish_patterns,
            price_series=ohlc_htf["low"],
            fill_pct=self.htf_fill_pct,
            direction="Bullish",
        )

        # Process bearish patterns
        self.bearish_htf_result = self._process_htf_fvg_patterns(
            patterns=bearish_patterns,
            price_series=ohlc_htf["high"],
            fill_pct=1 - self.htf_fill_pct,
            direction="Bearish",
        )

        self._log_htf_results()

    def signals(self) -> None:
        """Generate trading signals based on FVG patterns."""
        self.state = 0
        self._log("Generating trading signals")

        # Check position limits
        _, current_open_position = self.get_positions_info()
        self._log(f"Current open positions: {current_open_position}")

        if current_open_position >= self.limit_number_position:
            self._log(f"Maximum open position reached: {current_open_position}")
            return

        # Validate HTF results exist
        if self.bullish_htf_result is None or self.bearish_htf_result is None:
            self._log("HTF FVG patterns not initialized. Run fit() first.")
            return

        # Get filtered patterns
        bullish_filtered = self.bullish_htf_result.filtered_patterns
        bearish_filtered = self.bearish_htf_result.filtered_patterns

        if bullish_filtered.is_empty() and bearish_filtered.is_empty():
            self._log("No HTF FVG patterns detected, no action required")
            return

        # Get current price data
        ohlc_ltf = self.data[self.symbols[0]]
        last_price = ohlc_ltf["close"].iloc[-1]

        self._log(f"Pulled data for symbol: {self.symbols[0]}")
        self._log(f"Last Price: ${last_price}")

        # Determine trading direction based on HTF FVG
        direction = self._determine_direction(
            last_price, bullish_filtered, bearish_filtered
        )

        if direction is None:
            self._log("Price not in any Bullish OR Bearish FVG H4, no action required")
            return

        # Look for LTF momentum FVG confirmation
        is_bullish = direction == 1
        momentum_patterns = self.detect_fvg_momentum_tres_strong(
            ohlc_ltf.iloc[-4:-1], is_bullish
        )

        if not momentum_patterns:
            self._log("No LTF FVG patterns detected, no action required")
            return

        current_momentum = momentum_patterns[0]

        # Validate direction alignment
        momentum_is_bullish = current_momentum.direction == "bullish"
        if momentum_is_bullish != is_bullish:
            self._log("LTF FVG direction doesn't match HTF, no action required")
            return

        self._log("We are in FVG H4!")

        # Calculate trade parameters
        self._calculate_trade_parameters(ohlc_ltf, current_momentum, is_bullish)

    def _determine_direction(
        self,
        last_price: float,
        bullish_patterns: FVGPatternCollection,
        bearish_patterns: FVGPatternCollection,
    ) -> Optional[int]:
        """
        Determine trading direction based on which FVG contains the price.

        Returns:
            1 for bullish, 0 for bearish, None if price not in any FVG
        """
        bullish_containing = bullish_patterns.get_patterns_containing_price(last_price)
        bearish_containing = bearish_patterns.get_patterns_containing_price(last_price)

        in_bullish = len(bullish_containing) > 0
        in_bearish = len(bearish_containing) > 0

        if not in_bullish and not in_bearish:
            return None

        if in_bullish and in_bearish:
            # Price in both - use the most recent FVG
            latest_bull = max(bullish_containing, key=lambda p: p.timestamp)
            latest_bear = max(bearish_containing, key=lambda p: p.timestamp)

            self._log("Price in both bearish and bullish FVG H4")
            self._log(f"Time of last bullish FVG H4: {latest_bull.timestamp}")
            self._log(f"Time of last bearish FVG H4: {latest_bear.timestamp}")

            return 1 if latest_bull.timestamp > latest_bear.timestamp else 0

        if in_bullish:
            self._log("Price in Bullish FVG H4, looking for long setups")
            return 1

        self._log("Price in Bearish FVG H4, looking for short setups")
        return 0

    def _calculate_trade_parameters(
        self,
        ohlc_ltf: pd.DataFrame,
        momentum_pattern: FVGPattern,
        is_bullish: bool,
    ) -> None:
        """Calculate entry, stop loss, and take profit levels."""
        ohlc_values = ohlc_ltf.values

        # Calculate ATR
        atr_value = atr(
            ohlc_values[:, 0],  # open
            ohlc_values[:, 1],  # high
            ohlc_values[:, 2],  # low
            self.ATR_PERIOD,
        )[-1]
        self._log(f"ATR value: {round(atr_value, 2)}$")

        # Get pivot points for stop loss
        last_pivot_low, last_pivot_high = self._retrieve_last_pivots(ohlc_ltf)

        # Calculate trade levels
        # State: 1 for long, -1 for short
        future_state = 1 if is_bullish else -1

        self.entry = (
            momentum_pattern.gap_low if is_bullish else momentum_pattern.gap_high
        )
        self.sl = last_pivot_low if is_bullish else last_pivot_high
        self.tp = (
            self.entry
            + atr_value * self.atr_sensitivity * self.risk_reward * future_state
        )
        self.state = future_state

    def check_conditions(self) -> None:
        """Check trading conditions and execute trades based on current state."""
        self._log(f"Checking conditions with state: {self.state}")

        if self.state == 0:
            self._log("State is 0, no action required")
            return

        symbol = self.symbols[0]
        symbol_info = mt5.symbol_info(symbol)

        if symbol_info is None:
            self._log(f"Failed to get symbol info for {symbol}")
            return

        # Use symbol_info.digits + 1 for proper rounding (after decimal)
        digits = symbol_info.digits + 1

        self._log(f"Rounding for {symbol}: {digits}")
        self._log(f"Strategy Volume: {self.size_position}")
        self._log(f"Strategy RRR: {self.risk_reward}")

        # Round trade levels
        tp = round(self.tp, digits)
        sl = round(self.sl, digits)
        entry = round(self.entry, digits)

        if self.state == 1:
            self._log(
                f"Long setup valid - SL: ${sl}, ENTRY: ${entry}, TP: ${tp}, "
                f"VOLUME: {self.size_position}"
            )
            self.execute(
                symbol=symbol,
                sl=sl,
                tp=tp,
                entry=entry,
                is_limit=True,
                is_eod=True,
            )
        elif self.state == -1:
            self._log(
                f"Short setup valid - SL: ${sl}, ENTRY: ${entry}, TP: ${tp}, "
                f"VOLUME: {self.size_position}"
            )
            self.execute(
                symbol=symbol,
                sl=sl,
                tp=tp,
                entry=entry,
                is_limit=True,
                is_eod=True,
            )
        else:
            self._log(f"Unknown state: {self.state}")

    # =========================================================================
    # Backward Compatibility Properties
    # =========================================================================
    # These properties maintain backward compatibility with any external code
    # that may reference the old attribute names

    @property
    def all_bullish_htf_fvg_patterns(self) -> pd.DataFrame:
        """Backward compatible access to all bullish patterns as DataFrame."""
        if self.bullish_htf_result is None:
            return pd.DataFrame()
        return self.bullish_htf_result.all_patterns.to_dataframe()

    @property
    def filt_bullish_htf_fvg_patterns(self) -> pd.DataFrame:
        """Backward compatible access to filtered bullish patterns as DataFrame."""
        if self.bullish_htf_result is None:
            return pd.DataFrame()
        return self.bullish_htf_result.filtered_patterns.to_dataframe()

    @property
    def all_bearish_htf_fvg_patterns(self) -> pd.DataFrame:
        """Backward compatible access to all bearish patterns as DataFrame."""
        if self.bearish_htf_result is None:
            return pd.DataFrame()
        return self.bearish_htf_result.all_patterns.to_dataframe()

    @property
    def filt_bearish_htf_fvg_patterns(self) -> pd.DataFrame:
        """Backward compatible access to filtered bearish patterns as DataFrame."""
        if self.bearish_htf_result is None:
            return pd.DataFrame()
        return self.bearish_htf_result.filtered_patterns.to_dataframe()
