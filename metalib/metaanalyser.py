from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytz as pytz
from metalib.metastrategy import MetaStrategy


class MetaAnalyser(MetaStrategy):
    def __init__(self, symbols, timeframe, tag, active_hours, vol_window=24, hist_length=1000, vol_tf="4h"):
        super().__init__(symbols, timeframe, tag, active_hours)
        self.vol_window = vol_window  # 4h periods for volatility calculation
        self.hist_length = hist_length  # Historical periods for ranking
        self.vols = {}  # Store volatility data for each symbol
        self.vol_tf = vol_tf
        self.current_vol = None
        self.vol_rank = None
        self.fitted_vols = None

    def signals(self):
        """Compute current 4h volatility and its rank/quantile"""
        # Load current data like MetaGO
        ohlc = self.data[self.symbols[0]]

        # Calculate 4h realized volatility
        returns = np.log(ohlc['close'] / ohlc['close'].shift(1))
        current_vol = returns.rolling(window=self.vol_window).std() * np.sqrt(self.vol_window)

        # Store volatility data
        self.vols[self.symbols[0]] = current_vol
        self.current_vol = current_vol.iloc[-1]

        # Calculate rank/quantile of current vol vs historical
        historical_vols = self.fitted_vols[self.symbols[0]]
        self.vol_rank = (historical_vols < self.current_vol).mean()

        print(f"{self.tag}::: Current 4h Vol: {self.current_vol:.4f}, Rank: {self.vol_rank:.2f}")

        # Create signal data for storage
        signal_line = pd.Series({
            'timestamp': ohlc.index[-1],
            'current_vol': self.current_vol,
            'vol_rank': self.vol_rank,
            'symbol': self.symbols[0]
        })

        self.signalData = signal_line

    def check_conditions(self):
        """Evaluate vol rank thresholds and trigger actions"""
        if self.vol_rank is None:
            return

        # Example thresholds - customize as needed
        if self.vol_rank > 0.95:
            print(f"{self.tag}::: EXTREME HIGH volatility detected (rank: {self.vol_rank:.2f})")
            self.send_telegram_message(
                f"🔴 EXTREME HIGH VOL: {self.symbols[0]} - Vol: {self.current_vol:.4f} (Rank: {self.vol_rank:.2f})")
        elif self.vol_rank < 0.05:
            print(f"{self.tag}::: EXTREME LOW volatility detected (rank: {self.vol_rank:.2f})")
            self.send_telegram_message(
                f"🟢 EXTREME LOW VOL: {self.symbols[0]} - Vol: {self.current_vol:.4f} (Rank: {self.vol_rank:.2f})")

    def fit(self):
        """Load historical data for volatility computation like MetaGO"""
        # Define UTC timezone and time range like MetaGO
        utc = pytz.timezone('UTC')
        end_time = datetime.now(utc)
        start_time = end_time - timedelta(days=60)
        end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(utc)
        start_time = start_time.astimezone(utc)

        # Load historical data for fitting
        self.loadData(start_time, end_time)

        ohlc = self.data[self.symbols[0]]

        # Calculate 4h realized volatility
        returns = np.log(ohlc['close'] / ohlc['close'].shift(1))
        vols = returns.resample(self.vol_tf).apply(lambda x: x.std())
        self.fitted_vols[self.symbols[0]] = vols

        print(f"{self.tag}::: Loaded {len(self.data[self.symbols[0]])} periods for volatility analysis")
        print(
            f"{self.tag}::: Data range: {self.data[self.symbols[0]].index[0]} to {self.data[self.symbols[0]].index[-1]}")