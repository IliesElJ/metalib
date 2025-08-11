from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytz as pytz
from metalib.metastrategy import MetaStrategy


class MetaVolAn(MetaStrategy):
    def __init__(self, symbols, timeframe, tag, active_hours, vol_window=24, hist_length=1000, vol_tf="4h"):
        super().__init__(symbols, timeframe, tag, active_hours)
        self.vol_window = vol_window  # 4h periods for volatility calculation
        self.hist_length = hist_length  # Historical periods for ranking
        self.vols = {}  # Store volatility data for each symbol
        self.vol_tf = vol_tf  # Volatility timeframe for resampling
        self.fitted_vols = {}  # Store fitted volatilities for each symbol
        self.fitted_covariances = {}  # Store fitted covariance matrices by period
        self.fitted_correlations = {}  # Store fitted correlation matrices by period
        self.correlations = None  # Store current correlation matrix
        self.covariances = None  # Store current covariance matrix

    def signals(self):
        """Compute current 4h volatility and its rank/quantile for all symbols"""
        signal_data = []
        returns_data = {}

        for symbol in self.symbols:
            ohlc = self.data[symbol]

            # Calculate 4h realized volatility
            returns = np.log(ohlc['close'] / ohlc['close'].shift(1))
            current_vol = returns.rolling(window=self.vol_window).std() * np.sqrt(self.vol_window)

            # Store returns for correlation/covariance computation
            returns_data[symbol] = returns

            # Store volatility data
            self.vols[symbol] = current_vol
            current_vol_value = current_vol.iloc[-1]

            # Calculate rank/quantile of current vol vs historical fitted vols
            historical_vols = self.fitted_vols[symbol]
            vol_rank = (historical_vols < current_vol_value).mean()

            print(f"{self.tag}::: {symbol} - Current 4h Vol: {current_vol_value:.4f}, Rank: {vol_rank:.2f}")

            # Store individual symbol data
            signal_data.append({
                'timestamp': ohlc.index[-1],
                'current_vol': current_vol_value,
                'vol_rank': vol_rank,
                'symbol': symbol
            })

        # Compute correlations and covariances
        if len(self.symbols) > 1:
            returns_df = pd.DataFrame(returns_data).dropna()

            # Calculate correlation matrix
            self.correlations = returns_df.corr()

            # Calculate covariance matrix (annualized)
            self.covariances = returns_df.cov() * (252 * 6)  # Assuming 4h bars, 6 per day

            print(f"{self.tag}::: Computed correlation and covariance matrices")
            print(f"Correlations:\n{self.correlations.round(3)}")

            # Add correlation/covariance info to signal data
            for i, row in enumerate(signal_data):
                symbol = row['symbol']
                # Add average correlation with other symbols
                avg_corr = self.correlations[symbol].drop(symbol).mean()
                signal_data[i]['avg_correlation'] = avg_corr
                signal_data[i]['volatility_contribution'] = self.covariances[symbol][symbol]

        # Create combined signal data for storage
        self.signalData = pd.DataFrame(signal_data)

    def check_conditions(self):
        """Evaluate vol rank thresholds and trigger actions for all symbols"""
        if not hasattr(self, 'signalData') or self.signalData is None:
            return

        for _, row in self.signalData.iterrows():
            symbol = row['symbol']
            vol_rank = row['vol_rank']
            current_vol = row['current_vol']

            # Example thresholds - customize as needed
            if vol_rank > 0.95:
                print(f"{self.tag}::: {symbol} EXTREME HIGH volatility detected (rank: {vol_rank:.2f})")
                message = f"🔴 EXTREME HIGH VOL: {symbol} - Vol: {current_vol:.4f} (Rank: {vol_rank:.2f})"
                if 'avg_correlation' in row:
                    message += f" | Avg Corr: {row['avg_correlation']:.3f}"
                self.send_telegram_message(message)
            elif vol_rank < 0.05:
                print(f"{self.tag}::: {symbol} EXTREME LOW volatility detected (rank: {vol_rank:.2f})")
                message = f"🟢 EXTREME LOW VOL: {symbol} - Vol: {current_vol:.4f} (Rank: {vol_rank:.2f})"
                if 'avg_correlation' in row:
                    message += f" | Avg Corr: {row['avg_correlation']:.3f}"
                self.send_telegram_message(message)

        # Check for high correlation conditions across portfolio
        if self.correlations is not None and len(self.symbols) > 1:
            # Get upper triangle of correlation matrix (exclude diagonal)
            upper_corr = self.correlations.where(np.triu(np.ones(self.correlations.shape), k=1).astype(bool))
            max_corr = upper_corr.max().max()
            min_corr = upper_corr.min().min()

            if max_corr > 0.9:
                print(f"{self.tag}::: HIGH CORRELATION detected: {max_corr:.3f}")
                self.send_telegram_message(f"⚠️ HIGH CORRELATION detected: {max_corr:.3f}")
            elif min_corr < -0.9:
                print(f"{self.tag}::: HIGH NEGATIVE CORRELATION detected: {min_corr:.3f}")
                self.send_telegram_message(f"⚠️ HIGH NEGATIVE CORRELATION detected: {min_corr:.3f}")

    def fit(self):
        """Load historical data for volatility computation for all symbols"""
        # Define UTC timezone and time range like MetaGO
        utc = pytz.timezone('UTC')
        end_time = datetime.now(utc)
        start_time = end_time - timedelta(days=60)
        end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(utc)
        start_time = start_time.astimezone(utc)

        # Load historical data for fitting
        self.loadData(start_time, end_time)

        # Collect all returns for cross-symbol analysis
        all_returns = {}

        # Fit volatilities for each symbol
        for symbol in self.symbols:
            ohlc = self.data[symbol]

            # Calculate volatility using resampling approach like MetaAnalyser
            returns = np.log(ohlc['close'] / ohlc['close'].shift(1))
            vols = returns.resample(self.vol_tf).apply(lambda x: x.std())
            self.fitted_vols[symbol] = vols

            # Store returns for correlation/covariance computation
            all_returns[symbol] = returns

            print(f"{self.tag}::: {symbol} - Fitted {len(vols)} volatility periods")

        # Compute fitted correlations and covariances by resampled periods
        if len(self.symbols) > 1:
            returns_df = pd.DataFrame(all_returns).dropna()

            # Resample returns and compute rolling correlations/covariances
            resampled_returns = returns_df.resample(self.vol_tf)

            for period, period_returns in resampled_returns:
                if len(period_returns) > 1:  # Need at least 2 observations
                    # Compute correlation matrix for this period
                    period_corr = period_returns.corr()
                    self.fitted_correlations[period] = period_corr

                    # Compute covariance matrix for this period (annualized)
                    period_cov = period_returns.cov() * (252 * 6)  # Assuming 4h bars, 6 per day
                    self.fitted_covariances[period] = period_cov

            print(f"{self.tag}::: Fitted {len(self.fitted_correlations)} correlation matrices")
            print(f"{self.tag}::: Fitted {len(self.fitted_covariances)} covariance matrices")

        print(f"{self.tag}::: Loaded {len(self.data[self.symbols[0]])} periods for volatility analysis")
        print(
            f"{self.tag}::: Data range: {self.data[self.symbols[0]].index[0]} to {self.data[self.symbols[0]].index[-1]}")
        print(f"{self.tag}::: Fitted volatilities for {len(self.symbols)} symbols")