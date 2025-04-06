from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
import pytz as pytz
import xgboost as xgb

from metalib.indicators import *
from metalib.metastrategy import MetaStrategy


class MetaGA(MetaStrategy):

    def __init__(   self, 
                    symbols, 
                    timeframe, 
                    tag, 
                    active_hours, 
                    low_length=60, 
                    mid_length=8*60,
                    high_length=24*60,
                    prob_bound=0.05,
                    risk_factor=1, 
                ):
        super().__init__(symbols, timeframe, tag, active_hours)
        
        if not (low_length < mid_length < high_length):
            raise ValueError("Length parameters should be ordered.")
        
        self.indicators     = None
        self.quantile       = None
        self.model          = None
        self.indicators_std = None
        self.indicators_mean= None
        self.low_length     = low_length 
        self.mid_length     = mid_length
        self.high_length    = high_length
        self.prob_bound     = prob_bound
        self.state          = None
        self.risk_factor    = risk_factor
        self.telegram       = True
        self.logger         = logging.getLogger(__name__)
        
        logging.basicConfig(filename=f'../logs/{self.tag}.log', encoding='utf-8', level=logging.DEBUG)

    def signals(self):

        ohlc = self.data[self.symbols[0]]
        indicators = self.retrieve_indicators(ohlc)

        self.indicators = indicators

        # Demean Indicators
        indicators = indicators.tail(3)
        indicators = (indicators - self.indicators_mean) / self.indicators_std

        dummy_extremes_indicators = abs(indicators) > 1.0

        y_hat = self.model.predict_proba(indicators)[:, 1]
        vote = np.sum(dummy_extremes_indicators.iloc[-1])
        mean_entry_price, num_positions = self.get_positions_info()

        if y_hat[-1] < 0.3 and self.are_positions_with_tag_open(position_type="buy"):
            self.state = -2
        elif y_hat[-1] > 0.7 and self.are_positions_with_tag_open(position_type="sell"):
            self.state = -2
        elif vote >= 18 and y_hat[-1] > 1 - self.prob_bound and num_positions < 5:
            self.state = 1
        elif vote >= 18 and y_hat[-1] < self.prob_bound and num_positions < 5:
            if ohlc.iloc[-1]['close'] < self.quantile:
                self.state = -1
            else:
                self.state = 0
        else:
            self.state = 0

        print(f"{self.tag}::: Open positions for strategy: {self.tag}: {self.are_positions_with_tag_open()}")
        print(f"{self.tag}::: Vote of indicators: {vote}, and last 3 predicted probabilities: {y_hat}")

        signal_line = indicators.iloc[[-1]]
        signal_line.loc[:, 'vote'] = vote
        signal_line.loc[:, 'predicted_proba'] = y_hat[-1]

        self.signals_data = signal_line

    def position_sizing_vol_adjusted(self, risk_percentage=0.01, account_balance=None):
        """
        Adjusts position size based on daily volatility to dollar-scale positions.

        Parameters:
            symbol (str): Trading symbol.
            risk_percentage (float): Percentage of account balance to risk per trade (default is 1%).
            account_balance (float): Optional, current account balance. If None, retrieved from MetaTrader5.

        Returns:
            float: Adjusted position size (in lots).
        """
        
        # Retrieve account balance if not provided
        if account_balance is None:
            account_info = mt5.account_info()
            if account_info is None:
                print("Failed to get account balance, error code =", mt5.last_error())
                mt5.shutdown()
                return
            account_balance = account_info.balance

        # Calculate dollar risk per trade
        dollar_risk = account_balance * risk_percentage

        # Get the symbol info
        symbol = self.symbols[0]
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Failed to get symbol info for {symbol}, error code =", mt5.last_error())
            mt5.shutdown()
            return

        # Retrieve OHLC data for volatility calculation
        ohlc = self.data[symbol]
        if ohlc is None or len(ohlc) < 24 * self.mid_length:
            print(f"Not enough data to compute daily volatility for {symbol}.")
            return

        # Calculate daily volatility (standard deviation of daily returns)
        returns = np.log(ohlc['close'] / ohlc['close'].shift(1)).dropna()
        daily_vol = returns.rolling(window=24 * self.mid_length).std().iloc[-1]

        if np.isnan(daily_vol) or daily_vol == 0:
            print(f"Invalid daily volatility computed for {symbol}.")
            return

        # Calculate position size based on volatility
        price = mt5.symbol_info_tick(symbol).ask
        contract_size = symbol_info.trade_contract_size

        position_size = dollar_risk / (daily_vol * price * contract_size)

        # Apply risk factor scaling
        adjusted_lots = round(self.risk_factor * position_size, 2)

        print(f"Adjusted Position Size for {symbol}: {adjusted_lots} lots based on daily volatility: {daily_vol:.4f}")

        return adjusted_lots

    def check_conditions(self):
        volume = self.position_sizing_vol_adjusted()
        mean_entry_price, num_positions = self.get_positions_info()
        mean_entry_price = round(mean_entry_price, 4)
        if self.state == 0:
            pass
        elif self.state == 1:
            self.execute(symbol=self.symbols[0], volume=volume, short=False)
            # Send a message when an order is entered
            self.send_telegram_message(
                f"Entered BUY order for {self.symbols[0]} with volume: {volume} et pelo sa achete! Mean Entry Price: {mean_entry_price}, Number of Positions: {num_positions}")
        elif self.state == -1:
            self.execute(symbol=self.symbols[0], volume=volume, short=True)
            # Send a message when an order is entered
            self.send_telegram_message(
                f"Entered SELL order for {self.symbols[0]} with volume: {volume}et pelo ca vend: Mean Entry Price: {mean_entry_price}, Number of Positions: {num_positions}")
        elif self.state == -2:
            self.close_all_positions()
            # Send a message when positions are closed
            self.send_telegram_message(f"Closed all positions for {self.symbols[0]}")

    def fit(self):

        # self.model = pickle.load(open("xgb_metaga.pkl", "rb"))
        # self.indicators_mean = pickle.load(open("hist_ind_means.pkl", "rb"))
        # self.indicators_std = pickle.load(open("hist_ind_stds.pkl", "rb"))

        # Define the UTC timezone
        utc = pytz.timezone('UTC')
        # Get the current time in UTC
        end_time = datetime.now(utc)
        start_time = end_time - timedelta(days=self.mid_length)
        # Set the time components to 0 (midnight) and maintain the timezone
        end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(utc)
        start_time = start_time.astimezone(utc)

        # Pulling last days of data
        self.loadData(start_time, end_time)
        data = self.data[self.symbols[0]]
        returns = data.loc[:, 'close'].apply(np.log) - data.loc[:, 'open'].apply(np.log)

        # Compute rolling next returns series
        T = returns.shape[0]
        next_five_returns = [np.sum(returns[i + 1: i + day_length+1]) for i in range(T)]
        next_five_returns = pd.Series(next_five_returns, index=returns.index)

        # Indicators
        indicators = self.retrieve_indicators(ohlc_df=data)

        # Retrieve history
        hist_indicators = indicators[:24*day_length]
        hist_next_five_returns = next_five_returns.loc[hist_indicators.index]
        indicators = indicators.loc[indicators.index.difference(hist_indicators.index)]
        next_five_returns = next_five_returns.loc[indicators.index]

        # Demean from history
        indicators = (indicators - hist_indicators.mean()) / hist_indicators.std()
        next_five_returns = (next_five_returns - hist_next_five_returns.mean()) / hist_next_five_returns.std()

        # Transform to dummy
        dummy_extremes_indicators = abs(indicators) > 1.0
        indicators = indicators[dummy_extremes_indicators.sum(axis=1) > 18]
        dummy_extremes_next_five_returns = next_five_returns.loc[indicators.index].apply(assign_cat)

        X, y = indicators.ffill(), dummy_extremes_next_five_returns

        xgb_dummy = xgb.XGBClassifier().fit(X, y)
        print(f"{self.tag}::: XGBoost Model trained from {X.index[0]} to {X.index[-1]}.")
        # Add line to log file to signal training completion
        self.logger.info(f"XGBoost Model trained from {X.index[0]} to {X.index[-1]}.")
        
        # Save model, 1st and 2nd indicator moments
        self.model = xgb_dummy
        self.indicators_mean = hist_indicators.mean()
        self.indicators_std = hist_indicators.std()
        # Save the 90% quantile of the closes
        self.quantile = data['close'].quantile(0.9)

        print(f"{self.tag}::: XGBoost Model and first/second moments saved.")

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
        closes = ohlc.loc[:, 'close']
        returns = closes.apply(np.log).diff().dropna()
        
        low_length = self.low_length
        mid_length = self.mid_length
        high_length = self.high_length

        # Log-Returns EMAs
        emas = ewma_sets(returns.values)
        emas = pd.DataFrame(emas, index=returns.index)

        # Rolling Realized Volatilities
        vols_rolling_session = returns.rolling(mid_length).apply(lambda x: np.sum(np.square(x.values))).rename(
            "vol_session")
        vols_rolling_hour = returns.rolling(mid_length).apply(lambda x: np.sum(np.square(x.values))).rename("vol_hour")
        vols_rolling_daily = returns.rolling(high_length).apply(lambda x: np.sum(np.square(x.values))).rename("vol_daily")
        vols_rollings = pd.concat([vols_rolling_hour, vols_rolling_session, vols_rolling_daily], axis=1)
        print(f"{self.tag}::: Computed rolling volatilies")

        # Rolling Skewness
        skewness_rolling_session = returns.rolling(mid_length).apply(skewness_nb, engine='numba', raw=True).rename(
            "skew_session")
        skewness_rolling_hour = returns.rolling(mid_length).apply(skewness_nb, engine='numba', raw=True).rename("skew_hour")
        skewness_rolling_daily = returns.rolling(high_length).apply(skewness_nb, engine='numba', raw=True).rename(
            "skew_daily")
        skewness_rollings = pd.concat([skewness_rolling_hour, skewness_rolling_session, skewness_rolling_daily], axis=1)
        print(f"{self.tag}::: Computed rolling skewness")

        # Rolling Kurtosis
        kurtosis_rolling_session = returns.rolling(mid_length).apply(kurtosis_nb, engine='numba', raw=True).rename(
            "kurt_session")
        kurtosis_rolling_hour = returns.rolling(mid_length).apply(kurtosis_nb, engine='numba', raw=True).rename("kurt_hour")
        kurtosis_rolling_daily = returns.rolling(high_length).apply(kurtosis_nb, engine='numba', raw=True).rename(
            "kurt_daily")
        kurtosis_rollings = pd.concat([kurtosis_rolling_hour, kurtosis_rolling_session, kurtosis_rolling_daily], axis=1)
        print(f"{self.tag}::: Computed rolling kurtosis")

        # Rolling Number of Mean Crossings
        crossings_rolling_session = closes.rolling(mid_length).apply(retrieve_number_of_crossings_nb, engine='numba',
                                                                 raw=True).rename("crossings_session")
        crossings_rolling_hour = closes.rolling(mid_length).apply(retrieve_number_of_crossings_nb, engine='numba',
                                                          raw=True).rename("crossings_hour")
        crossings_rolling_daily = closes.rolling(high_length).apply(retrieve_number_of_crossings_nb, engine='numba',
                                                                raw=True).rename("crossings_daily")
        crossings_rollings = pd.concat([crossings_rolling_hour, crossings_rolling_session, crossings_rolling_daily],
                                       axis=1)
        print(f"{self.tag}::: Computed rolling mean crossings")

        # Trend T-statistic
        tval_rolling_session = closes.rolling(mid_length).apply(ols_tval_nb, engine='numba', raw=True).rename(
            "tval_session")
        tval_rolling_hour = closes.rolling(mid_length).apply(ols_tval_nb, engine='numba', raw=True).rename("tval_hour")
        tval_rolling_daily = closes.rolling(high_length).apply(ols_tval_nb, engine='numba', raw=True).rename("tval_daily")
        tval_rollings = pd.concat([tval_rolling_hour, tval_rolling_session, tval_rolling_daily], axis=1)
        print(f"{self.tag}::: Computed rolling OLS t-values")

        # Technical Indicators
        rsi_compute(ohlc)
        pivot_points_compute(ohlc)
        bollinger_bands_compute(ohlc)
        ewa_compute(ohlc)
        macd_compute1(ohlc)
        ohlc = ohlc.iloc[19:]
        technical_indicators = ohlc.drop(axis=1, columns=['open', 'high', 'low', 'close', 'spread', 'real_volume'])
        print(f"{self.tag}::: Computed technical indicators")

        # Merge Features
        indicators = [emas, vols_rollings, skewness_rollings, kurtosis_rollings, tval_rollings, crossings_rollings,
                      technical_indicators]
        indicators = pd.concat(indicators, axis=1).iloc[1:]
        indicators.index = pd.to_datetime(indicators.index)
        indicators.columns = indicators.columns.astype(str)
        print(f"{self.tag}::: Merged indicators")

        return indicators

def assign_cat(val):
    if val < 0.:
        return 0
    else:
        return 1
