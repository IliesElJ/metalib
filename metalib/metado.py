from datetime import datetime, timedelta
import MetaTrader5 as mt5
import numpy as np
import pytz as pytz
import vectorbt as vbt
import matplotlib.pyplot as plt
import xgboost as xgb

from metalib.indicators import *
from metalib.metastrategy import MetaStrategy
from dateutil.relativedelta import relativedelta, FR


class MetaDO(MetaStrategy):

    def __init__(self, symbols, timeframe, tag, active_hours, lookback, risk_factor=1):
        super().__init__(symbols, timeframe, tag, active_hours)
        self.indicators = None
        self.quantile = None
        self.model = None
        self.indicators_std = None
        self.indicators_mean = None
        self.state = None
        self.risk_factor = risk_factor
        self.telegram = True
        self.lookback = lookback
        self.stop_loss = 5
        self.columns_to_drop = [
            "next_returns",
            "open",
            "high",
            "low",
            "close",
            "lower",
            "higher",
            "real_volume",
            "tick_volume",
            "spread",
        ]

    def signals(self):
        ohlc = self.data[self.symbols[0]]
        indicators = (
            self.retrieve_indicators(ohlc)
            .drop(self.columns_to_drop, axis=1, errors="ignore")
            .astype(float)
        )

        self.indicators = indicators

        # Demean Indicators
        indicators = (indicators - self.indicators_mean) / self.indicators_std
        open_condition = (
            (indicators[["rolling_high_breaks", "rolling_low_breaks"]] > 0)
            .astype(int)
            .sum(axis=1)
            .iloc[-1]
            .squeeze()
        )
        last_ind_arr = indicators.tail(3).values

        # Check if the last indicators line contains NaN
        if np.isnan(last_ind_arr[-1, :]).any():
            print(f"{self.tag}::: NaN values found in the last indicators line.")
            return

        y_hat = self.model.predict_proba(last_ind_arr)[:, 1]
        y_hat = pd.Series(y_hat)

        eps = self.eps_signal

        if (
            y_hat.vbt.crossed_above(0.5 + eps).iloc[-1]
            and not self.are_positions_with_tag_open()
            and open_condition > 0
        ):
            self.state = 1
        elif (
            y_hat.vbt.crossed_below(0.5 - eps).iloc[-1]
            and not self.are_positions_with_tag_open()
            and open_condition > 0
        ):
            self.state = -1
        else:
            if self.are_positions_with_tag_open():
                self.check_close_conditions()
            else:
                self.state = 0
        # Show some plots of price and signals
        indicators.iloc[-200:].plot(figsize=(22, 8), title=f"{self.tag} Indicators")
        plt.ylim(-2, 2)
        plt.show()

        print(
            f"{self.tag}::: Open positions for strategy: {self.tag}: {self.are_positions_with_tag_open()}"
        )
        print(f"{self.tag}::: Last 3 predicted probabilities: {y_hat.to_list()}")

        signal_line = indicators.iloc[[-1]]
        signal_line.loc[:, "predicted_proba"] = y_hat.iloc[-1]

        self.signals_data = signal_line

    def check_close_conditions(self):
        # Recover open positions that have the tag
        positions = mt5.positions_get(symbol=self.symbols[0])

        # Check if positions is None or an empty list
        if not positions:
            print(f"{self.tag}::: No open positions found for symbol {self.symbols[0]}")
            return

        # Debug print for the positions
        print(f"{self.tag}::: Positions: {positions}")

        # Assuming the first position is the one we are checking
        position = positions[0]

        # Debug print for the specific position
        print(f"{self.tag}::: Position to check: {position}")

        # Ensure position.time is accessed correctly
        open_time = pd.to_datetime(position.time, unit="s", utc=True)
        current_time = pd.to_datetime(datetime.now() + timedelta(hours=1), utc=True)
        time_diff = (
            current_time - open_time
        ).total_seconds() / 60  # Convert seconds to minutes

        # Debug print for the time difference
        print(f"{self.tag}::: Time difference in minutes: {time_diff}")

        # Close the position if it has been open for more than lookback minutes
        if time_diff > self.lookback and position.profit < 0:
            self.state = -2
            print(
                f"{self.tag}::: Position has been open for more than {self.lookback} minutes, setting state to -2."
            )
            return

        if time_diff > 3 * self.lookback and position.profit > 0:
            self.state = -2
            print(
                f"{self.tag}::: Position has been open for more than {self.lookback} minutes, setting state to -2."
            )
            return

        if position.profit < -self.stop_loss:
            self.state = -2
            print(f"{self.tag}::: Position is in loss, setting state to -2.")
            return

    def check_conditions(self):
        volume = self.position_sizing(0.5, self.symbols[0])
        mean_entry_price, num_positions = self.get_positions_info()
        mean_entry_price = round(mean_entry_price, 4)
        if self.state == 0:
            pass
        elif self.state == 1:
            self.execute(symbol=self.symbols[0], short=False)
            # Send a message when an order is entered
            self.send_telegram_message(
                f"Entered BUY order for {self.symbols[0]} with volume: {volume} et pelo sa achete! Mean Entry Price: {mean_entry_price}, Number of Positions: {num_positions}"
            )
        elif self.state == -1:
            self.execute(symbol=self.symbols[0], short=True)
            # Send a message when an order is entered
            self.send_telegram_message(
                f"Entered SELL order for {self.symbols[0]} with volume: {volume}et pelo ca vend: Mean Entry Price: {mean_entry_price}, Number of Positions: {num_positions}"
            )
        elif self.state == -2:
            self.close_all_positions()
            # Send a message when positions are closed
            self.send_telegram_message(f"Closed all positions for {self.symbols[0]}")

        self.state = 0

    def fit(self):
        # Define the UTC timezone
        utc = pytz.timezone("UTC")
        # Get the current time in UTC
        current_time = datetime.now(utc)
        # Set the end_time to the last Friday
        end_time = current_time + relativedelta(weekday=FR(-1))
        start_time = end_time - timedelta(days=30)
        # Set the time components to 0 (midnight) and maintain the timezone
        end_time = end_time.replace(
            hour=0, minute=0, second=0, microsecond=0
        ).astimezone(utc)
        start_time = start_time.astimezone(utc)

        # Pulling last days of data
        self.loadData(start_time, end_time)
        data = self.data[self.symbols[0]]

        # Delete lines that contain values outside 3*IQR range for each column
        indicators = self.retrieve_indicators(ohlc_df=data).dropna()
        indicators = self.remove_outliers(indicators)

        next_five_returns = indicators.loc[:, "next_returns"]

        insample_df = indicators[
            indicators[["rolling_high_breaks", "rolling_low_breaks"]].sum(axis=1) > 0
        ]
        y_insample = (
            next_five_returns.apply(lambda x: x > 0).astype(int).loc[insample_df.index]
        )
        X_insample = insample_df.drop(
            self.columns_to_drop, axis=1, errors="ignore"
        ).astype(float)

        # Normalize the data
        mean = X_insample.mean()
        std = X_insample.std()

        (
            mean["broke_higher"],
            mean["broke_lower"],
            mean["rolling_high_breaks"],
            mean["rolling_low_breaks"],
        ) = (0, 0, 0, 0)
        (
            std["broke_higher"],
            std["broke_lower"],
            std["rolling_high_breaks"],
            std["rolling_low_breaks"],
        ) = (1, 1, 1, 1)

        X_insample = (X_insample - mean) / std

        # Save the mean and standard deviation

        print(f"{self.tag}::: Insample shape: {X_insample.shape}")

        self.is_data = X_insample
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X_insample, y_insample)

        print(
            f"{self.tag}::: XGBoost Model trained from {insample_df.index[0]} to {insample_df.index[-1]}."
        )

        # Save the model and scaler
        self.model = xgb_model
        self.indicators_mean = mean
        self.indicators_std = std
        self.eps_signal = (
            (pd.Series(xgb_model.predict_proba(X_insample)[:, 1]) - 0.5)
            .abs()
            .quantile(0.9)
        )
        print(f"{self.tag}::: XGBoost Model and scaler parameters saved.")

    def fit_logit(self):
        # Define the UTC timezone
        utc = pytz.timezone("UTC")
        # Get the current time in UTC
        current_time = datetime.now(utc)
        # Set the end_time to the last Friday
        end_time = current_time + relativedelta(weekday=FR(-1))
        start_time = end_time - timedelta(days=30)
        # Set the time components to 0 (midnight) and maintain the timezone
        end_time = end_time.replace(
            hour=0, minute=0, second=0, microsecond=0
        ).astimezone(utc)
        start_time = start_time.astimezone(utc)

        # Pulling last days of data
        self.loadData(start_time, end_time)
        data = self.data[self.symbols[0]]

        # Delete lines that containes values outside 3*IQR range for each column
        indicators = self.retrieve_indicators(ohlc_df=data).dropna()
        indicators = self.remove_outliers(indicators)

        next_five_returns = indicators.loc[:, "next_returns"]

        insample_df = indicators[
            indicators[["rolling_high_breaks", "rolling_low_breaks"]].sum(axis=1) > 0
        ]
        y_insample = (
            next_five_returns.apply(lambda x: x > 0).astype(int).loc[insample_df.index]
        )
        X_insample = insample_df.drop(
            self.columns_to_drop, axis=1, errors="ignore"
        ).astype(float)
        mean = X_insample.mean()
        std = X_insample.std()

        (
            mean["broke_higher"],
            mean["broke_lower"],
            mean["rolling_high_breaks"],
            mean["rolling_low_breaks"],
        ) = (0, 0, 0, 0)
        (
            std["broke_higher"],
            std["broke_lower"],
            std["rolling_high_breaks"],
            std["rolling_low_breaks"],
        ) = (1, 1, 1, 1)

        # Mean of categorical variable is 0 and std is 1
        X_insample = (X_insample - mean) / std

        print(f"{self.tag}::: Insample shape: {X_insample.shape}")
        self.is_data = X_insample
        logit_model = sm.Logit(y_insample, X_insample)
        result = logit_model.fit()

        print(
            f"{self.tag}::: Logit Model trained from {X_insample.index[0]} to {X_insample.index[-1]}."
        )
        # Print the summary of the model
        print(result.summary())

        # Add line to log file to signal training completion
        # Save model, 1st and 2nd indicator moments
        self.model = result
        self.indicators_mean = mean
        self.indicators_std = std
        self.eps_signal = (result.predict(X_insample) - 0.5).abs().quantile(0.8)
        print(f"{self.tag}::: Logit Model and first/second moments saved.")

    def remove_outliers(self, df):
        # Compute the IQR for each column
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1

        # Determine the lower and upper bounds for each column
        lower_bound = Q1 - 4 * IQR
        upper_bound = Q3 + 4 * IQR

        # Create a mask for each column
        mask = pd.DataFrame(True, index=df.index, columns=df.columns)
        for column in df.columns:
            if df[column].nunique() > 10 * self.lookback:
                mask[column] = (df[column] >= lower_bound[column]) & (
                    df[column] <= upper_bound[column]
                )

        # Filter out rows that contain values outside these bounds for columns with more than 2 unique values
        df_filtered = df[mask.all(axis=1)]

        return df_filtered

    def create_ema_dummy_variables(self, df, periods=None):
        if periods is None:
            periods = [20, 50, 100, 200]

        ema_dict = {}
        for period in periods:
            ema_col = f"EMA_{period}"
            dummy_col = f"Above_{ema_col}"

            # Calculate the EMA
            ema_col_series = df["close"].ewm(span=period, adjust=False).mean()
            ema_dict[period] = ema_col_series
            # Create the dummy variable
            df[dummy_col] = np.where(df["close"] > ema_col_series, 1, 0)

        df["ema_dir"] = np.where(ema_dict[periods[-1]] > ema_dict[periods[0]], 1, 0)

        return df

    def retrieve_indicators(self, ohlc_df):

        ohlc = ohlc_df.copy()
        lookback = self.lookback

        closes = ohlc.loc[:, "close"]
        returns = closes.apply(np.log).diff().dropna()
        ohlc["lower"] = ohlc["close"].rolling(lookback).min().shift(1)
        ohlc["higher"] = ohlc["close"].rolling(lookback).max().shift(1)

        ohlc["broke_higher"] = ohlc["close"].vbt.crossed_above(ohlc["higher"])
        ohlc["broke_lower"] = ohlc["close"].vbt.crossed_below(ohlc["lower"])

        ohlc["rolling_high_breaks"] = ohlc["broke_higher"].rolling(10 * lookback).sum()
        ohlc["rolling_low_breaks"] = ohlc["broke_lower"].rolling(10 * lookback).sum()
        ohlc["session"] = ohlc.index.hour.map(get_session)

        closes = ohlc.loc[:, "close"]
        returns = closes.apply(np.log).diff().dropna()

        vols_rollings = (
            returns.rolling(10 * lookback)
            .apply(lambda x: np.sum(np.square(x.values)))
            .rename("vol_rolling")
            .apply(np.sqrt)
        )
        r_squared_rollings = (
            closes.rolling(10 * lookback)
            .apply(ols_tval_nb, engine="numba", raw=True)
            .rename("t_val_rolling")
        )
        half_life_rollings = (
            closes.rolling(10 * lookback)
            .apply(calculate_half_life_nb, engine="numba", raw=True)
            .rename("half_life_rolling")
        )
        ohlc = self.create_ema_dummy_variables(ohlc, periods=[5, 10, 50, 200])

        # Merge Features
        indicators = pd.concat(
            [
                ohlc,
                r_squared_rollings,
                half_life_rollings,
                vols_rollings,
                returns.rolling(lookback).sum().shift(lookback).rename("next_returns"),
            ],
            axis=1,
        ).apply(pd.to_numeric)

        indicators.index = pd.to_datetime(indicators.index)
        indicators.columns = indicators.columns.astype(str)
        print(f"{self.tag}::: Merged indicators")

        return indicators.astype(float)

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
            total_volume = sum(pos.volume for pos in filtered_positions)
            mean_entry_price = (
                sum(pos.price_open * pos.volume for pos in filtered_positions)
                / total_volume
            )
            num_positions = len(filtered_positions)
        else:
            mean_entry_price = 0
            num_positions = 0

        # Return the mean entry price and number of positions
        return mean_entry_price, num_positions


def assign_cat(val):
    if val < 0.0:
        return 0
    else:
        return 1
