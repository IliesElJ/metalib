from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
import pytz as pytz
import os
from typing import Dict, Optional, Tuple

from metalib.indicators import *
from metalib.metastrategy import MetaStrategy


class MetaHAR(MetaStrategy):

    def __init__(   self, 
                    symbols,
                    predicted_symbol,
                    timeframe, 
                    tag, 
                    active_hours, 
                    short_factor=60,
                    long_factor=8*60,
                ):
        super().__init__(symbols, timeframe, tag, active_hours)
        
        if not (short_factor < long_factor):
            raise ValueError("Length parameters should be ordered.")

        if not (predicted_symbol in symbols):
            raise ValueError("Predicted symbol should be in the list of symbols to be used for training the model.")

        self.indicators     = None
        self.model          = None
        self.short_factor   = short_factor
        self.long_factor    = long_factor
        self.predicted_symbol = predicted_symbol
        self.telegram       = True
        self.logger         = logging.getLogger(__name__)

    TRAINING_PERIOD_DAYS = 66
    UTC_TIMEZONE = 'UTC'
    RESAMPLE_FREQUENCY = "1t"

    def signals(self) -> None:
        """Process market signals and update strategy state."""
        # Process close prices
        closes = self._get_processed_closes()

        # Get and process indicators
        indicators = self.retrieve_indicators(closes)
        self.indicators = indicators
        recent_indicators = indicators.tail(3)

        # Make predictions
        volatility_predictions = self.model.predict(recent_indicators)


        # Update strategy state
        self._update_strategy_state(recent_indicators, volatility_predictions)

        # Process and log predictions
        self._process_predictions()

    def _get_processed_closes(self) -> pd.DataFrame:
        """Transform raw close prices into resampled DataFrame."""
        closes_dict = dict(map(lambda kv: (kv[0], kv[1].close), self.data.items()))
        return pd.DataFrame(closes_dict).resample(
            "1t",
            closed="right",
            label="right"
        ).last().dropna()

    def _update_strategy_state(self, indicators: pd.DataFrame, predictions: np.ndarray) -> None:
        """Update internal strategy state with new data."""
        print(f"{self.tag}::: Open positions for strategy: {self.tag}: {self.are_positions_with_tag_open()}")

        self.state = 0
        self.signals_data = indicators.iloc[[-1]]
        self.predicted_vol_diff = predictions
        self.realized_previous_diff = indicators[f"short_scale_std_{self.predicted_symbol}"].diff().dropna().iloc[:1]
        self.timestamp = indicators.index[-1]

    def _process_predictions(self) -> None:
        """Process and log prediction results."""
        path = f"../indicators/{self.tag}_signals_log.csv"
        file_path = path.format(tag=self.tag)
        previous_prediction = self._get_previous_prediction(file_path)

        if previous_prediction:
            prediction_diff = float(self.realized_previous_diff) - previous_prediction
            print(
                f"Previous Predicted: {previous_prediction}, "
                f"Current Realized: {self.realized_previous_diff}, "
                f"Difference: {prediction_diff}"
            )
        else:
            prediction_diff = None

        self._write_prediction_data(prediction_diff)

    def _get_previous_prediction(self, file_path: str) -> Optional[float]:
        """Retrieve the previous prediction from the log file."""
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            return None

        previous_data = pd.read_csv(file_path)
        if previous_data.empty:
            return None

        return float(previous_data.iloc[-1].get("prediction", None))

    def _write_prediction_data(self, prediction_diff: Optional[float]) -> None:
        """Write prediction data to the log file."""
        data_to_write = {
            "timestamp": [self.timestamp],
            "last_indicators": [self.signals_data.iloc[-1].to_dict()],
            "prediction": [float(self.predicted_vol_diff[-1])],
            "realized_diff": [float(self.realized_previous_diff)],
            "prediction_realized_difference": [prediction_diff],
            "same_sign": [int(np.sign(self.predicted_vol_diff[-1]) == np.sign(self.realized_previous_diff))]
        }

        file_path = f"../indicators/{self.tag}_signals_log.csv"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Check if the row already exists
        if os.path.isfile(file_path):
            existing_data = pd.read_csv(file_path)
            if self.timestamp in existing_data["timestamp"].values:
                return  # Skip appending if the row already exists

        # Write data to the file
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            pd.DataFrame(data_to_write).to_csv(file_path, index=False)
        else:
            pd.DataFrame(data_to_write).to_csv(file_path, mode='a', header=False, index=False)


    def check_conditions(self):
        return True

    def get_training_period(self) -> Tuple[datetime, datetime]:
        """Calculate start and end times for the training period in UTC."""
        utc = pytz.timezone(self.UTC_TIMEZONE)
        end_time = datetime.now(utc)
        start_time = end_time - timedelta(days=self.TRAINING_PERIOD_DAYS)

        # Normalize end time to midnight
        normalized_end = end_time.replace(
            hour=0, minute=0, second=0, microsecond=0
        ).astimezone(utc)
        normalized_start = start_time.astimezone(utc)

        return normalized_start, normalized_end

    def prepare_training_data(self, raw_data: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable for model training."""
        # Extract closing prices
        closes = pd.DataFrame({
            symbol: data.close for symbol, data in raw_data.items()
        })

        # Resample and clean data
        resampled_closes = closes.resample(
            self.RESAMPLE_FREQUENCY,
            closed="right",
            label="right"
        ).last().dropna()

        # Generate features and target
        indicators = self.retrieve_indicators(resampled_closes)
        target = (
            indicators[f"short_scale_std_{self.predicted_symbol}"]
            .shift(-1)
            .diff()
            .dropna()
        )

        # Align features with target
        features = indicators.loc[
                   indicators.index.intersection(target.index), :
                   ]
        target = target.loc[target.index]

        return features, target

    def fit(self) -> None:
        """
        Train the LASSO model using historical data.
        Updates self.model and self.best_alpha with trained model.
        """
        # Get training period
        start_time, end_time = self.get_training_period()

        # Load and prepare data
        self.loadData(start_time, end_time)

        features, target = self.prepare_training_data(self.data)

        # Train model
        self.model, self.best_alpha = build_lasso_cv(features, target)

        # Log training completion
        training_period = f"from {features.index[0]} to {features.index[-1]}"
        print(f"{self.tag}::: Lasso Model trained {training_period}.")
        self.logger.info(f"Lasso Model trained {training_period}.")
        print(f"{self.tag}::: Lasso Model and coefficients saved.")


    def position_sizing(self, percentage, symbol, account_balance=None):
        return 0

    def retrieve_indicators(self, close_df):
        # Constants
        COLUMN_PREFIXES = {
            # 'long_corr_': 'long_factor_log_corr',
            # 'short_corr_': 'short_factor_log_corr',
            'long_scale_std_': 'long_scale_std',
            'short_scale_std_': 'short_scale_std',
            'trend_ewm_short_factor_': 'trend_ewm_short_factor',
            'trend_ewm_long_factor_': 'trend_ewm_long_factor',
            'sq_ret_ewm_short_factor_': 'sq_ret_ewm_short_factor',
            'sq_ret_ewm_long_factor_': 'sq_ret_ewm_long_factor',
        }

        def calculate_rolling_correlations(price_data):
            fx_rolling_log_corr = price_data.rolling(
                self.short_factor,
                method="table"
            ).apply( corr_eigenvalues, engine='numba', raw=True).dropna()

            long_factor_log_corr = fx_rolling_log_corr.rolling(self.long_factor).apply(
                lambda x: np.median(x), engine='numba', raw=True
            )

            short_factor_log_corr = (fx_rolling_log_corr - long_factor_log_corr).rolling(
                self.short_factor
            ).apply(lambda x: np.median(x), engine='numba', raw=True)

            return long_factor_log_corr, short_factor_log_corr

        def calculate_scale_std(price_data):
            long_scale = price_data.rolling(self.long_factor).apply(
                log_var, engine='numba', raw=True
            )
            short_scale = price_data.rolling(self.short_factor).apply(
                log_var, engine='numba', raw=True
            )
            return long_scale, short_scale

        def calculate_ewm_indicators(returns):
            trend_short = returns.ewm(halflife=self.short_factor).mean()
            trend_long = returns.ewm(halflife=self.long_factor).mean()

            squared_returns = returns.apply(lambda x: x ** 2, engine="numba", raw=True)
            sq_ret_short = squared_returns.ewm(halflife=self.short_factor).mean()
            sq_ret_long = squared_returns.ewm(halflife=self.long_factor).mean()

            return trend_short, trend_long, sq_ret_short, sq_ret_long

        def calculate_asym_std(log_returns):
            downside_squared_returns = log_returns.where(log_returns < 0, 0).apply(lambda x: x ** 2, engine="numba",
                                                                                   raw=True)
            upside_squared_returns = log_returns.where(log_returns > 0, 0).apply(lambda x: x ** 2, engine="numba",
                                                                                 raw=True)

            # EWM of realized semi-variances (HAR-RSV style)
            downside_vol_ewm_short = downside_squared_returns.ewm(halflife=self.short_factor).mean()
            upside_vol_ewm_short = upside_squared_returns.ewm(halflife=self.short_factor).mean()

            downside_vol_ewm_long = downside_squared_returns.ewm(halflife=self.long_factor).mean()
            upside_vol_ewm_long = upside_squared_returns.ewm(halflife=self.long_factor).mean()

            return downside_vol_ewm_short, downside_vol_ewm_long, upside_vol_ewm_short, upside_vol_ewm_long



        # Calculate base metrics
        price_series = close_df
        log_returns = price_series.apply(np.log).diff().dropna()

        # Calculate all indicators
        # long_corr, short_corr = calculate_rolling_correlations(price_series)
        long_std, short_std = calculate_scale_std(price_series)
        trend_short, trend_long, sq_ret_short, sq_ret_long = calculate_ewm_indicators(log_returns)
        downside_short, downside_long, upside_short, upside_long = calculate_asym_std(log_returns)

        # Validate unique indices
        for name, df in [
            # ("long_factor_log_corr", long_corr),
            # ("short_factor_log_corr", short_corr),
            ("long_scale_std", long_std),
            ("short_scale_std", short_std),
            ("long_scale_upside", upside_long),
            ("short_scale_upside", upside_short),
            ("long_scale_downside", downside_long),
            ("short_scale_downside", downside_short),
        ]:
            if not df.index.is_unique:
                raise IndexError(f"Duplicates found in {name}")

        # Combine all indicators
        indicators = pd.concat([
            # long_corr.add_prefix("long_corr_"),
            # short_corr.add_prefix("short_corr_"),
            long_std.add_prefix("long_scale_std_").apply(np.log),
            short_std.add_prefix("short_scale_std_").apply(np.log),
            trend_short.add_prefix("trend_ewm_short_factor_"),
            trend_long.add_prefix("trend_ewm_long_factor_"),
            sq_ret_short.add_prefix("sq_ret_ewm_short_factor_"),
            sq_ret_long.add_prefix("sq_ret_ewm_long_factor_"),
            downside_short.add_prefix("downside_ewm_short_factor_"),
            downside_long.add_prefix("downside_ewm_long_factor_"),
            upside_short.add_prefix("upside_ewm_short_factor_"),
            upside_long.add_prefix("upside_ewm_long_factor_")
        ], axis=1).dropna()
        indicators.loc[:, 'const'] = 1

        # Resample and clean data
        resampled_indicators = indicators.resample(
            f"{self.short_factor}t",
            closed="right",
            label="right"
        ).last().replace([np.inf, -np.inf], np.nan).dropna()

        resampled_indicators.index = pd.to_datetime(resampled_indicators.index)
        resampled_indicators.columns = resampled_indicators.columns.astype(str)

        print(f"{self.tag}::: Merged indicators")
        return resampled_indicators
