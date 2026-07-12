from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
import pytz as pytz
import os
from typing import Dict, Optional, Tuple

from metalib.indicators import *
from metalib.metastrategy import MetaStrategy

TIMEFRAME_MINUTES = {
    mt5.TIMEFRAME_M1: 1,
    mt5.TIMEFRAME_M5: 5,
    mt5.TIMEFRAME_M15: 15,
    mt5.TIMEFRAME_M30: 30,
    mt5.TIMEFRAME_H1: 60,
    mt5.TIMEFRAME_H4: 240,
}


class MetaHAR(MetaStrategy):
    """Univariate HAR-style volatility-change forecaster.

    Predicts the next-model-bar change in log realized variance of
    predicted_symbol, where one model bar = short_factor input bars
    (M15 x 16 = 4h with the default config). Realized variance is the variance
    of consecutive log returns over the short_factor window.

    Features are built from the predicted symbol ONLY, plus deterministic
    calendar dummies. Walk-forward validated 2026-07 (see
    notebooks/metahar_wf_backtest.py): cross-asset basket features reduced
    OOS accuracy on every symbol once calendar dummies were present, so the
    old multi-symbol basket was dropped. Predictions are clamped to the
    training target range — unclamped linear extrapolation blew up on the
    Apr-2025 XAUUSD vol shock (predicted log-vol changes of -11 vs realized
    +/-2).
    """

    TRAINING_PERIOD_DAYS = 66
    UTC_TIMEZONE = "UTC"

    def __init__(
        self,
        symbols,
        predicted_symbol,
        timeframe,
        tag,
        active_hours,
        short_factor=16,
        long_factor=256,
    ):
        if not (short_factor < long_factor):
            raise ValueError("Length parameters should be ordered.")

        if timeframe not in TIMEFRAME_MINUTES:
            raise ValueError("Unsupported timeframe for MetaHAR.")

        # Univariate: only the predicted symbol is loaded, regardless of the
        # symbols list in the config (kept for config compatibility).
        if list(symbols) != [predicted_symbol]:
            print(
                f"{tag}::: MetaHAR is univariate; ignoring extra symbols "
                f"{[s for s in symbols if s != predicted_symbol]}"
            )
        super().__init__([predicted_symbol], timeframe, tag, 0.0, active_hours)

        self.indicators = None
        self.model = None
        self.clip_bounds = None
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.predicted_symbol = predicted_symbol
        bar_minutes = TIMEFRAME_MINUTES[timeframe]
        self.resample_freq = f"{short_factor * bar_minutes}min"
        self.telegram = True
        self.logger = logging.getLogger(__name__)

    def signals(self) -> None:
        """Process market signals and update strategy state."""
        closes = self._get_processed_closes()

        indicators = self.retrieve_indicators(closes)
        self.indicators = indicators
        recent_indicators = indicators.tail(3)

        volatility_predictions = self.model.predict(recent_indicators)
        if self.clip_bounds is not None:
            volatility_predictions = np.clip(volatility_predictions, *self.clip_bounds)

        self._update_strategy_state(recent_indicators, volatility_predictions)
        self._process_predictions()

    def _get_processed_closes(self) -> pd.Series:
        """Regularize the predicted symbol's closes onto the input-bar grid."""
        bar_minutes = TIMEFRAME_MINUTES[self.timeframe]
        return (
            self.data[self.predicted_symbol]
            .close.resample(f"{bar_minutes}min", closed="right", label="right")
            .last()
            .dropna()
        )

    def _update_strategy_state(
        self, indicators: pd.DataFrame, predictions: np.ndarray
    ) -> None:
        """Update internal strategy state with new data."""
        print(
            f"{self.tag}::: Open positions for strategy: {self.tag}: {self.are_positions_with_tag_open()}"
        )

        self.state = 0
        self.signals_data = indicators.iloc[[-1]]
        self.predicted_vol_diff = predictions
        self.realized_previous_diff = float(
            indicators[f"short_scale_std_{self.predicted_symbol}"]
            .diff()
            .dropna()
            .iloc[0]
        )
        self.timestamp = indicators.index[-1]

    def _process_predictions(self) -> None:
        """
        Process prediction results and store them in signalData.

        This method:
        1. Retrieves previous predictions (if available)
        2. Calculates the difference between predicted and realized values
        3. Creates a Series with all prediction metrics and indicator values
        4. Stores the result in self.signalData for use by the strategy
        """
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

        # Create a dictionary with prediction data
        prediction_data = {
            "timestamp": self.timestamp,
            "prediction": float(self.predicted_vol_diff[-1]),
            "realized_diff": float(self.realized_previous_diff),
            "prediction_realized_difference": (
                prediction_diff if prediction_diff is not None else 0.0
            ),
            "same_sign": int(
                np.sign(self.predicted_vol_diff[-1])
                == np.sign(self.realized_previous_diff)
            ),
        }

        # Add all indicator values as individual fields
        for col, value in self.signals_data.iloc[-1].items():
            # Ensure values are simple types (int, float, bool, or string)
            if isinstance(value, (int, float, bool, str)):
                prediction_data[col] = value
            elif pd.isna(value):
                prediction_data[col] = 0.0
            else:
                # Convert complex types to float if possible, otherwise to string
                try:
                    prediction_data[col] = float(value)
                except (ValueError, TypeError):
                    prediction_data[col] = str(value)

        # Save to signalData
        self.signalData = pd.Series(prediction_data)

        # Publish to the shared prediction store; never let a DB hiccup kill
        # the strategy loop
        try:
            from metalib import volstore

            volstore.save_prediction(
                tag=self.tag,
                symbol=self.predicted_symbol,
                ts=self.timestamp,
                horizon_min=int(pd.Timedelta(self.resample_freq).total_seconds() // 60),
                prediction=float(self.predicted_vol_diff[-1]),
                realized_prev_diff=self.realized_previous_diff,
            )
        except Exception as e:
            print(f"{self.tag}::: volstore write failed: {e}")

    def _get_previous_prediction(self, file_path: str) -> Optional[float]:
        """
        Retrieve the previous prediction from the log file.
        This is used for comparison with current realized values.
        """
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            return None

        previous_data = pd.read_csv(file_path)
        if previous_data.empty:
            return None

        return float(previous_data.iloc[-1].get("prediction", None))

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

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable for model training."""
        closes = self._get_processed_closes()
        indicators = self.retrieve_indicators(closes)

        x = indicators[f"short_scale_std_{self.predicted_symbol}"]
        target = (x.shift(-1) - x).dropna()

        features = indicators.loc[indicators.index.intersection(target.index), :]
        target = target.loc[features.index]

        return features, target

    def fit(self) -> None:
        """
        Train the Lasso model using historical data.
        Updates self.model, self.best_alpha and self.clip_bounds.
        """
        start_time, end_time = self.get_training_period()

        self.loadData(start_time, end_time)
        features, target = self.prepare_training_data()

        # BLAS-free numba Lasso: sklearn crashes on GEMM in this env
        self.model, self.best_alpha = build_lasso_cv_nb(features, target)
        self.clip_bounds = (float(target.min()), float(target.max()))

        training_period = f"from {features.index[0]} to {features.index[-1]}"
        print(f"{self.tag}::: Lasso Model trained {training_period}.")
        self.logger.info(f"Lasso Model trained {training_period}.")

    def retrieve_indicators(self, closes: pd.Series) -> pd.DataFrame:
        """Univariate HAR features + calendar dummies, resampled to the model
        bar (short_factor input bars). Realized variance = variance of
        consecutive log returns (NOT the old log_var, which measured log-price
        dispersion within the window)."""
        sym = self.predicted_symbol
        ret = np.log(closes).diff().dropna()
        sq = ret**2
        down = ret.where(ret < 0, 0.0) ** 2
        up = ret.where(ret > 0, 0.0) ** 2

        indicators = pd.concat(
            {
                f"long_scale_std_{sym}": np.log(
                    ret.rolling(self.long_factor).var(ddof=0)
                ),
                f"short_scale_std_{sym}": np.log(
                    ret.rolling(self.short_factor).var(ddof=0)
                ),
                f"trend_ewm_short_factor_{sym}": ret.ewm(
                    halflife=self.short_factor
                ).mean(),
                f"trend_ewm_long_factor_{sym}": ret.ewm(
                    halflife=self.long_factor
                ).mean(),
                f"sq_ret_ewm_short_factor_{sym}": sq.ewm(
                    halflife=self.short_factor
                ).mean(),
                f"sq_ret_ewm_long_factor_{sym}": sq.ewm(
                    halflife=self.long_factor
                ).mean(),
                f"downside_ewm_short_factor_{sym}": down.ewm(
                    halflife=self.short_factor
                ).mean(),
                f"downside_ewm_long_factor_{sym}": down.ewm(
                    halflife=self.long_factor
                ).mean(),
                f"upside_ewm_short_factor_{sym}": up.ewm(
                    halflife=self.short_factor
                ).mean(),
                f"upside_ewm_long_factor_{sym}": up.ewm(
                    halflife=self.long_factor
                ).mean(),
            },
            axis=1,
        )
        indicators.loc[:, "const"] = 1

        resampled_indicators = (
            indicators.resample(self.resample_freq, closed="right", label="right")
            .last()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        dummies = seasonal_event_dummies(resampled_indicators.index, self.resample_freq)
        resampled_indicators = pd.concat([resampled_indicators, dummies], axis=1)

        resampled_indicators.index = pd.to_datetime(resampled_indicators.index)
        resampled_indicators.columns = resampled_indicators.columns.astype(str)

        print(f"{self.tag}::: Merged indicators")
        return resampled_indicators
