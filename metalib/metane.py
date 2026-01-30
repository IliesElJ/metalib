from datetime import datetime, timedelta
import pandas as pd
import logging
import pytz as pytz

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, accuracy_score

from metalib.indicators import *
from metalib.metastrategy import MetaStrategy
import vectorbt as vbt


def _evaluate_random_forest_with_tuning(X, y, n_splits=4, feature_names=None):
    """
    Evaluate Random Forest with hyperparameter tuning on most important parameters.
    Focus on simplicity and robustness.
    """

    # Convert to arrays and handle feature names
    if hasattr(X, "values"):
        X_array = X.values
        if feature_names is None:
            feature_names = X.columns.tolist()
    else:
        X_array = X

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

    # Simplified parameter grid - focus on most impactful parameters
    param_grid = {
        "randomforestregressor__n_estimators": [100, 200],  # More trees = more stable
        "randomforestregressor__max_depth": [10, 15, None],  # Control overfitting
        "randomforestregressor__min_samples_leaf": [
            2,
            5,
        ],  # Prevent overfitting on leaves
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Base pipeline with robust defaults for other parameters
    base_model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            min_samples_split=10,  # Conservative default to prevent overfitting
            max_features="sqrt",  # Good default for most cases
            random_state=42,
            n_jobs=-1,
        ),
    )

    # Use GridSearchCV for full search since we have fewer parameters
    rf_search = GridSearchCV(
        base_model,
        param_grid,
        cv=tscv,
        scoring="r2",
        n_jobs=-1,
        verbose=1,  # Show progress
    )

    # Fit and find best hyperparameters
    rf_search.fit(X_array, y)

    # Get best model and make predictions
    best_model = rf_search.best_estimator_
    y_pred = best_model.predict(X_array)

    # Calculate metrics
    in_sample_r2 = r2_score(y, y_pred)
    sign_accuracy = accuracy_score(y > 0, y_pred > 0)

    # Get feature importance from the best model
    rf = best_model.named_steps["randomforestregressor"]
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    # Print results
    print("=" * 60)
    print("RANDOM FOREST WITH HYPERPARAMETER TUNING RESULTS")
    print("=" * 60)
    print(f"Best parameters: {rf_search.best_params_}")
    print(f"Best CV R2 score: {rf_search.best_score_:.4f}")
    print(f"In-sample R2: {in_sample_r2:.4f}")
    print(f"Sign prediction accuracy: {sign_accuracy:.4f}")

    print(f"\nFinal Model Configuration:")
    print(f"  n_estimators: {rf.n_estimators}")
    print(f"  max_depth: {rf.max_depth}")
    print(f"  min_samples_leaf: {rf.min_samples_leaf}")
    print(f"  min_samples_split: {rf.min_samples_split}")
    print(f"  max_features: {rf.max_features}")

    print(f"\nTop 5 Most Important Features:")
    print(feature_importance.head(5).to_string(index=False))

    return {
        "model": best_model,
        "best_params": rf_search.best_params_,
        "best_cv_score": rf_search.best_score_,
        "in_sample_r2": in_sample_r2,
        "sign_accuracy": sign_accuracy,
        "feature_importance": feature_importance,
        "cv_results": rf_search.cv_results_,
    }


class MetaNE(MetaStrategy):
    def __init__(
        self,
        symbols,
        timeframe,
        tag,
        active_hours,
        lookahead,
        size_position,
        rrr,
        long_threshold,
        tz="UTC",
        don_window=240,
        don_lag=60,
        ewma_ewm_span=15,
        time_ewm_span=30,
        ols_window=120,
    ):

        super().__init__(symbols, timeframe, tag, size_position, active_hours)
        self.indicators = None
        self.lookahead = lookahead
        self.rrr = rrr
        self.long_threshold = long_threshold
        self.tz = tz
        self.don_window = don_window
        self.don_lag = don_lag
        self.ewma_ewm_span = ewma_ewm_span
        self.time_ewm_span = time_ewm_span
        self.ols_window = ols_window
        self.fwd_returns_ser = pd.Series()
        self.logger = logging.getLogger(__name__)

    def retrieve_indicators(
        self,
        ohlc_df: pd.DataFrame,
        *,
        tz: str = "UTC",
        don_window: int = 240,
        don_lag: int = 60,
        ewma_ewm_span: int = 15,
        time_ewm_span: int = 30,
        ols_window: int = 120,
    ) -> pd.DataFrame:
        """
        Build indicator matrix from OHLC with:
          - Bollinger band dummies (smoothed)
          - EWMA set dummies (smoothed)
          - Rolling OLS t-stat transform
          - Time/session calendar features (smoothed)
        Assumes external helpers exist: bollinger_bands_compute, ewma_sets, ols_tval_nb.
        """

        close = ohlc_df["close"].reset_index(drop=True)
        timestamps = ohlc_df.index.to_series()

        # ----------------------- utils -----------------------
        def _ensure_utc(ts: pd.Series) -> pd.Series:
            ts = (
                pd.to_datetime(ts, utc=True)
                if ts.dt.tz is None
                else ts.dt.tz_convert("UTC")
            )
            if tz and tz.upper() != "UTC":
                # convert to UTC baseline first, then keep UTC for session logic
                # (we keep UTC for flags because your definitions are UTC-based)
                pass
            return ts

        def _compute_donchian_dummies(close_s: pd.Series, don_window) -> pd.DataFrame:
            donchian_high = close_s.rolling(don_window).max().shift(don_lag)
            donchian_low = close_s.rolling(don_window).min().shift(don_lag)

            out = pd.DataFrame(index=close_s.index)

            out["above_upper"] = (close_s > donchian_high).astype(float)
            out["below_lower"] = (close_s < donchian_low).astype(float)

            out["crossed_high"] = close_s.vbt.crossed_above(donchian_high)
            out["crossed_low"] = close_s.vbt.crossed_below(donchian_low)

            return out.ewm(ewma_ewm_span).mean()

        def _compute_ewma_dummies(close_s: pd.Series, ewma_ewm_span) -> pd.DataFrame:
            ew = pd.DataFrame(ewma_sets(close_s.values))  # returns levels (same length)
            # compare each EWMA series to current close (elementwise)
            ew = ew.apply(lambda col: (col > close_s.values).astype(float))
            return ew.ewm(ewma_ewm_span).mean()

        def _compute_ols_tval(close_s: pd.Series, ols_window) -> pd.Series:
            # rolling apply with numba-compiled ols_tval_nb on raw window arrays
            tvals = close_s.rolling(ols_window).apply(
                ols_tval_nb, engine="numba", raw=True
            )
            # squeeze to ~[0,1]
            return tvals.apply(lambda x: x / 200.0 + 0.5)

        def _session_time_features(ts_utc: pd.Series, time_ewm_span) -> pd.DataFrame:
            hours = ts_utc.dt.hour
            minutes = ts_utc.dt.minute
            day = ts_utc.dt.day
            weekday = ts_utc.dt.weekday
            tod = hours + minutes / 60.0

            tf = pd.DataFrame(index=ts_utc.index)
            # Sessions (UTC)
            tf["is_asia_session"] = tod.between(0, 9, inclusive="left")
            tf["is_london_session"] = tod.between(7, 16, inclusive="left")
            tf["is_ny_session"] = tod.between(13, 22, inclusive="left")
            tf["is_london_ny_overlap"] = tod.between(13, 16, inclusive="left")
            tf["is_fixing_hour"] = hours == 15  # 16:00 London / 15:00 UTC

            # Calendar flags
            tf["is_eom"] = ts_utc.dt.is_month_end
            tf["is_som"] = day <= 3
            tf["is_monday"] = weekday == 0
            tf["is_friday"] = weekday == 4
            tf["is_monday_asia"] = tf["is_monday"] & (tod < 7)

            # Week-of-month, time encodings
            tf["week_of_month"] = ((day - 1) // 7) + 1
            tf["timeofday_float"] = tod
            tf["sin_time"] = np.sin(2 * np.pi * tod / 24.0)
            tf["cos_time"] = np.cos(2 * np.pi * tod / 24.0)

            tf = tf.astype(float).reset_index(drop=True)
            return tf.ewm(time_ewm_span).mean()

        # -------------------- build blocks -------------------
        ts_utc = _ensure_utc(timestamps)
        bb_dummies = _compute_donchian_dummies(close, don_window)
        ewma_dummies = _compute_ewma_dummies(close, ewma_ewm_span)
        ols_tvals = _compute_ols_tval(close, ols_window).rename("ols_tval_transformed")
        time_feats = _session_time_features(ts_utc, time_ewm_span)

        # -------------------- concat & return ----------------
        indicators = pd.concat(
            [ewma_dummies, bb_dummies, ols_tvals, time_feats], axis=1
        )

        # Str column names
        indicators.columns = indicators.columns.astype(str)
        return indicators

    def signals(self):
        ohlc = self.data[self.symbols[0]]
        close = ohlc["close"].reset_index(drop=True)
        ema_fast = close.ewm(span=self.lookahead * 10).mean()

        indicators = self.retrieve_indicators(ohlc)[self.selected_features]
        del ohlc
        model = self.clb_result["model"]
        y_hat = model.predict(indicators)[-1] / self.historical_vol

        self.fwd_returns_ser = pd.concat([self.fwd_returns_ser, pd.Series([y_hat])])
        y_hat_smoothed = self.fwd_returns_ser.ewm(self.lookahead).mean().iloc[-1]

        print(f"Predicted returns: {y_hat}")
        print(f"Smoothed predicted returns: {y_hat_smoothed}")

        if np.isnan(y_hat_smoothed):
            y_hat_smoothed = 0.0

        mean_entry_price, num_positions = self.get_positions_info()

        if close.iloc[-1] < ema_fast.iloc[-1] and self.are_positions_with_tag_open(
            position_type="buy"
        ):
            self.state = -2
        elif close.iloc[-1] > ema_fast.iloc[-1] and self.are_positions_with_tag_open(
            position_type="sell"
        ):
            self.state = -2
        elif y_hat > self.long_threshold and not num_positions:
            self.state = 1
        elif y_hat < -self.long_threshold and not num_positions:
            self.state = -1
        else:
            self.state = 0

        print(
            f"{self.tag}::: Open positions for strategy: {self.tag}: {self.are_positions_with_tag_open()}"
        )
        print(f"{self.tag}::: Current signal: {self.state}")
        print(f"{self.tag}::: Predicted forward return: {y_hat}")
        print(f"{self.tag}::: Number of positions: {num_positions}")

        signal_line = indicators.iloc[-1]
        signal_line["predicted_fwd_return"] = y_hat

        self.signalData = signal_line

        return

    def check_conditions(self):
        mean_entry_price, num_positions = self.get_positions_info()
        mean_entry_price = round(mean_entry_price, 4)
        if self.state == 0:
            pass
        elif self.state == 1:
            self.execute(symbol=self.symbols[0], short=False)
            # Send a message when an order is entered
            self.send_telegram_message(
                f"Entered BUY order for {self.symbols[0]} with volume: {self.size_position} et pelo sa achete! Mean Entry Price: {mean_entry_price}, Number of Positions: {num_positions}"
            )
        elif self.state == -1:
            self.execute(symbol=self.symbols[0], short=True)
            # Send a message when an order is entered
            self.send_telegram_message(
                f"Entered SELL order for {self.symbols[0]} with volume: {self.size_position}et pelo ca vend: Mean Entry Price: {mean_entry_price}, Number of Positions: {num_positions}"
            )
        elif self.state == -2:
            self.close_all_positions()
            # Send a message when positions are closed
            self.send_telegram_message(f"Closed all positions for {self.symbols[0]}")

    def fit(self):
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
        returns = data.close.apply(np.log).diff()
        forward_returns = returns.rolling(self.lookahead).mean().shift(-self.lookahead)
        forward_returns = forward_returns.replace([np.inf, -np.inf, np.nan], 0).values

        self.historical_vol = returns.std() * np.sqrt(self.lookahead)

        indicators = self.retrieve_indicators(
            data,
            tz=self.tz,
            don_window=self.don_window,
            don_lag=self.don_lag,
            ewma_ewm_span=self.ewma_ewm_span,
            time_ewm_span=self.time_ewm_span,
            ols_window=self.ols_window,
        )

        selected_features = self._select_least_correlated_features(
            indicators, n_features=15
        )
        self.selected_features = selected_features
        print("Selected features:", selected_features)

        X_subset_train = indicators[selected_features].values
        rf_clb_result = _evaluate_random_forest_with_tuning(
            X_subset_train, forward_returns, n_splits=4, feature_names=selected_features
        )

        self.clb_result = rf_clb_result
        return

    def _select_least_correlated_features(
        self, df: pd.DataFrame, n_features: int = 10
    ) -> list:
        """
        Select n_features from df.columns such that average pairwise correlation
        among selected features is minimized (greedy algorithm).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of features.
        n_features : int, default 10
            Number of features to select.

        Returns
        -------
        list
            Names of the selected features.
        """
        corr = df.corr().abs()

        # Start with the feature that has the lowest average correlation to all others
        avg_corr = corr.mean()
        first = avg_corr.idxmin()
        selected = [first]
        remaining = set(corr.columns) - {first}

        while len(selected) < min(n_features, df.shape[1]):
            # For each remaining feature, compute its average correlation to already selected
            scores = {feat: corr.loc[feat, selected].mean() for feat in remaining}
            # Pick the feature with the lowest average correlation
            next_feat = min(scores, key=scores.get)
            selected.append(next_feat)
            remaining.remove(next_feat)

        return selected
