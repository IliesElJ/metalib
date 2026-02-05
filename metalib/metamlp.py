from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
import pytz
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from metalib.metastrategy import MetaStrategy


class MetaMLP(MetaStrategy):
    """
    Multi-Horizon MLP Mean Reversion Strategy.

    Trains 4 MLP regressors (one per forward-return horizon) on rolling
    z-score features and trades only when all 4 models agree on direction
    (convergence signal).
    """

    def __init__(
        self,
        symbols,
        timeframe,
        tag,
        size_position,
        active_hours=None,
        # feature engineering
        rolling_windows=(100, 500, 1000),
        thresholds=(-3.0, -2.0, -1.5, 1.5, 2.0, 3.0),
        # MLP horizons
        horizons=(1, 4, 12, 24),
        # MLP architecture
        hidden_layers=(32, 16),
        max_iter=500,
        # training
        fit_lookback_days=90,
        train_ratio=0.8,
        # risk
        risk_reward=2.0,
    ):
        super().__init__(symbols, timeframe, tag, size_position, active_hours)

        self.rolling_windows = tuple(rolling_windows)
        self.thresholds = tuple(thresholds)
        self.horizons = tuple(horizons)
        self.hidden_layers = tuple(hidden_layers)
        self.max_iter = int(max_iter)
        self.fit_lookback_days = int(fit_lookback_days)
        self.train_ratio = float(train_ratio)
        self.risk_reward = float(risk_reward)

        # Set by fit()
        self.mlp_models_ = {}
        self.scaler_ = None
        self.feature_cols_ = []

        # Set by signals()
        self.sma_target_ = None
        self.sl_ = None
        self.tp_ = None

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(self, log_close: pd.Series) -> pd.DataFrame:
        """
        Build z-score features from log-close prices.

        For each rolling window w:
          - continuous z-score: z_w = (log_close - SMA_w) / STD_w
          - binary tail indicators: 1(z_w > threshold) for each threshold

        Cross-scale diffs: z_w1 - z_w2 for every pair w1 < w2.
        """
        features = {}

        z_scores = {}
        for w in self.rolling_windows:
            sma = log_close.rolling(w).mean()
            std = log_close.rolling(w).std()
            z = (log_close - sma) / std
            z_scores[w] = z

            features[f"z_{w}"] = z
            for th in self.thresholds:
                features[f"z_{w}_gt_{th}"] = (z > th).astype(float)

        # Cross-scale diffs
        for w1, w2 in combinations(sorted(self.rolling_windows), 2):
            features[f"z_diff_{w1}_{w2}"] = z_scores[w1] - z_scores[w2]

        return pd.DataFrame(features, index=log_close.index)

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def fit(self):
        utc = pytz.timezone("UTC")
        end_time = datetime.now(utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(days=self.fit_lookback_days)

        self.loadData(start_time, end_time)
        data = self.data[self.symbols[0]]
        close = data["close"]
        log_close = np.log(close)

        # Features and targets
        feat_df = self._build_features(log_close)
        targets = {}
        for h in self.horizons:
            targets[h] = log_close.shift(-h) - log_close

        # Align: drop any row where features or any target is NaN
        valid = feat_df.notna().all(axis=1)
        for h in self.horizons:
            valid &= targets[h].notna()
        feat_df = feat_df.loc[valid]
        for h in self.horizons:
            targets[h] = targets[h].loc[valid]

        if len(feat_df) < 50:
            print(
                f"[{self.tag}] fit(): not enough data ({len(feat_df)} rows), skipping"
            )
            return

        self.feature_cols_ = list(feat_df.columns)

        # Time-series split
        split_idx = int(len(feat_df) * self.train_ratio)
        X_train = feat_df.iloc[:split_idx].values
        X_val = feat_df.iloc[split_idx:].values

        # Fit scaler on training set
        self.scaler_ = StandardScaler()
        X_train_scaled = self.scaler_.fit_transform(X_train)
        X_val_scaled = self.scaler_.transform(X_val)

        # Train one MLP per horizon
        self.mlp_models_ = {}
        for h in self.horizons:
            y_train = targets[h].iloc[:split_idx].values
            y_val = targets[h].iloc[split_idx:].values

            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                max_iter=self.max_iter,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
            )
            mlp.fit(X_train_scaled, y_train)
            self.mlp_models_[h] = mlp

            # Validation metrics
            y_pred_val = mlp.predict(X_val_scaled)
            r2 = r2_score(y_val, y_pred_val)
            dir_acc = accuracy_score(y_val > 0, y_pred_val > 0)
            print(
                f"[{self.tag}] fit() horizon={h}: "
                f"val R2={r2:.4f}, dir_acc={dir_acc:.4f}"
            )

        print(
            f"[{self.tag}] fit() complete: "
            f"{len(self.mlp_models_)} models trained on {split_idx} rows, "
            f"{len(self.feature_cols_)} features"
        )

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def signals(self):
        symbol = self.symbols[0]
        ohlc = self.data[symbol]
        close = ohlc["close"]
        current_price = close.iloc[-1]
        log_close = np.log(close)

        # Guard: models must be fitted
        if not self.mlp_models_ or self.scaler_ is None:
            print(f"[{self.tag}] signals(): models not fitted, state=0")
            self.state = 0
            return

        # Build features for the full series
        feat_df = self._build_features(log_close)
        last_row = feat_df[self.feature_cols_].iloc[[-1]]

        if last_row.isna().any(axis=1).iloc[0]:
            print(f"[{self.tag}] signals(): NaN in features, state=0")
            self.state = 0
            return

        last_scaled = self.scaler_.transform(last_row.values)

        # Predictions per horizon
        preds = {}
        signs = []
        for h in self.horizons:
            pred = self.mlp_models_[h].predict(last_scaled)[0]
            preds[h] = pred
            signs.append(np.sign(pred))

        # SMA target: average of rolling SMAs in price space
        sma_values = []
        for w in self.rolling_windows:
            sma_log = log_close.rolling(w).mean().iloc[-1]
            if not np.isnan(sma_log):
                sma_values.append(sma_log)
        sma_target = np.exp(np.mean(sma_values)) if sma_values else current_price
        self.sma_target_ = sma_target

        # Consensus
        all_positive = all(s > 0 for s in signs)
        all_negative = all(s < 0 for s in signs)

        has_long = self.are_positions_with_tag_open(position_type="buy")
        has_short = self.are_positions_with_tag_open(position_type="sell")
        has_position = has_long or has_short

        # State machine
        if all_positive and sma_target > current_price:
            if has_short:
                self.state = -2  # close short first
            elif not has_long:
                self.state = 1
            else:
                self.state = 0  # already long
        elif all_negative and sma_target < current_price:
            if has_long:
                self.state = -2  # close long first
            elif not has_short:
                self.state = -1
            else:
                self.state = 0  # already short
        elif has_position and not (all_positive or all_negative):
            self.state = -2  # consensus lost, close
        else:
            self.state = 0

        # Compute TP / SL
        self.tp_ = sma_target
        if self.state == 1:
            sl_distance = abs(sma_target - current_price) / self.risk_reward
            self.sl_ = current_price - sl_distance
        elif self.state == -1:
            sl_distance = abs(current_price - sma_target) / self.risk_reward
            self.sl_ = current_price + sl_distance
        else:
            self.sl_ = None

        # Diagnostics
        preds_str = ", ".join(f"h{h}={v:.6f}" for h, v in preds.items())
        print(
            f"[{self.tag}] signals(): {preds_str} | "
            f"consensus={'BUY' if all_positive else 'SELL' if all_negative else 'NONE'} | "
            f"sma_target={sma_target:.5f} | price={current_price:.5f} | "
            f"state={self.state}"
        )

        # Signal data for logging
        signal_line = pd.Series(
            {
                "timestamp": ohlc.index[-1],
                "price": current_price,
                "sma_target": sma_target,
                "state": self.state,
                **{f"pred_h{h}": v for h, v in preds.items()},
            }
        )
        self.signalData = signal_line

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def check_conditions(self):
        symbol = self.symbols[0]

        if self.state == 0:
            pass
        elif self.state == 1:
            self.execute(
                symbol=symbol,
                sl=round(self.sl_, 5) if self.sl_ else None,
                tp=round(self.tp_, 5) if self.tp_ else None,
                short=False,
            )
            self.send_telegram_message(
                f"[MetaMLP] BUY {symbol} | "
                f"TP={self.tp_:.5f} SL={self.sl_:.5f} | "
                f"tag={self.tag}"
            )
        elif self.state == -1:
            self.execute(
                symbol=symbol,
                sl=round(self.sl_, 5) if self.sl_ else None,
                tp=round(self.tp_, 5) if self.tp_ else None,
                short=True,
            )
            self.send_telegram_message(
                f"[MetaMLP] SELL {symbol} | "
                f"TP={self.tp_:.5f} SL={self.sl_:.5f} | "
                f"tag={self.tag}"
            )
        elif self.state == -2:
            self.close_all_positions()
            self.send_telegram_message(
                f"[MetaMLP] CLOSED all positions for {symbol} | tag={self.tag}"
            )
