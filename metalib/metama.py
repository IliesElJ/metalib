from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pytz as pytz
import numpy as np
import pandas as pd
import pickle
import vectorbt as vbt

import xgboost as xgb
from scipy.stats import skew, kurtosis

from metalib.metastrategy import MetaStrategy
from metalib.indicators import *


class MetaMA(MetaStrategy):

    def __init__(self, 
                 symbols, 
                 asset, 
                 timeframe, 
                 tag, 
                 active_hours, 
                 risk_factor=1, 
                ):
        
        super().__init__(symbols, timeframe, tag, active_hours)
        self.model = None
        self.indicators_std = None
        self.indicators_mean = None
        self.state = None
        self.risk_factor = risk_factor
        self.asset = asset
        self.telegram = True

    def signals(self):
        returns = pd.concat(map(lambda x: x['close'], self.data.values()), axis=1).apply(np.log).diff()
        returns.columns = self.symbols
        indicators = self.retrieve_indicators(returns=returns)

        # Demean Indicatorss
        indicators = indicators.tail(3)
        indicators = (indicators - self.indicators_mean) / self.indicators_std
        current_time = indicators.index[-1]

        if indicators.iloc[[-1]].between_time("08:00", "11:00").shape[0] > 0:
            model = self.model['LDN']
        elif indicators.iloc[[-1]].between_time("14:00", "17:00").shape[0] > 0:
            model = self.model['NY']
        elif indicators.iloc[[-1]].between_time("01:00", "04:00").shape[0] > 0:
            model = self.model['AS']
        else:
            if self.are_positions_with_tag_open():
                # compute the timedelta between the current time and 9:30, 15:30, 2:30
                time_diff_ldn = abs((current_time - current_time.replace(hour=9, minute=30)).total_seconds())
                time_diff_ny = abs((current_time - current_time.replace(hour=15, minute=30)).total_seconds())
                time_diff_as = abs((current_time - current_time.replace(hour=2, minute=30)).total_seconds())

                if time_diff_ldn < time_diff_ny and time_diff_ldn < time_diff_as:
                    print(f"{self.tag}::: Using LDN model.")
                    model = self.model['LDN']
                elif time_diff_ny < time_diff_ldn and time_diff_ny < time_diff_as:
                    print(f"{self.tag}::: Using NY model.")
                    model = self.model['NY']
                elif time_diff_as < time_diff_ldn and time_diff_as < time_diff_ny:
                    print(f"{self.tag}::: Using AS model.")
                    model = self.model['AS']
            else:
                print(f"{self.tag}::: No traders available for the current time slot.")
                return

        y_hat = model.predict_proba(indicators)[:, 1]

        if y_hat[-1] < 0.3 and self.are_positions_with_tag_open(position_type="buy"):
            self.state = -2
        elif y_hat[-1] > 0.7 and self.are_positions_with_tag_open(position_type="sell"):
            self.state = -2
        elif y_hat[-1] > 0.96 and not self.are_positions_with_tag_open(position_type="buy"):
            self.state = 1
        elif y_hat[-1] < 0.04 and not self.are_positions_with_tag_open(position_type="sell"):
            self.state = -1
        else:
            self.state = 0

        print(f"{self.tag}::: Open positions for strategy: {self.tag}: {self.are_positions_with_tag_open()}")
        print(f"{self.tag}::: Last 3 predicted probabilities: {y_hat}")

        signal_line = indicators.iloc[[-1]]
        signal_line.loc[:, 'predicted_proba'] = y_hat[-1]

        self.signals_data = signal_line

    def check_conditions(self):
        volume = self.position_sizing(0.5, self.symbols[0])

        if self.state == 0:
            pass
        elif self.state == 1:
            self.execute(symbol=self.asset, short=False)
            # Send a message when an order is entered
            self.send_telegram_message(f"Entered order for {self.symbols[0]} with volume: {volume}")
        elif self.state == -1:
            self.execute(symbol=self.asset, short=True)
            # Send a message when an order is entered
            self.send_telegram_message(f"Entered order for {self.symbols[0]} with volume: {volume}")
        elif self.state == -2:
            self.close_all_positions()
            # Send a message when positions are closed
            self.send_telegram_message(f"Closed all positions for {self.symbols[0]}")

    def fit(self):
        # Define the UTC timezone
        utc = pytz.timezone('UTC')
        # Get the current time in UTC
        end_time = datetime.now(utc)
        start_time = end_time - timedelta(days=60)
        # Set the time components to 0 (midnight) and maintain the timezone
        end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(utc)
        start_time = start_time.astimezone(utc)

        # Pulling last days of data
        self.loadData(start_time, end_time)
        returns = pd.concat(map(lambda x: x['close'], self.data.values()), axis=1).apply(np.log).diff().dropna()
        returns.columns = self.symbols

        # Compute rolling next returns series
        T = returns.shape[0]
        next_five_returns = [np.sum(returns[self.asset][i + 1: i + 121]) for i in range(T)]
        next_five_returns = pd.Series(next_five_returns, index=returns.index)

        # Indicators
        indicators = self.retrieve_indicators(returns=returns)

        # Retrieve history
        hist_indicators = indicators[:35000]
        hist_next_five_returns = next_five_returns.loc[hist_indicators.index]
        indicators = indicators.loc[indicators.index.difference(hist_indicators.index)]
        next_five_returns = next_five_returns.loc[indicators.index]

        # Demean from history
        indicators = (indicators - hist_indicators.mean()) / hist_indicators.std()
        next_five_returns = (next_five_returns - hist_next_five_returns.mean()) / hist_next_five_returns.std()

        # Transform to dummy
        dummy_extremes_indicators = abs(indicators) > 1.0
        indicators = indicators[dummy_extremes_indicators.sum(axis=1) > 6]
        dummy_extremes_next_five_returns = next_five_returns.loc[indicators.index].apply(assign_cat)

        session_a_indicators = indicators.between_time("08:00", "11:00")
        session_b_indicators = indicators.between_time("14:00", "17:00")
        session_c_indicators = indicators.between_time("01:00", "04:00")

        next_returns_a = next_five_returns.loc[session_a_indicators.index].apply(assign_cat)
        next_returns_b = next_five_returns.loc[session_b_indicators.index].apply(assign_cat)
        next_returns_c = next_five_returns.loc[session_c_indicators.index].apply(assign_cat)

        X_a, y_a = session_a_indicators.ffill(), next_returns_a
        X_b, y_b = session_b_indicators.ffill(), next_returns_b
        X_c, y_c = session_c_indicators.ffill(), next_returns_c

        xgb_dummy_a = xgb.XGBClassifier().fit(X_a, y_a)
        xgb_dummy_b = xgb.XGBClassifier().fit(X_b, y_b)
        xgb_dummy_c = xgb.XGBClassifier().fit(X_c, y_c)

        print(f"{self.tag}::: XGBoost Model trained from {X_c.index[0]} to {X_b.index[-1]}.")

        # Save model, 1st and 2nd indicator moments
        self.model = {"LDN": xgb_dummy_a, "NY": xgb_dummy_b, "AS": xgb_dummy_c}
        self.indicators_mean = hist_indicators.mean()
        self.indicators_std = hist_indicators.std()
        print(f"{self.tag}::: XGBoost Model and first/second moments saved.")

    def retrieve_indicators(self, returns):
        returns_ = returns.copy()

        # Rolling Covariance Matrices
        cov_matrices_hour = rolling_covariance_nb(returns_.values, window_size=60)
        cov_matrices_session = rolling_covariance_nb(returns_.values, window_size=3 * 60)
        cov_matrices_daily = rolling_covariance_nb(returns_.values, window_size=20 * 60)
        print(f"{self.tag}::: Computed rolling covariance matrices")

        # Rolling covariances descriptive vars
        coeff_stats_hour = apply_to_3d_array(cov_matrices_hour, compute_stats, 20)
        coeff_stats_session = apply_to_3d_array(cov_matrices_session, compute_stats, 20)
        coeff_stats_daily = apply_to_3d_array(cov_matrices_daily, compute_stats, 20)

        coeff_stats_hour = pd.DataFrame(coeff_stats_hour, index=returns.index).dropna(axis=1, how='all')
        coeff_stats_session = pd.DataFrame(coeff_stats_session, index=returns.index).dropna(axis=1, how='all')
        coeff_stats_daily = pd.DataFrame(coeff_stats_daily, index=returns.index).dropna(axis=1, how='all')

        coeff_features = ['coeffs_mean', 'coeffs_std', 'coeffs_min', 'coeffs_max', 'coeffs_50%', 'coeffs_1%',
                          'coeffs_99%', 'coeffs_10%', 'coeffs_90%', 'coeffs_iqr1', 'coeffs_iqr2']
        coeff_stats_hour.columns = [sub + '_hour' for sub in coeff_features]
        coeff_stats_session.columns = [sub + '_session' for sub in coeff_features]
        coeff_stats_daily.columns = [sub + '_daily' for sub in coeff_features]
        print(f"{self.tag}::: Computed covariances descriptive vars")

        # Rolling Eigen features
        eigen_features_hour = apply_to_3d_array(cov_matrices_hour, compute_eigen_features_nb, 20)
        eigen_features_session = apply_to_3d_array(cov_matrices_session, compute_eigen_features_nb, 20)
        eigen_features_daily = apply_to_3d_array(cov_matrices_daily, compute_eigen_features_nb, 20)

        eigen_features_hour = pd.DataFrame(eigen_features_hour, index=returns.index).dropna(axis=1, how='all')
        eigen_features_session = pd.DataFrame(eigen_features_session, index=returns.index).dropna(axis=1, how='all')
        eigen_features_daily = pd.DataFrame(eigen_features_daily, index=returns.index).dropna(axis=1, how='all')

        eigen_features = [
            "varex_eig1",  # Variance Explained by First Eigenvalue
            "varex_eig_top5",  # Variance Explained by Top 5 Eigenvalues
            "varex_eig_top30",  # Variance Explained by Top 30 Eigenvalues
            "varex_5-1",  # Difference in Variance Explained between Top 5 Eigenvalues and First Eigenvalue
            "varex_30-5",  # Difference in Variance Explained between Top 30 Eigenvalues and Top 5 Eigenvalues
            "varex_eig_MP",  # Variance Explained by Eigenvalues Outside Marcenko-Pastur Distribution
            "determinant",  # Product of Positive Eigenvalues
        ]

        eigen_features_hour.columns = [sub + '_hour' for sub in eigen_features]
        eigen_features_session.columns = [sub + '_session' for sub in eigen_features]
        eigen_features_daily.columns = [sub + '_daily' for sub in eigen_features]
        print(f"{self.tag}::: Computed eigen-features")

        eigen_features = pd.concat([eigen_features_hour,
                                    eigen_features_session,
                                    eigen_features_daily], axis=1)

        coeff_stats = pd.concat([coeff_stats_hour,
                                 coeff_stats_session,
                                 coeff_stats_daily], axis=1)

        # Merge Features
        indicators = [eigen_features, coeff_stats]
        indicators = pd.concat(indicators, axis=1).iloc[1:]

        indicators.index = pd.to_datetime(indicators.index)
        indicators.columns = list(range(indicators.shape[1]))
        indicators.columns = indicators.columns.astype(str)
        print(f"{self.tag}::: Merged indicators")

        return indicators.dropna(how='all', axis=1)


def assign_cat(val):
    if val < 0.:
        return 0
    else:
        return 1
