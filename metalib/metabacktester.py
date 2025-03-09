import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import vectorbt as vbt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import xgboost as xgb

from metalib.utils import *
from metalib.indicators import *
from metalib.metama import MetaMA


class MetaBackTester:
    """
    Class to backtest a trading strategy using OHLC data and indicators computed from it.
    :param dictionary: A dictionary containing the following keys:
        - symbols: List of symbols to import
        - symbol: The symbol to trade
        - timeframe: Timeframe of the data
        - start_time: Start time of the data
        - end_time: End time of the data
        - indicators: List of dictionaries containing the following keys:
            - name: Name of the indicator
            - func: Numba function to apply
            - params: Parameters to apply to the function
            - dim: Dimension of the data to apply the function
        - return_lookahead: Lookahead for the returns
        - history_len: Length of the history to consider
        - trading_start: Start time of the trading day (UTC)
        - trading_end: End time of the trading day (UTC)
        - n_principal_comp: Number of principal components to use (if 0, no PCA is applied)
    """
    def __init__(self, dictionary):
        self.validate_keys(dictionary)
        self.assign_attributes(dictionary)
        self.compute_weights()

    def validate_keys(self, dictionary):
        """
        Validates the keys of the input dictionary
        :param dictionary:
        :return:
        """
        required_keys = ['symbols', 'symbol', 'timeframe', 'start_time', 'end_time', 'indicators', 'return_lookahead',
                         'history_len', 'trading_start', 'trading_end', 'n_principal_comp', 'long_open_threshold',
                         'long_exit_threshold', 'short_open_threshold', 'short_exit_threshold', 'year']
        for key in required_keys:
            if key not in dictionary:
                raise ValueError(f"Missing required key in dictionary: {key}")

    def assign_attributes(self, dictionary):
        """
        Assigns the attributes of the class from the dictionary
        """
        for k, v in dictionary.items():
            setattr(self, k, v)

    def load_data(self):
        """
        Loads the data from the MetaMA class and computes the returns and rebased prices along with the weights
        """
        self.data = load_multiple_hist_data(self.symbols, self.year)

        if len(self.data) != 0:
            print(f"Successful data import")
            self.returns = pd.concat(map(lambda x: x['close'], self.data.values()), axis=1).apply(
                np.log).diff().dropna()
            self.returns.columns = self.symbols
            self.rebased_prices = self.returns.vbt.returns.cumulative(1).dropna()
            
    def compute_future_returns(self):
        """
        Computes the future returns of the selected symbol for the given return_lookahead
        """
        if self.returns is None:
            raise ValueError("The 'returns' attribute has not been set.")

        T = self.returns[self.symbol].shape[0]
        next_returns = [np.sum(self.returns[self.symbol][i + 1: i + self.return_lookahead]) for i in range(T)]
        next_returns = pd.Series(next_returns, index=self.returns[self.symbol].index)
        self.next_returns = next_returns

    def compute_weights(self):
        """
        Computes the weights for the indicators computation using the selected symbol as the primary asset
        """
        n = len(self.symbols)
        weights = -np.ones(n) / (n - 1)
        weights[self.symbols.index(self.symbol)] = 1
        self.weights = weights

    def compute_indicators(self):
        """
        Computes the indicators from the data and the indicators dictionary
        """
        returns = self.returns
        rebased_prices = self.rebased_prices
        indicators_df_list = []

        common_index_ = common_index([data for data in self.data.values()])
        ohlc_list = [data.iloc[:, :4].apply(np.log).values for data in self.data.values()]
        n = len(common_index_)

        for ind in self.indicators:
            if ind['dim'] == 1:
                ind_df = apply_w_diff_params_1d_nb(rebased_prices.values, ind['func'], ind['params'], self.weights)
                ind_df = pd.DataFrame(ind_df, columns=ind['params'], index=returns.index).add_suffix(f"_{ind['name']}")
                indicators_df_list.append(ind_df)
            elif ind['dim'] == 2:
                ind_df = apply_w_diff_params_2d_nb(np.array(ohlc_list), ind['func'], ind['params'], self.weights)
                ind_df = pd.DataFrame(ind_df[1:], columns=ind['params'], index=returns.index).add_suffix(f"_{ind['name']}")
                indicators_df_list.append(ind_df)
            else:
                pass

        indicators = pd.concat(indicators_df_list, axis=1).iloc[1:].dropna()
        indicators.index = pd.to_datetime(indicators.index)
        indicators.columns = indicators.columns.astype(str)
        indicators = indicators.loc[:, (indicators != indicators.iloc[0]).any()]
        self.indicators_data = indicators

    def retrieve_history(self):
        """
        Saves a historical set of indicators and returns to train the model
        """
        n = int(self.history_len * self.indicators_data.shape[0])
        hist_indicators = self.indicators_data.iloc[:n]
        hist_next_returns = self.next_returns.loc[hist_indicators.index]
        indicators = self.indicators_data.loc[self.indicators_data.index.difference(hist_indicators.index)]
        next_returns = self.next_returns.loc[indicators.index]

        stds = hist_indicators.std()
        to_keep = stds[stds != 0].index

        self.indicators_data = indicators.between_time(self.trading_start, self.trading_end).loc[:, to_keep]
        self.dummy_next_returns = next_returns.loc[indicators.index].apply(assign_cat)
        self.hist_next_returns = hist_next_returns
        self.hist_indicators = hist_indicators.loc[:, to_keep]

    def compute_pc_indicators(self):
        """
        Computes the principal components of the indicators if n_principal_comp > 0
        """
        if self.n_principal_comp > 0:
            pca = PCA(n_components=self.n_principal_comp)
            pca = pca.fit((self.hist_indicators - self.hist_indicators.mean()) / self.hist_indicators.std())
            self.indicators_pc = pd.DataFrame(
                pca.transform((self.indicators_data - self.hist_indicators.mean()) / self.hist_indicators.std()),
                index=self.indicators_data.index)
            return
        else:
            return

    def fit_model(self):
        """
        Fits an XGB model to the indicators data to predict the future returns of the selected symbol
        """
        if self.n_principal_comp > 0:
            index_features = self.indicators_pc.index
            X_train, X_test, y_train, y_test = train_test_split(self.indicators_pc,
                                                                self.dummy_next_returns.loc[index_features],
                                                                test_size=0.7, shuffle=False)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.indicators_data, self.dummy_next_returns,
                                                                test_size=0.7, shuffle=False)

        xgb_dummy = xgb.XGBClassifier().fit(X_train, y_train)
        y_hat = xgb_dummy.predict(X_test)
        self.y_hat_prob = xgb_dummy.predict_proba(X_test)
        self.y_hat_prob = pd.Series(self.y_hat_prob[:, 1], index=y_test.index)
        print(f'Fitted XGB model')

    def get_entry_exit_signals(self, signals):
        """
        Computes the entry and exit signals based on the provided signals and the thresholds defined in the class attributes.
        :param signals: The signals to use for computing the entry and exit signals.
        :return: A tuple containing the entry and exit signals for long and short positions.
        """
        entry_longs = (signals > self.long_open_threshold).astype(int)
        entry_longs = entry_longs.reindex(self.returns.index, fill_value=0)

        exit_longs = (signals < self.long_exit_threshold).astype(int)
        exit_longs = exit_longs.reindex(self.returns.index, fill_value=0)

        entry_shorts = (signals < self.short_open_threshold).astype(int)
        entry_shorts = entry_shorts.reindex(self.returns.index, fill_value=0)

        exit_shorts = (signals > self.short_exit_threshold).astype(int)
        exit_shorts = exit_shorts.reindex(self.returns.index, fill_value=0)

        return entry_longs, exit_longs, entry_shorts, exit_shorts

    def compute_backtest(self):
        """
        Computes the backtest of the trading strategy using the predicted probabilities and vectorbt
        """
        signals = self.y_hat_prob
        entry_longs, exit_longs, entry_shorts, exit_shorts = self.get_entry_exit_signals(signals)

        # Initialize the vectorbt portfolio
        portfolio = vbt.Portfolio.from_signals(
            self.returns[self.symbol].vbt.returns.cumulative(1),  # Using the 'Close' column for price data
            entries=entry_longs, accumulate=False,
            exits=exit_longs,
            short_entries=entry_shorts,
            short_exits=exit_shorts,
            size=1,  # 0.01% of the capital
            fees=0.0,  # Trading fees
            freq='1T',  # Frequency of the data
        )

        # Analyze the performance
        self.performance = portfolio.stats()
        self.portfolio = portfolio

    def run(self):
        """
        Runs the backtest
        """
        self.compute_indicators()
        self.compute_future_returns()
        self.retrieve_history()
        self.compute_pc_indicators()
        self.fit_model()
        self.compute_backtest()

    def clean_data_fields(self):
        """
        Cleans the data fields to save memory
        """
        self.hist_next_returns = None
        self.dummy_next_returns = None
        self.indicators_pc = None
        self.y_hat_prob = None
        self.next_returns = None
        self.indicators_data = None
