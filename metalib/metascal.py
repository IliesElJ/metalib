from metalib.metastrategy import MetaStrategy
import pytz
from datetime import datetime, timedelta
import os
import yaml
import pandas as pd
import MetaTrader5 as mt5
import numpy as np
import cvxpy as cx
import sys


class MetaScale(MetaStrategy):
    """MetaTrader FVG (Fair Value Gap) Trading Strategy"""

    def __init__(self, tag, risk_pct=0.005, config_dir="../config/prod"):
        """
        Initialize the Position Sizing Strategy
        """
        super().__init__([], 0, tag, 0)  # Init with empty symbols and timeframe

        print(f"{self.tag}::    Initializing MetaScaler strategy..")
        self.timeframe = mt5.TIMEFRAME_M15
        self.connect()
        # Strategy weights
        strategy_weights = dict()
        strategy_weights["metafvg"] = 20 / np.sqrt(48)
        strategy_weights["metago"] = np.sqrt(1.5)
        strategy_weights["metane"] = 10 / np.sqrt(12)
        strategy_weights["metaga"] = 30 / np.sqrt(48)
        strategy_weights["metaob"] = np.sqrt(1)

        self.running_strategies = list(strategy_weights.keys())
        self.strategy_weights = strategy_weights
        self.risk_pct = risk_pct
        self.balance = mt5.account_info().balance
        self.mu = self.balance * self.risk_pct
        self.config_dir = config_dir
        self.weights = None

        print(f"{self.tag}::    Running strategies:     {self.running_strategies}")
        print(f"{self.tag}::    Risk percentage:        {self.risk_pct}")
        print(f"{self.tag}::    Initial balance:        {self.balance}")
        print(f"{self.tag}::    Amount to risk:         {self.mu}")

    def _pull_yaml_configs(self, config_dir: str):
        """
        Reads all YAML files in the specified directory and extracts the assets along with their strategy names.

        Args:
            config_dir (str): The path to the directory containing the YAML files.

        Returns:
            dict: A dictionary where the keys are strategy names (file names without extensions),
                  and values are the asset data from each YAML file.

        Raises:
            FileNotFoundError: If the specified directory does not exist.
            ValueError: If a YAML file cannot be parsed.
        """

        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Directory '{config_dir}' does not exist.")

        instances = []
        yaml_files = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]

        for yaml_file in yaml_files:
            strategy_name = os.path.splitext(yaml_file)[
                0
            ]  # Get the file name without extension
            yaml_path = os.path.join(config_dir, yaml_file)

            with open(yaml_path, "r") as file:
                try:
                    data = yaml.safe_load(file)  # Safely load the YAML data
                    instances += [
                        (i["strategy_type"], i["symbols"][0]) for i in data.values()
                    ]

                except yaml.YAMLError as e:
                    raise ValueError(f"Error reading YAML file '{yaml_file}': {e}")

        instances = pd.DataFrame(instances, columns=["strategy_type", "symbol"])
        self.instances = instances
        self.symbols = instances["symbol"].unique()
        return

    def _apply_mt5_tick_params(self, diff_df: pd.DataFrame):
        """
        Apply MT5 tick parameters to normalize price differences.

        Args:
            diff_df: DataFrame containing price differences for each symbol

        Returns:
            DataFrame with adjusted price differences based on tick value and size
        """
        for symbol in self.symbols:
            symbol_info = mt5.symbol_info(symbol)
            tick_value = symbol_info.trade_tick_value
            tick_size = symbol_info.trade_tick_size

            assert (
                tick_size > 0
            ), f"Tick size must be greater than 0, current value: {tick_size}"
            assert (
                tick_value > 0
            ), f"Tick value must be greater than 0, current value: {tick_value}"
            print(f"{self.tag}::    Applying MT5 tick params for {symbol}")
            print(f"{self.tag}::    Tick value: {tick_value}, Tick size: {tick_size}")
            diff_df.loc[:, symbol] = diff_df.loc[:, symbol].apply(
                lambda x: x * tick_value / tick_size
            )
        return diff_df

    def _fetch_strategies_cov(self) -> pd.DataFrame:
        """
        Calculate covariance matrix for running strategy instances.

        Returns:
            DataFrame containing the covariance matrix weighted by strategy weights
        """
        strategies_running_cov = pd.DataFrame()
        running_instances = self.instances[
            self.instances["strategy_type"].isin(self.running_strategies)
        ]
        symbols_running_cov = self.cov_assets

        for f_instance in running_instances.itertuples():
            for s_instance in running_instances.itertuples():
                if f_instance[1] != s_instance[1]:
                    val = 0
                else:
                    f_symbol = f_instance[2]
                    s_symbol = s_instance[2]
                    val = (
                        symbols_running_cov.loc[f_symbol, s_symbol]
                        * self.strategy_weights[f_instance[1]] ** 2
                    )
                strategies_running_cov.loc[f_instance, s_instance] = val

        self.running_instances = running_instances
        return strategies_running_cov

    def _run_optimization(self, Sigma: np.ndarray, mu: np.number) -> np.ndarray:
        """
        Run portfolio optimization to find optimal position sizes.

        Args:
            Sigma: Covariance matrix of strategies
            mu: Target risk amount

        Returns:
            Optimized weight vector

        Raises:
            ValueError: If optimization fails to converge
        """
        n_instances = len(self.running_instances)
        assert Sigma.shape[0] > 0, "Sigma matrix is empty"
        assert n_instances > 0, "No running instances found"
        w = cx.Variable(n_instances)
        # Numerical stability
        Sigma = Sigma + 1e-3 * np.eye(n_instances)

        # Ensure Sig is symmetric
        assert np.allclose(Sigma, Sigma.T), "Covariance matrix not symmetric"
        # Check positive semidefinite
        eigenvalues = np.linalg.eigvals(Sigma)
        assert np.all(eigenvalues >= -1e-8), "Matrix not positive semidefinite"

        sigma_diag = np.sqrt(np.diag(Sigma))

        prob = cx.Problem(
            cx.Minimize(cx.quad_form(w, Sigma)),
            [cx.sum(cx.multiply(w, sigma_diag)) == mu, w >= 0],
        )
        prob.solve()
        if prob.status == "optimal":
            print(f"{self.tag}::    Optimization successful..")
            return w
        else:
            raise ValueError("Optimization failed..")

    def fit(self):
        print(f"{self.tag}::    Starting the MetaScaler fit!!..")
        # Pulling YAML configs
        self._pull_yaml_configs(self.config_dir)
        print(f"{self.tag}::    Running symbols: {self.symbols}")

        # Load the last 92 days of data
        utc = pytz.timezone("UTC")
        # Get the current time in UTC
        end_time = datetime.now(utc)
        start_time = end_time - timedelta(days=92)
        # Set the time components to 0 (midnight) and maintain the timezone
        end_time = end_time.replace(
            hour=0, minute=0, second=0, microsecond=0
        ).astimezone(utc)
        start_time = start_time.astimezone(utc)

        # Pulling last days of data
        self.loadData(start_time, end_time)
        data = pd.concat([self.data[s].close for s in self.symbols], axis=1)
        data.columns = self.symbols

        # Convert using mt5 tick value and size
        price_diffs = data.diff()
        price_diffs_adj = self._apply_mt5_tick_params(price_diffs)
        # Compute daily covariance of price differences
        self.cov_assets = price_diffs_adj.cov() * 4 * 24  # Assuming 15-minute bars
        self.cov_strategies = self._fetch_strategies_cov()

        # Run optimization and save rounded weights
        Sigma = self.cov_strategies.values
        weights = self._run_optimization(Sigma, self.mu)
        weights = np.round(weights.value, 2)
        self.weights = {
            tuple(k): v for k, v in zip(self.running_instances.values, weights)
        }
        print(f"{self.tag}::    Weights: {self.weights}")

        # Write changes to yaml config files
        self._write_changes_to_yamls(self.config_dir)
        print(f"{self.tag}::    Done!")

    def _write_changes_to_yamls(self, config_dir: str):
        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Directory '{config_dir}' does not exist.")

        yaml_files = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]

        for yaml_file in yaml_files:
            yaml_path = os.path.join(config_dir, yaml_file)
            write_yaml = False

            with open(yaml_path, "r+") as file:
                try:
                    data = yaml.safe_load(file)  # Safely load the YAML data
                    for instance in data:
                        if data[instance]["strategy_type"] in self.running_strategies:
                            write_yaml = True
                            strategy_type = data[instance]["strategy_type"]
                            asset = data[instance]["symbols"][0]
                            weight = self.weights[(strategy_type, asset)]
                            data[instance]["size_position"] = float(weight)
                        else:
                            print(
                                f"{self.tag}::    Strategy {instance['strategy_type']} not running.."
                            )
                    if write_yaml:
                        file.seek(0)  # Go to the start of the file
                        yaml.dump(
                            data, file, default_flow_style=False
                        )  # Write updated data
                        file.truncate()  # Remove any leftover content
                        print(
                            f"{self.tag}::    Changes written to YAML file '{yaml_file}'"
                        )
                except Exception as e:
                    print(
                        f"{self.tag}::    Error writing to YAML file '{yaml_file}': {e}"
                    )

    def signals(self):
        return

    def check_conditions(self):
        return
