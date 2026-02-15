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


# Default strategy weight parameters: (numerator, trades_per_day)
# Weight = numerator / sqrt(trades_per_day)
DEFAULT_STRATEGY_PARAMS = {
    "metafvg": {"numerator": 20, "trades_per_day": 48},
    "metago": {"numerator": 1.5, "trades_per_day": 1},
    "metane": {"numerator": 10, "trades_per_day": 12},
    "metaga": {"numerator": 30, "trades_per_day": 48},
    "metaob": {"numerator": 1, "trades_per_day": 1},
}


def compute_weight(numerator: float, trades_per_day: float) -> float:
    """Compute strategy weight from numerator and trades per day."""
    return numerator / np.sqrt(trades_per_day)


def params_to_weights(strategy_params: dict) -> dict:
    """Convert strategy params dict to weights dict."""
    return {
        k: compute_weight(v["numerator"], v["trades_per_day"])
        for k, v in strategy_params.items()
    }


class MetaScale(MetaStrategy):
    """Portfolio Weight Optimization Strategy using convex optimization."""

    def __init__(
        self,
        tag,
        risk_pct=0.005,
        config_dir="../config/prod",
        strategy_params=None,
        lookback_days=92,
        auto_connect=True,
        verbose=True,
    ):
        """
        Initialize the Position Sizing Strategy.

        Args:
            tag: Strategy identifier for logging
            risk_pct: Fraction of account balance to risk (e.g., 0.015 = 1.5%)
            config_dir: Path to YAML config directory
            strategy_params: Dict of {strategy_type: {"numerator": N, "trades_per_day": T}}
                             If None, uses DEFAULT_STRATEGY_PARAMS
            lookback_days: Days of price history for covariance calculation
            auto_connect: Whether to connect to MT5 on init
            verbose: Whether to print progress messages
        """
        super().__init__([], 0, tag, 0)  # Init with empty symbols and timeframe

        self.verbose = verbose
        self._log(f"Initializing MetaScaler strategy..")
        self.timeframe = mt5.TIMEFRAME_M15
        self.lookback_days = lookback_days

        if auto_connect:
            self.connect()

        # Strategy weights - use provided params or defaults
        if strategy_params is None:
            strategy_params = DEFAULT_STRATEGY_PARAMS

        self.strategy_params = strategy_params
        self.strategy_weights = params_to_weights(strategy_params)
        self.running_strategies = list(self.strategy_weights.keys())

        self.risk_pct = risk_pct
        self.config_dir = config_dir
        self.weights = None
        self.weights_df = None  # DataFrame version of results
        self.old_sizes = {}  # Track old sizes for comparison

        # Get balance if connected
        if auto_connect:
            account_info = mt5.account_info()
            self.balance = account_info.balance if account_info else 0
        else:
            self.balance = 0

        self.mu = self.balance * self.risk_pct

        self._log(f"Running strategies:     {self.running_strategies}")
        self._log(f"Strategy weights:       {self.strategy_weights}")
        self._log(f"Risk percentage:        {self.risk_pct}")
        self._log(f"Initial balance:        {self.balance}")
        self._log(f"Amount to risk:         {self.mu}")

    def _log(self, message: str):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(f"{self.tag}::    {message}")

    def set_balance(self, balance: float):
        """Manually set account balance (useful when MT5 not connected)."""
        self.balance = balance
        self.mu = self.balance * self.risk_pct
        self._log(f"Balance set to: {self.balance}, mu = {self.mu}")

    def _pull_yaml_configs(self, config_dir: str):
        """
        Reads all YAML files in the specified directory and extracts the assets along with their strategy names.

        Args:
            config_dir (str): The path to the directory containing the YAML files.

        Returns:
            None (sets self.instances, self.symbols, self.old_sizes)

        Raises:
            FileNotFoundError: If the specified directory does not exist.
            ValueError: If a YAML file cannot be parsed.
        """

        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Directory '{config_dir}' does not exist.")

        instances = []
        old_sizes = {}
        yaml_files = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]

        for yaml_file in yaml_files:
            yaml_path = os.path.join(config_dir, yaml_file)

            with open(yaml_path, "r") as file:
                try:
                    data = yaml.safe_load(file)  # Safely load the YAML data
                    if data is None:
                        continue
                    for instance_config in data.values():
                        if not isinstance(instance_config, dict):
                            continue
                        strategy_type = instance_config.get("strategy_type")
                        symbols = instance_config.get("symbols", [])
                        size_position = instance_config.get("size_position", 0.0)
                        if strategy_type and symbols:
                            symbol = symbols[0]
                            instances.append((strategy_type, symbol))
                            old_sizes[(strategy_type, symbol)] = size_position

                except yaml.YAMLError as e:
                    raise ValueError(f"Error reading YAML file '{yaml_file}': {e}")

        instances = pd.DataFrame(instances, columns=["strategy_type", "symbol"])
        self.instances = instances
        self.symbols = instances["symbol"].unique()
        self.old_sizes = old_sizes
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
            if symbol not in diff_df.columns:
                continue
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self._log(f"Warning: Could not get symbol info for {symbol}")
                continue

            tick_value = symbol_info.trade_tick_value
            tick_size = symbol_info.trade_tick_size

            if tick_size <= 0 or tick_value <= 0:
                self._log(f"Warning: Invalid tick params for {symbol}")
                continue

            self._log(f"Applying MT5 tick params for {symbol}: "
                      f"tick_value={tick_value}, tick_size={tick_size}")
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

        # Ensure symmetric
        Sigma = (Sigma + Sigma.T) / 2

        # Check positive semidefinite
        eigenvalues = np.linalg.eigvals(Sigma)
        if not np.all(eigenvalues >= -1e-8):
            self._log("Warning: Matrix not positive semidefinite, adjusting...")
            Sigma = Sigma + (abs(min(eigenvalues)) + 1e-3) * np.eye(n_instances)

        sigma_diag = np.sqrt(np.diag(Sigma))

        prob = cx.Problem(
            cx.Minimize(cx.quad_form(w, Sigma)),
            [cx.sum(cx.multiply(w, sigma_diag)) == mu, w >= 0],
        )
        prob.solve()
        if prob.status == "optimal":
            self._log("Optimization successful..")
            return w
        else:
            raise ValueError(f"Optimization failed with status: {prob.status}")

    def fit(self, save_to_yaml: bool = True):
        """
        Run the optimization and compute optimal position sizes.

        Args:
            save_to_yaml: If True, write results to YAML config files

        Returns:
            pd.DataFrame with columns: strategy_type, symbol, old_size, new_size
        """
        self._log("Starting the MetaScaler fit!!..")

        # Pulling YAML configs
        self._pull_yaml_configs(self.config_dir)
        self._log(f"Running symbols: {list(self.symbols)}")

        # Load price data
        utc = pytz.timezone("UTC")
        end_time = datetime.now(utc)
        start_time = end_time - timedelta(days=self.lookback_days)
        end_time = end_time.replace(
            hour=0, minute=0, second=0, microsecond=0
        ).astimezone(utc)
        start_time = start_time.astimezone(utc)

        # Pulling last days of data
        self.loadData(start_time, end_time)

        # Build price matrix for available symbols
        available_symbols = [s for s in self.symbols if s in self.data]
        if not available_symbols:
            raise ValueError("No price data available for any symbol")

        data = pd.concat([self.data[s].close for s in available_symbols], axis=1)
        data.columns = available_symbols

        # Convert using mt5 tick value and size
        price_diffs = data.diff()
        price_diffs_adj = self._apply_mt5_tick_params(price_diffs)

        # Compute daily covariance of price differences (scale from 15-min to daily)
        self.cov_assets = price_diffs_adj.cov() * 4 * 24
        self.cov_strategies = self._fetch_strategies_cov()

        # Run optimization and save rounded weights
        Sigma = self.cov_strategies.values
        weights = self._run_optimization(Sigma, self.mu)
        weights = np.round(weights.value, 2)

        self.weights = {
            tuple(k): v for k, v in zip(self.running_instances.values, weights)
        }
        self._log(f"Weights: {self.weights}")

        # Build results DataFrame
        results = []
        for (strat, sym), new_size in self.weights.items():
            old_size = self.old_sizes.get((strat, sym), 0.0)
            results.append({
                "strategy_type": strat,
                "symbol": sym,
                "old_size": old_size,
                "new_size": float(new_size),
            })

        self.weights_df = pd.DataFrame(results)

        # Write changes to yaml config files if requested
        if save_to_yaml:
            self._write_changes_to_yamls(self.config_dir)

        self._log("Done!")
        return self.weights_df

    def _write_changes_to_yamls(self, config_dir: str = None) -> dict:
        """
        Write optimized weights to YAML config files.

        Args:
            config_dir: Path to config directory (uses self.config_dir if None)

        Returns:
            dict with keys: success, files_updated, errors
        """
        if config_dir is None:
            config_dir = self.config_dir

        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Directory '{config_dir}' does not exist.")

        yaml_files = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]
        files_updated = 0
        errors = []

        for yaml_file in yaml_files:
            yaml_path = os.path.join(config_dir, yaml_file)
            write_yaml = False

            try:
                with open(yaml_path, "r") as file:
                    data = yaml.safe_load(file)

                if data is None:
                    continue

                for instance in data:
                    if not isinstance(data[instance], dict):
                        continue
                    strategy_type = data[instance].get("strategy_type")
                    if strategy_type in self.running_strategies:
                        symbols = data[instance].get("symbols", [])
                        if symbols:
                            asset = symbols[0]
                            key = (strategy_type, asset)
                            if key in self.weights:
                                write_yaml = True
                                data[instance]["size_position"] = float(self.weights[key])

                if write_yaml:
                    with open(yaml_path, "w") as file:
                        yaml.dump(data, file, default_flow_style=False, sort_keys=False)
                    files_updated += 1
                    self._log(f"Changes written to YAML file '{yaml_file}'")

            except Exception as e:
                error_msg = f"Error writing to '{yaml_file}': {e}"
                errors.append(error_msg)
                self._log(error_msg)

        return {
            "success": len(errors) == 0,
            "files_updated": files_updated,
            "errors": errors,
        }

    def save_weights(self, config_dir: str = None) -> dict:
        """
        Public method to save weights to YAML files.

        Args:
            config_dir: Path to config directory (uses self.config_dir if None)

        Returns:
            dict with keys: success, files_updated, errors
        """
        if self.weights is None:
            raise ValueError("No weights to save. Run fit() first.")
        return self._write_changes_to_yamls(config_dir)

    def signals(self):
        return

    def check_conditions(self):
        return
