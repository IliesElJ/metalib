"""
Calibration Utilities
Wraps MetaScale functionality for use in the dashboard calibration tab.
"""

import os
import sys
import pandas as pd

# Add parent directory to path for metalib imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def run_metascale_optimization(
    strategy_params: dict,
    risk_pct: float = 0.015,
    lookback_days: int = 92,
    config_dir: str = "../config/prod",
) -> dict:
    """
    Run the MetaScale portfolio optimization.

    Args:
        strategy_params: Dict of {strategy_type: {"numerator": N, "trades_per_day": T}}
        risk_pct: Risk percentage as decimal (e.g., 0.015 for 1.5%)
        lookback_days: Days of history for covariance calculation
        config_dir: Path to YAML config directory

    Returns:
        dict with keys:
            - success: bool
            - weights_df: DataFrame with (strategy_type, symbol, old_size, new_size)
            - error: str (if success is False)
    """
    try:
        from metalib.metascal import MetaScale

        # Resolve config directory path relative to metadash
        if config_dir.startswith(".."):
            config_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", config_dir)
            )

        # Create MetaScale instance with custom params
        scaler = MetaScale(
            tag="dashboard",
            risk_pct=risk_pct,
            config_dir=config_dir,
            strategy_params=strategy_params,
            lookback_days=lookback_days,
            auto_connect=True,
            verbose=True,
        )

        # Run optimization without auto-saving
        weights_df = scaler.fit(save_to_yaml=False)

        return {
            "success": True,
            "weights_df": weights_df,
            "scaler": scaler,
            "cov_assets": scaler.cov_assets,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def save_weights_to_yaml(weights_df: pd.DataFrame, config_dir: str, scaler=None) -> dict:
    """
    Save optimized weights to YAML config files.

    Args:
        weights_df: DataFrame with (strategy_type, symbol, new_size)
        config_dir: Path to YAML config directory
        scaler: Optional MetaScale instance (if available, uses its save method)

    Returns:
        dict with keys:
            - success: bool
            - files_updated: int
            - error: str (if success is False)
    """
    try:
        # Resolve config directory path relative to metadash
        if config_dir.startswith(".."):
            config_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", config_dir)
            )

        # If we have a scaler instance, use its save method
        if scaler is not None:
            result = scaler.save_weights(config_dir)
            return result

        # Otherwise, do manual save using the weights_df
        from metalib.metascal import MetaScale
        import yaml

        if not os.path.exists(config_dir):
            return {"success": False, "error": f"Directory not found: {config_dir}"}

        # Build lookup dict
        weights_lookup = {
            (row["strategy_type"], row["symbol"]): row["new_size"]
            for _, row in weights_df.iterrows()
        }

        running_strategies = set(weights_df["strategy_type"].unique())

        yaml_files = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]
        files_updated = 0

        for yaml_file in yaml_files:
            yaml_path = os.path.join(config_dir, yaml_file)
            write_needed = False

            with open(yaml_path, "r") as file:
                try:
                    data = yaml.safe_load(file)
                    if data is None:
                        continue
                except yaml.YAMLError:
                    continue

            for instance_name, instance_config in data.items():
                if not isinstance(instance_config, dict):
                    continue

                strategy_type = instance_config.get("strategy_type")
                symbols = instance_config.get("symbols", [])

                if strategy_type in running_strategies and symbols:
                    symbol = symbols[0]
                    key = (strategy_type, symbol)

                    if key in weights_lookup:
                        instance_config["size_position"] = float(weights_lookup[key])
                        write_needed = True

            if write_needed:
                with open(yaml_path, "w") as file:
                    yaml.dump(data, file, default_flow_style=False, sort_keys=False)
                files_updated += 1

        return {"success": True, "files_updated": files_updated}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
