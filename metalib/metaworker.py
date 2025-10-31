from metalib.metado2 import MetaDO
import schedule
import pytz
import time
from datetime import datetime, timedelta
import MetaTrader5 as mt5

# Registry maps a keyword to a strategy class and expected init keys
from metalib.metado2 import MetaDO
from metalib.metaga import MetaGA
from metalib.metagomano import MetaGO
from metalib.metahar import MetaHAR
from metalib.metafvg import MetaFVG
from metalib.metane import MetaNE

strategy_registry = {
    "metado": {
        "class": MetaDO,
        "init_keys": [
            "symbols",
            "timeframe",
            "tag",
            "size_position",
            "active_hours",
            "risk_factor",
            "mode"
        ]
    },
    "metane": {
        "class": MetaNE,
        "init_keys": [
            "symbols",
            "timeframe",
            "tag",
            "size_position",
            "active_hours",
            "lookahead",
            "rrr",
            "long_threshold",
            "tz",
            "don_window",
            "don_lag",
            "ewma_ewm_span",
            "time_ewm_span",
            "ols_window"
        ]
    },
    "metaga": {
        "class": MetaGA,
        "init_keys": [
            "symbols",
            "timeframe",
            "tag",
            "size_position",
            "active_hours",
            "low_length",
            "mid_length",
            "high_length",
            "prob_bound"
        ]
    },
    "metago": {
        "class": MetaGO,
        "init_keys": [
            "symbols",
            "timeframe",
            "tag",
            "size_position",
            "active_hours",
            "mean_rev_tf"
        ]
    },
    "metahar": {
        "class": MetaHAR,
        "init_keys": [
            "symbols",
            "predicted_symbol",
            "timeframe",
            "tag",
            "size_position",
            "active_hours",
            "short_factor",
            "long_factor"
        ]
    },
    "metafvg": {
        "class": MetaFVG,
        "init_keys": [
            "symbols",
            "timeframe",
            "tag",
            "size_position",
            "limit_number_position",
        ]
    },
}

timeframe_mapping = {
    mt5.TIMEFRAME_M1: 1,
    mt5.TIMEFRAME_M2: 2,
    mt5.TIMEFRAME_M3: 3,
    mt5.TIMEFRAME_M10: 10,
    mt5.TIMEFRAME_M15: 15,
    mt5.TIMEFRAME_M30: 30,
    mt5.TIMEFRAME_H1: 60,
    mt5.TIMEFRAME_H4: 240,
    mt5.TIMEFRAME_D1: 60*24,
}

def run_strategy_loop(strategy_type, init_args):
    strategy_info = strategy_registry.get(strategy_type)
    if not strategy_info:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    StrategyClass = strategy_info["class"]
    expected_keys = strategy_info["init_keys"]

    # Ensure required keys are present
    for key in expected_keys:
        if key not in init_args:
            raise ValueError(f"Missing required argument: {key}")

    instance = StrategyClass(**{k: init_args[k] for k in expected_keys})
    instance.connect()
    instance.fit()

    # Schedule to run every minute and fit daily
    def run_wrapper():
        end_time = datetime.now(pytz.utc) + timedelta(hours=3)
        start_time = end_time - timedelta(days=30)

        try:
            instance.run(start_time, end_time)
        except Exception as e:
            print(f"Error running strategy: {e}")

    schedule_time = timeframe_mapping[init_args["timeframe"]]

    schedule.every(schedule_time).minutes.at(":00").do(run_wrapper)
    schedule.every().day.at("00:00").do(instance.connect)
    schedule.every().day.at("00:01").do(instance.fit)

    # Fpr MetaDo, let us close all the positions every day
    # at 8 PM UTC time (3 PM NYC)
    if strategy_info["class"] == MetaDO:
        schedule.every().day.at("20:00").do(instance.close_all_positions())

    # Loop
    while True:
        schedule.run_pending()
        time.sleep(1)
