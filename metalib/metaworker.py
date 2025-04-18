from metalib.metado2 import MetaDO
import schedule
import pytz
import time
from datetime import datetime, timedelta

# Registry maps a keyword to a strategy class and expected init keys
from metalib.metado2 import MetaDO
from metalib.metaga import MetaGA
from metalib.metago import MetaGO
from metalib.metahar import MetaHAR

strategy_registry = {
    "metado": {
        "class": MetaDO,
        "init_keys": [
            "symbols",
            "timeframe",
            "tag",
            "active_hours",
            "risk_factor",
            "mode"
        ]
    },
    "metaga": {
        "class": MetaGA,
        "init_keys": [
            "symbols",
            "timeframe",
            "tag",
            "active_hours",
            "risk_factor",
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
            "predicted_symbol",
            "timeframe",
            "tag",
            "active_hours",
            "short_factor",
            "long_factor"
        ]
    },
    "metahar": {
        "class": MetaHAR,
        "init_keys": [
            "symbols",
            "predicted_symbol",
            "timeframe",
            "tag",
            "active_hours",
            "short_factor",
            "long_factor"
        ]
    }
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
        start_time = end_time - timedelta(days=40)
        instance.run(start_time, end_time)

    schedule.every().minute.at(":00").do(run_wrapper)
    schedule.every().day.do(instance.fit)

    # Loop
    while True:
        schedule.run_pending()
        time.sleep(1)
