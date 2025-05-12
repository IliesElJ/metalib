import yaml
import warnings
from multiprocessing import Process
from metalib.metaworker import run_strategy_loop
import MetaTrader5 as mt5

warnings.filterwarnings("ignore")

def main():
    processes = []

    with open("../config/prod/metaga.yaml", "r") as f:
        config_data = yaml.safe_load(f)

    for name, entry in config_data.items():
        strategy_type = entry.pop("strategy_type")
        init_args = entry.copy()

        # Convert timeframe string (e.g. "TIMEFRAME_M1") to actual mt5 constant
        if isinstance(init_args.get("timeframe"), str):
            init_args["timeframe"] = eval(init_args["timeframe"])

        # Convert null active_hours to None
        if "active_hours" in init_args and init_args["active_hours"] == "None":
            init_args["active_hours"] = None

        # Start the strategy in its own process
        p = Process(target=run_strategy_loop, args=(strategy_type, init_args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()