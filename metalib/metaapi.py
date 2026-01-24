import warnings
from typing import Dict, Any
from pathlib import Path
import yaml
from fastapi import FastAPI, Request
from metalib.metacontroller import MetaController
from metalib.constants import DEFAULT_CONFIG_PATH

warnings.filterwarnings("ignore")
app = FastAPI()
controller = MetaController()


def start_strategy_instances(metacontroller) -> Dict[str, Any]:
    """
    Start strategy instances based on configurations from all YAML files in CONFIG_PATH.

    Args:
        metacontroller: MetaController instance managing strategy processes

    Returns:
        Dictionary with status information for each started strategy
    """
    instances = {}

    try:
        # Process all yaml files in the config directory
        configs = sorted(DEFAULT_CONFIG_PATH.glob("*.yaml"))
        print(f"API::Found {len(configs)} configuration files in {DEFAULT_CONFIG_PATH}")

        for config_file in configs:
            print(f"API::Starting instances from {config_file.name}")

            with config_file.open("r") as f:
                strategy_configs = yaml.safe_load(f)

            for strategy_name, config in strategy_configs.items():
                strategy_type = config.pop("strategy_type")
                init_args = config.copy()
                message, pid, running = metacontroller.start_script(
                    strategy_type, init_args
                )

                instances[strategy_name] = {
                    "Message": message,
                    "PID": pid,
                    "Running": running,
                }

        return instances

    except FileNotFoundError:
        return {"error": f"Configuration directory not found at {DEFAULT_CONFIG_PATH}"}
    except yaml.YAMLError:
        return {"error": f"Invalid YAML configuration file"}


# Usage in FastAPI endpoint
@app.get("/start_stored_instances")
def start_stored_instances():
    print("API::Starting stored instances")
    return start_strategy_instances(controller)


@app.get("/")
def read_root():
    return {"Meta": "API"}


@app.get("/list")
def list():
    return controller.list_processes()


@app.get("/stop/{tag}")
def stop(tag: str):
    return controller.stop_instance(tag)


@app.get("/stop_all_running")
def stop(tag: str):
    return controller.stop_all_running()


@app.get("/start")
async def start(request: Request):
    query_params = dict(request.query_params)
    strategy_type = query_params.pop("strategy_type")
    init_args = query_params.copy()
    return controller.start_script(strategy_type, init_args)
