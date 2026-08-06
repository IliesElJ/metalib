"""Shared config loader with local-override support for strategy main scripts."""

import os
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
PROD_CONFIG_DIR = os.path.join(_HERE, 'config', 'prod')
LOCAL_CONFIG_DIR = os.path.join(_HERE, 'config', 'local')


def load_strategy_config(strategy_name):
    """Return parsed YAML config dict, preferring config/local/ over config/prod/."""
    local = os.path.join(LOCAL_CONFIG_DIR, f'{strategy_name}.yaml')
    prod = os.path.join(PROD_CONFIG_DIR, f'{strategy_name}.yaml')
    path = local if os.path.exists(local) else prod
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}
