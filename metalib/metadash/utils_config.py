"""Dashboard-side config utilities: load, save, and list strategy YAML configs."""

import os
import yaml

_METADASH_DIR = os.path.dirname(os.path.abspath(__file__))
_METALIB_DIR = os.path.dirname(_METADASH_DIR)
PROD_CONFIG_DIR = os.path.join(_METALIB_DIR, 'config', 'prod')
LOCAL_CONFIG_DIR = os.path.join(_METALIB_DIR, 'config', 'local')


def list_prod_strategies():
    if not os.path.exists(PROD_CONFIG_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(PROD_CONFIG_DIR) if f.endswith('.yaml'))


def is_local_override(strategy_name):
    return os.path.exists(os.path.join(LOCAL_CONFIG_DIR, f'{strategy_name}.yaml'))


def load_config(strategy_name):
    """Load config preferring local/ override over prod/."""
    local = os.path.join(LOCAL_CONFIG_DIR, f'{strategy_name}.yaml')
    prod = os.path.join(PROD_CONFIG_DIR, f'{strategy_name}.yaml')
    path = local if os.path.exists(local) else prod
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def save_local_config(strategy_name, config_dict):
    os.makedirs(LOCAL_CONFIG_DIR, exist_ok=True)
    path = os.path.join(LOCAL_CONFIG_DIR, f'{strategy_name}.yaml')
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True, allow_unicode=True)


def parse_field_value(param, value):
    """Convert a form field string value back to the appropriate Python type."""
    if value is None:
        return None

    if param == 'symbols':
        if isinstance(value, list):
            return value
        parts = [s.strip() for s in str(value).split(',') if s.strip()]
        return parts if parts else []

    if param == 'active_hours':
        sv = str(value).strip().lower() if value is not None else ''
        if not sv or sv in ('none', 'null'):
            return None
        try:
            return [int(h.strip()) for h in sv.split(',') if h.strip()]
        except ValueError:
            return None

    if param == 'timeframe':
        return str(value) if value is not None else None

    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        v = value.strip()
        if not v or v.lower() in ('none', 'null'):
            return None
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    return value
