"""
Constants and configuration paths for the metalib project.

This module centralizes all path definitions and constants used across the project
to make maintenance easier and prevent duplication.
"""

import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = ROOT_DIR / "config"
LOGS_DIR = ROOT_DIR / "logs"
STORE_DIR = ROOT_DIR / "store"
DATA_DIR = ROOT_DIR / "data"
INDICATORS_DIR = ROOT_DIR / "indicators"
CHARTS_DIR = ROOT_DIR / "charts"


# Configuration environments
CONFIG_DEV = CONFIG_DIR / "dev"
CONFIG_PROD = CONFIG_DIR / "prod"

# Default configuration path (can be overridden at runtime)
DEFAULT_CONFIG_PATH = CONFIG_DEV

# Database paths
DATABASE_PRICE = STORE_DIR / "price"
DATABASE_SIGNAL = STORE_DIR / "signals"

# File extensions
HDF5_EXTENSION = ".hdf5"
CSV_EXTENSION = ".csv"
YAML_EXTENSION = ".yaml"
LOG_EXTENSION = ".log"

# File paths
SIGNALS_FILE = f"{DATABASE_SIGNAL}{HDF5_EXTENSION}"
PRICE_FILE = f"{DATABASE_PRICE}{HDF5_EXTENSION}"

# MetaTrader constants
MT5_ORDER_TYPE_BUY = 0
MT5_ORDER_TYPE_SELL = 1
MT5_ORDER_TYPE_BUY_LIMIT = 2
MT5_ORDER_TYPE_SELL_LIMIT = 3
MT5_ORDER_TYPE_BUY_STOP = 4
MT5_ORDER_TYPE_SELL_STOP = 5

# API related constants
API_HOST = "0.0.0.0"
API_PORT = 8000

# Strategy state constants
STATE_NEUTRAL = 0
STATE_LONG = 1
STATE_SHORT = -1
STATE_EXIT = -2

# Ensure directories exist
def ensure_directories():
    """Create all necessary directories if they don't exist."""
    for directory in [LOGS_DIR, STORE_DIR, DATA_DIR, INDICATORS_DIR, CHARTS_DIR]:
        directory.mkdir(exist_ok=True, parents=True)
