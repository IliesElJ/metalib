"""
Indicator Utilities
Functions for reading indicator/signal data from HDF5 and classifying them.
"""

import pandas as pd
import os
import sys

# Add parent directory for metalib imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from metalib.constants import SIGNALS_FILE


# Indicators that overlay on the price chart (same y-axis as close price)
PRICE_LEVEL_INDICATORS = {
    "price",
    "sma_target",
    "bb_upper",
    "bb_sma",
    "bb_lower",
    "tp",
    "sl",
    "entry",
    "exit",
}

# Skip columns (metadata, not plottable)
SKIP_COLUMNS = {"timestamp", "symbol"}


def load_indicator_data(tag):
    """
    Read all rows for a tag from the signals HDF5 file.

    Args:
        tag: Strategy tag (will be sanitized for HDF5 key)

    Returns:
        pd.DataFrame or None if not found
    """
    if not os.path.exists(SIGNALS_FILE):
        return None

    key = "/" + tag.replace("-", "_").replace(".", "_")

    try:
        with pd.HDFStore(SIGNALS_FILE, mode="r") as store:
            if key not in store:
                return None
            df = store[key]
    except Exception:
        return None

    if df is not None and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def list_available_tags():
    """
    List all tag keys in the HDF5 file.

    Returns:
        List of tag strings (with leading '/' stripped)
    """
    if not os.path.exists(SIGNALS_FILE):
        return []

    try:
        with pd.HDFStore(SIGNALS_FILE, mode="r") as store:
            return [k.lstrip("/") for k in store.keys()]
    except Exception:
        return []


def classify_indicators(columns):
    """
    Split columns into overlay vs subplot groups.

    Args:
        columns: List or Index of column names

    Returns:
        dict with keys 'overlay' and 'subplot', each a list of column names
    """
    overlay = []
    subplot = []

    for col in columns:
        if col in SKIP_COLUMNS:
            continue
        if col in PRICE_LEVEL_INDICATORS:
            overlay.append(col)
        else:
            subplot.append(col)

    return {"overlay": overlay, "subplot": subplot}
