from numba import njit, prange
from metalib.fastfinance import *
import numpy as np
import os 
import pandas as pd
import MetaTrader5 as mt5


@njit(parallel=True)
def apply_w_diff_params_1d_nb(arr, transform, args, weights):
    """
    Apply a transformation to a 1D price array with different parameters and weights.
    :param arr: price array
    :param transform: numba function to apply (needs to have 1 argument)
    :param args: arguments to apply to the transformation
    :param weights: Different weights to dot apply to the transformed array
    :return: transformed array with different parameters and weights
    """
    out = np.zeros((arr.shape[0], args.shape[0]))
    transformed_arr = np.zeros((arr.shape[0], weights.shape[0]))

    for arg_idx in prange(args.shape[0]):
        arg = args[arg_idx]
        for col in prange(arr.shape[1]):
            transformed_arr[:, col] = transform(arr[:, col], arg)
        out[:, arg_idx] += transformed_arr.dot(weights)
        transformed_arr.fill(0)
    return out


@njit(parallel=True)
def apply_w_diff_params_2d_nb(arr, transform, args, weights):
    """
    Apply a transformation to a 2D price array with different parameters and weights.
    :param arr: ohlc (2d) array
    :param transform: numba function to apply (needs to have 1 argument)
    :param args: arguments to apply to the transformation
    :param weights: Different weights to dot apply to the transformed array
    :return: transformed array with different parameters and weights
    """
    out = np.zeros((arr[0].shape[0], args.shape[0]))
    transformed_arr = np.zeros((arr[0].shape[0], weights.shape[0]))

    for arg_idx in prange(args.shape[0]):
        arg = args[arg_idx]
        for col in prange(arr.shape[0]):
            transformed_arr[:, col] = transform(arr[col], arg)
        out[:, arg_idx] += transformed_arr @ weights
        transformed_arr.fill(0)
    return out


def common_index(list_df):
    """
    Find the common index of a list of dataframes.
    :param list_df: List of dataframes
    :return: Index
    """
    first_index = list_df[0].index
    for i in range(1, len(list_df)):
        first_index = first_index.intersection(list_df[i].index)
    return first_index


def assign_cat(val):
    """
    Assign the sign to a value.
    :param val: float
    :return: int
    """
    if val < 0.:
        return 0
    else:
        return 1

def load_hist_data(symbol, year):
    """
    Retrieve the csv in the '{symbol}' folder and check if the year is inside the csv file name.
    :param symbol: str
    :param year: int or str
    :return: DataFrame or None
    """
    try:
        year = str(year)  # Ensure year is a string
        current_directory = os.path.dirname(os.path.realpath(__file__))
        # Fix the path construction to use os.path.join exclusively
        directory = os.path.join(current_directory, "data", symbol.lower())
        colnames = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        
        for filename in os.listdir(directory):
            if year in filename and filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)  # Ensure consistent path joining
                data = pd.read_csv(filepath, names=colnames, header=None)
                data.loc[:, 'time'] = data.loc[:, 'date'] + ' ' + data.loc[:, 'time']
                data.loc[:, 'time'] = pd.to_datetime(data.loc[:, 'time'])
                return data.drop(columns=['date']).set_index('time')

        return None
    except FileNotFoundError:
        print(f"No directory found at {directory}")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_multiple_hist_data(symbols, year):
    data = {symbol:load_hist_data(symbol, year) for symbol in symbols}
    common_index_ = common_index(list(data.values()))
    return {k:v.loc[common_index_] for k, v in data.items()}
	
def clean_args(args):
    # Convert timeframe string (e.g. "TIMEFRAME_M1") to actual mt5 constant
    if isinstance(args.get("timeframe"), str):
        args["timeframe"] = eval(args["timeframe"])

    # Convert null active_hours to None
    if "active_hours" in args and args["active_hours"] is None:
        args["active_hours"] = None

    return args


def split_dataframe(df, insample_days, outsample_days):
    """
    Split dataframe into in-sample and out-sample periods, ensuring split points
    are on business days at minute=0 (start of trading hour).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    insample_days : int
        Number of business days for in-sample period
    outsample_days : int
        Number of business days for out-sample period

    Returns:
    --------
    tuple : (insample_dfs, outsample_dfs)
    """

    # Initialize lists to store in-sample and out-sample dataframes
    insample_dfs = []
    outsample_dfs = []

    # Get unique business days from the dataframe index
    business_days = pd.bdate_range(
        start=df.index.min().normalize(),
        end=df.index.max().normalize(),
        freq='B'  # Business day frequency
    )

    # Filter to only business days that actually exist in our data
    # and create timestamps at minute=0 (start of day)
    available_days = []
    for day in business_days:
        # Look for data on this day at minute=0 (or first available minute)
        day_data = df[df.index.date == day.date()]
        if not day_data.empty:
            # Get the first timestamp of the day (should be minute=0)
            first_timestamp = day_data.index.min()
            available_days.append(first_timestamp)

    # Start from the first available business day
    current_idx = 0

    while True:
        # Check if we have enough days left for both insample and outsample
        if current_idx + insample_days + outsample_days > len(available_days):
            break

        # In-sample period: from current day to current + insample_days
        insample_start = available_days[current_idx]
        insample_end_idx = current_idx + insample_days - 1
        insample_end = available_days[insample_end_idx]

        # Out-sample period: from next business day after insample to insample_end + outsample_days
        outsample_start_idx = insample_end_idx + 1
        outsample_end_idx = outsample_start_idx + outsample_days - 1

        # Check bounds
        if outsample_end_idx >= len(available_days):
            break

        outsample_start = available_days[outsample_start_idx]
        outsample_end = available_days[outsample_end_idx]

        # Create in-sample dataframe (from start of insample_start day to end of insample_end day)
        insample_end_of_day = insample_end + pd.DateOffset(days=1) - pd.Timedelta(minutes=1)
        insample_df = df.loc[insample_start:insample_end_of_day]

        # Create out-sample dataframe (from start of outsample_start day to end of outsample_end day)
        outsample_end_of_day = outsample_end + pd.DateOffset(days=1) - pd.Timedelta(minutes=1)
        outsample_df = df.loc[outsample_start:outsample_end_of_day]

        # Only add if we have data for both periods
        if not insample_df.empty and not outsample_df.empty:
            insample_dfs.append(insample_df)
            outsample_dfs.append(outsample_df)

        # Move to the next period (advance by outsample_days to avoid overlap)
        current_idx += outsample_days

    return insample_dfs, outsample_dfs

# Example usage:
# train_samples, test_samples = split_dataframe(df, insample_days=30, outsample_days=5)