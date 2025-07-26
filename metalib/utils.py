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
    # Convert days to pandas date offset
    insample_offset = pd.DateOffset(days=insample_days)
    outsample_offset = pd.DateOffset(days=outsample_days)

    # Initialize lists to store in-sample and out-sample dataframes
    insample_dfs = []
    outsample_dfs = []

    # Start date is the first date in the dataframe
    start_date = df.index.min()

    while True:
        # End date is start date plus in-sample period
        end_date = start_date + insample_offset

        # Out-sample start date is end date plus one day
        outsample_start_date = end_date + pd.DateOffset(days=1)

        # Out-sample end date is out-sample start date plus out-sample period
        outsample_end_date = outsample_start_date + outsample_offset

        # Break the loop if out-sample end date is greater than the last date in the dataframe
        if outsample_end_date > df.index.max():
            break

        # Create in-sample and out-sample dataframes
        insample_df = df.loc[start_date:end_date]
        outsample_df = df.loc[outsample_start_date:outsample_end_date]

        # Append dataframes to the lists
        insample_dfs.append(insample_df)
        outsample_dfs.append(outsample_df)

        # Update start date to be the out-sample end date plus one day for the next iteration
        start_date = start_date + outsample_offset

    return insample_dfs, outsample_dfs