from numba import njit, prange
import numpy as np
import pandas as pd
import statsmodels.api as sm
from metalib.fastfinance import sma, atr, rsi

@njit
def convolve(data, kernel):
    """
    Convolution 1D Array
    :type data: np.ndarray
    :type kernel: np.ndarray
    :rtype: np.ndarray
    """
    size_data = len(data)
    size_kernel = len(kernel)
    size_out = size_data - size_kernel + 1
    out = np.array([np.nan] * size_out)
    kernel = np.flip(kernel)
    for i in range(size_out):
        window = data[i:i + size_kernel]
        out[i] = sum([window[j] * kernel[j] for j in range(size_kernel)])
    return out


@njit
def ewma(data, period, alpha=1.0):
    """
    Exponential Weighted Moving Average
    :type data: np.ndarray
    :type period: int
    :type alpha: float
    :rtype: np.ndarray
    """
    weights = (1 - alpha) ** np.arange(period)
    weights /= np.sum(weights)
    out = convolve(data, weights)
    return np.concatenate((np.array([np.nan] * (len(data) - len(out))), out))


def ewma_sets(data):
    sets = np.zeros((len(data), 9))
    i = 0

    period = [5, 30, 100]
    alpha = [0.1, 0.5, 0.9]

    for k in range(3):
        for j in range(3):
            sets[:, i] = ewma(data, period[k], alpha[j])
            i += 1

    return sets

def rsi_compute(df, period=14):
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df['rsi'] = rsi

    return df

def ewa_compute(df, spans=[10, 20, 50]):
    for span in spans:
        ewa_column = f'ewa_{span}'
        df[ewa_column] = df['close'].ewm(span=span).mean()

    return df


@njit(cache=True)
def fit_ou_process_nb(ts):
    n = len(ts)
    delta_ts = ts[1:] - ts[:-1]
    ts_lagged = ts[:-1]

    # Calculate the mean of lagged series
    mean_ts_lagged = np.mean(ts_lagged)

    # Calculate phi (speed of reversion)
    num = np.sum((ts_lagged - mean_ts_lagged) * delta_ts)
    den = np.sum((ts_lagged - mean_ts_lagged) ** 2)

    if den == 0.:
        return -1

    phi = - num / den

    return phi


@njit(cache=True)
def calculate_half_life_nb(ts):
    # Fit OU process
    phi = fit_ou_process_nb(ts)

    # Calculate half-life
    if phi <= 0:
        return np.nan

    half_life = np.log(2) / phi
    return half_life


def bollinger_bands_compute(df, period=20, k=2):
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()

    upper_band = sma + (std * k)
    lower_band = sma - (std * k)

    df['bollinger_upper'] = upper_band
    df['bollinger_lower'] = lower_band

    return df


def pivot_points_compute(df):
    pivot = (df['high'] + df['low'] + df['close']) / 3
    r1 = (2 * pivot) - df['low']
    s1 = (2 * pivot) - df['high']

    df['pivot'] = pivot
    df['r1'] = r1
    df['s1'] = s1

    return df


def macd_compute1(df):
    # print(df)
    df.loc[:, 'return'] = df.close - df.open
    df['return_sq'] = df['return'].apply(np.square)
    mean_return = df['close'].mean()
    std_return = df['close'].std()
    df['Normalized_Close'] = (df['close'] - mean_return) / std_return
    # df['Rolling_Diff'] = df['signal_line'].diff(periods=1)
    fast_period = 12
    slow_period = 26
    signal_period = 9
    df['ema_fast'] = df['Normalized_Close'].ewm(span=fast_period).mean()
    df['ema_slow'] = df['Normalized_Close'].ewm(span=slow_period).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['signal_line'] = df['macd'].ewm(span=signal_period).mean()
    df['histogram'] = df['macd'] - df['signal_line']
    # print(f"Dataframe in 5min : {df}")
    # Normalize returns using z-score
    # self.data= df
    return df


@njit(cache=True)
def retrieve_number_of_crossings_nb(prices_arr):
    prices_arr_dem = prices_arr - np.mean(prices_arr)
    return np.sum(prices_arr_dem[1:] * prices_arr_dem[:-1] < 0.)

@njit(cache=True)
def skewness_nb(ts_arr):
    demeaned_ts = (ts_arr - np.mean(ts_arr))/np.std(ts_arr)
    return np.mean(np.power(demeaned_ts, 3))

@njit(cache=True)
def kurtosis_nb(ts_arr):
    demeaned_ts = (ts_arr - np.mean(ts_arr))/np.std(ts_arr)
    return np.mean(np.power(demeaned_ts, 4))

@njit(cache=True)
def ols_tval_nb(prices_arr):
    n = prices_arr.size
    X = np.vstack((np.ones(n), np.arange(n))).T  # Design matrix with intercept and slope
    y = prices_arr.reshape(-1, 1)

    XTX_inv = np.linalg.inv(X.T @ X)
    betas = XTX_inv @ (X.T @ y)

    y_hat = X @ betas
    residuals = y - y_hat

    sigma_squared = (residuals.T @ residuals) / (n - 2)
    se_betas = np.sqrt(np.diag(sigma_squared * XTX_inv))
    t_value = betas[1] / se_betas[1]

    return t_value.item()

@njit(cache=True)
def manual_cov(X, Y, window_size):
    cov = np.sum(X * Y) / (window_size - 1)
    return cov

def get_session(hour):
    asian_session = (0, 8)  # 00:00 to 08:00 UTC
    london_session = (8, 13)  # 08:00 to 16:00 UTC
    ny_session = (13, 24) # 16:00 to 24:00 UTC

    if asian_session[0] <= hour < asian_session[1]:
        return 0  # Asian session
    elif london_session[0] <= hour < london_session[1]:
        return 1  # London session
    elif ny_session[0] <= hour < ny_session[1]:
        return 2  # New York session
    else:
        return -1

@njit(parallel=True, cache=True)
def rolling_covariance_nb(returns, window_size):
    n_periods, n_assets = returns.shape
    # Initialize the array for covariance matrices with the same "length" as input but filled with NaN
    cov_matrices = np.full((n_periods, n_assets, n_assets), np.nan)

    # Compute covariance matrices starting from when the first full window is available
    for i in prange(window_size - 1, n_periods):
        # Adjust the indices for rolling window
        start_index = i - window_size + 1
        for j in prange(n_assets):
            for k in prange(n_assets):
                if j <= k:  # To ensure symmetry, compute only for j <= k
                    window_j = returns[start_index:i + 1, j] - np.mean(returns[start_index:i + 1, j])
                    window_k = returns[start_index:i + 1, k] - np.mean(returns[start_index:i + 1, k])
                    cov = manual_cov(window_j, window_k, window_size)
                    cov_matrices[i, j, k] = cov
                    cov_matrices[i, k, j] = cov  # Symmetry due to covariance properties
    return cov_matrices

@njit(parallel=True)
def apply_to_3d_array(arr, func, max_output_size):
    n_slices, n_rows, _ = arr.shape
    result = np.empty((n_slices, max_output_size))

    # loop over the first dim
    for i in prange(n_slices):
        # extract the 2D slice
        slice_ = arr[i, :, :]

        # apply the function of R2 matrix to the slice
        output = func(slice_)

        # store the output in the pre-allocated array, handle size mismatch
        output_size = min(len(output), max_output_size)
        result[i, :output_size] = output[:output_size]

        if output_size < max_output_size:
            result[i, output_size:] = np.nan

    return result

@njit(cache=True)
def compute_stats(coefficients):
    result = np.zeros(11)

    # Pre-compute common statistics
    result[0] += np.mean(coefficients)
    result[1] += np.std(coefficients)
    result[2] += np.min(coefficients)
    result[3] += np.max(coefficients)
    result[4] += np.median(coefficients)

    # Compute quantiles
    result[5] += np.quantile(coefficients, 0.01)
    result[6] += np.quantile(coefficients, 0.99)
    result[7] += np.quantile(coefficients, 0.1)
    result[8] += np.quantile(coefficients, 0.9)

    # Calculate additional features
    result[9] += result[6] - result[8]
    result[10] += result[7] - result[5]

    return result


@njit(cache=True)
def compute_eigen_features_nb(model_corr, T=252):
    # Check for NaN in model_corr
    if np.any(np.isnan(model_corr)):
        return np.full(7, np.nan)  # Adjust the number 7 based on the number of computed features

    # Ensure model_corr is a square matrix
    assert model_corr.shape[0] == model_corr.shape[1]
    N = model_corr.shape[0]

    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(model_corr)

    # Sort eigenvalues and eigenvectors
    permutation = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[permutation]
    eigenvecs = eigenvecs[:, permutation]

    # Adjust the principal factor vector sign
    pf_vector = eigenvecs[:, 0]
    if np.sum(pf_vector < 0) > np.sum(pf_vector > 0):
        pf_vector = -pf_vector

    # Compute features
    result = np.zeros(7)  # Adjust the size based on the number of features
    result[0] = eigenvals[0] / np.sum(eigenvals)  # varex_eig1
    result[1] = np.sum(eigenvals[:5]) / np.sum(eigenvals)  # varex_eig_top5
    result[2] = np.sum(eigenvals[:30]) / np.sum(eigenvals)  # varex_eig_top30
    result[3] = result[1] - result[0]  # varex_5-1
    result[4] = result[2] - result[1]  # varex_30-5

    # Marcenko-Pastur (RMT)
    MP_cutoff = (1 + np.sqrt(N / T)) ** 2
    result[5] = np.sum(eigenvals[eigenvals > MP_cutoff]) / np.sum(eigenvals)  # varex_eig_MP

    # Determinant
    result[6] = np.prod(eigenvals[eigenvals > 0])  # determinant, considering only positive eigenvals for stability

    return result

@njit
def cmma_nb(ohlc, period, atr_period=30):
    """
    Calculate the Close minus Moving Average of a price series.
    """
    # Calculate the moving average
    ma = sma(ohlc[:, 3], period)
    # Calculate the ATR
    atr_ = atr(ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], atr_period)

    return (ohlc[:, 3] - ma) / (atr_ * period ** 0.5)

@njit(cache=True)
def ols_r_squared_nb(prices_arr):
    n = prices_arr.size
    X = np.vstack((np.ones(n), np.arange(n))).T  # Design matrix with intercept and slope
    y = prices_arr.reshape(-1, 1)

    XTX_inv = np.linalg.inv(X.T @ X)
    betas = XTX_inv @ (X.T @ y)

    y_hat = X @ betas
    residuals = y - y_hat

    ss_res = np.sum(residuals**2)  # Sum of squared residuals
    ss_tot = np.sum((y - np.mean(y))**2)  # Total sum of squares
    
    if ss_tot !=0:
        r_squared = 1 - (ss_res / ss_tot)
    else:
        return np.nan

    return r_squared.item()

@njit(cache=True)
def ols_resid_std_nb(prices_arr):
    n = prices_arr.size
    X = np.vstack((np.ones(n), np.arange(n))).T  # Design matrix with intercept and slope
    y = prices_arr.reshape(-1, 1)

    XTX_inv = np.linalg.inv(X.T @ X)
    betas = XTX_inv @ (X.T @ y)

    y_hat = X @ betas
    residuals = y - y_hat

    return np.std(residuals)

@njit
def is_valid_fvg(arr):
    """
    Check if the array is a valid FVG pattern.
    1. The high of the first candle is lower than the low of the third candle.
    2. The close of the last candle is in the gap between the high of the first candle and the low of the third candle.
    3. The second candle is bullish and the third candle is bearish.
    4. The gap was left untouched until the last candle.
    """

    if len(arr) < 4:
        return 0
    
    # Check if the high of the first candle is higher than the low of the third candle
    if arr[0, 1] > arr[2, 2]:
        return 0
    
    # Check if the gap was left untouched until the last candle
    if np.any(arr[3:-1, 2] < arr[2, 2]):
        return 0

    # Check if the close of the last candle is in the gap between the high of the first candle and the low of the third candle
    if not (arr[-1, 2] < arr[2, 2] and arr[-1, 2] > arr[0, 1]):
        return 0
    
    # Check if the second candle is bullish and the third candle is bearish
    if arr[1, 3] < arr[1, 0]: 
        return 0
    
    gap = arr[2, 2] - arr[0, 1]

    return gap

## MetaGO indicators

# Function to find the open of the second Monday of the month
def get_second_monday_open_ffill(ohlc_df, index_to_refill):
    """
    Returns a Pandas Series containing the 'open' prices of the second Monday of each month,
    forward-filled to match the original DataFrame's index.

    Parameters:
        ohlc_df (pd.DataFrame): Daily OHLC DataFrame with a datetime index and an 'open' column.

    Returns:
        pd.Series: Series with the same index as ohlc_df, containing forward-filled second Monday 'open' prices.
    """
    # Ensure the index is a datetime index
    if not isinstance(ohlc_df.index, pd.DatetimeIndex):
        ohlc_df.index = pd.to_datetime(ohlc_df.index)

    # Extract Mondays
    mondays = ohlc_df[ohlc_df.index.weekday == 0]  # 0 represents Monday

    # Group by year and month to find the second Monday
    second_mondays = mondays.groupby([mondays.index.year, mondays.index.month]).nth(1)  # nth(1) is the second Monday

    # Create a Series for second Monday opens
    second_monday_series = second_mondays['open']

    # Reindex to match the original DataFrame and forward-fill missing values
    second_monday_ffill = second_monday_series.reindex(index_to_refill, method='ffill')

    return second_monday_ffill


def get_first_monday_of_april_open_ffill(ohlc_df, index_to_refill):
    """
    Returns a Pandas Series containing the 'open' prices of the first Monday
    in April for each year, forward-filled to match the given index.

    Parameters:
        ohlc_df (pd.DataFrame): A daily (or intraday) DataFrame with a DateTime index and an 'open' column.
        index_to_refill (pd.Index): Index to which the returned Series will be reindexed and then forward-filled.

    Returns:
        pd.Series: A Series (same index as index_to_refill) with forward-filled
                   first Monday of April 'open' prices.
    """
    # Ensure the index is a datetime index
    if not isinstance(ohlc_df.index, pd.DatetimeIndex):
        ohlc_df.index = pd.to_datetime(ohlc_df.index)

    # Filter for Mondays in April
    april_mondays = ohlc_df[
        (ohlc_df.index.month == 4) & (ohlc_df.index.weekday == 0)
        ]

    # Group by year to find the first Monday in April for each year
    first_mondays_april = april_mondays.groupby(april_mondays.index.year).nth(0)

    # Extract the 'open' column
    first_april_open = first_mondays_april['open']

    # Reindex to match the target index, then forward-fill
    first_april_open_ffill = first_april_open.reindex(index_to_refill, method='ffill')

    return first_april_open_ffill


def get_last_monday_6pm_open_ffill(ohlc_df, index_to_refill):
    """
    Returns a Pandas Series containing the 'open' prices of the last Monday at 6 PM
    for each month, forward-filled to match the given index.

    Parameters:
        ohlc_df (pd.DataFrame): An intraday DataFrame with a DateTime index (including hour)
                                and an 'open' column.
        index_to_refill (pd.Index): Index to which the returned Series will be reindexed and then forward-filled.

    Returns:
        pd.Series: A Series (same index as index_to_refill) with the forward-filled
                   last Monday 6 PM 'open' prices in each month.
    """

    # Ensure the index is a datetime index
    if not isinstance(ohlc_df.index, pd.DatetimeIndex):
        ohlc_df.index = pd.to_datetime(ohlc_df.index)

    # Filter to Mondays at 6 PM (18:00)
    monday_6pm = ohlc_df[
        (ohlc_df.index.weekday == 0) & (ohlc_df.index.hour == 23)
        ]

    # Group by year-month and select the last entry for each group
    last_monday_6pm = monday_6pm.groupby([
        monday_6pm.index.year,
        monday_6pm.index.month
    ]).tail(1)

    # Extract the 'open' column
    last_monday_6pm_open = last_monday_6pm['open']

    # Reindex to match the target index, then forward-fill
    last_monday_6pm_open_ffill = last_monday_6pm_open.reindex(index_to_refill, method='ffill')
    return last_monday_6pm_open_ffill

def get_previous_levels(static_series):
    changes = static_series.ne(static_series.shift())
    change_dates = static_series.loc[changes].index

    prev_level = pd.Series(index=static_series.index, dtype=float)
    second_prev_level = pd.Series(index=static_series.index, dtype=float)
    third_prev_level = pd.Series(index=static_series.index, dtype=float)
    fourth_prev_level = pd.Series(index=static_series.index, dtype=float)

    for date in static_series.index:
        past_changes = change_dates[change_dates < date]

        if len(past_changes) > 0:
            prev_level[date] = static_series[past_changes[-1]]
            if len(past_changes) > 1:
                second_prev_level[date] = static_series[past_changes[-2]]
            if len(past_changes) > 2:
                third_prev_level[date] = static_series[past_changes[-3]]
            if len(past_changes) > 3:
                fourth_prev_level[date] = static_series[past_changes[-4]]

    return prev_level, second_prev_level, third_prev_level, fourth_prev_level

