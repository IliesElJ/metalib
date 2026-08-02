"""
Standalone backtest engine for MetaOB (metalib/metaob.py).

Reuses the indicator functions from metalib/indicators.py verbatim (the same
rolling_min_shift1_nb / rolling_max_shift1_nb / true_range_nb / order_blocks_nb
/ crosses_nb / ols_tval_nb MetaOB.calculate_indicators calls live), so signal
logic never drifts from production. Unlike metafvg_backtest.py, MetaOB places
market orders (not pending limit orders) with no fill-scan phase - it's
OB+pivot signal on bar i-1's close data -> entry at bar i's open.

Trend filter is selectable via BacktestParams.trend_filter:
  - "none":       no trend gate (raw OB+pivot edge)
  - "old_sharpe": the original (buggy) filter this backtest validates the fix
                  against - rolling Sharpe of LTF returns over sma_long_hours
                  bars (~2 days at M15, despite the "_hours" name), thresholded
                  against the 25th/75th percentile of its own trailing-66-day
                  distribution, recalibrated once per calendar day (mirrors the
                  old MetaOB.fit(), which metaworker.py rescheduled daily).
  - "new":        the fix - rolling OLS t-stat of trend_timeframe (D1) closes
                  over trend_window bars, same as MetaOB.compute_trend_t().

Data: LTF (M15 by default) candles for entries, plus a separate higher
timeframe (D1) pull for the "new" trend regression - MetaOB.compute_trend_t()
does this live with one MT5 call per signal cycle; here the whole D1 history
is fetched once and the trend t-stat computed as a rolling series, then
aligned back onto the LTF timeline point-in-time correctly (a D1 bar's
regression output only becomes "known" starting the following calendar day -
this backtest uses only fully-closed D1 bars, slightly more conservative than
live, which can also see the still-forming current day's bar).

Reuses metafvg_backtest.py's Trade / trades_to_dataframe / build_vbt_portfolio
/ connect_mt5 / _fetch_ltf_rates - none of that machinery is FVG-specific.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import MetaTrader5 as mt5
import vectorbt as vbt

from metalib.indicators import (
    rolling_mean_nb,
    rolling_min_shift1_nb,
    rolling_max_shift1_nb,
    true_range_nb,
    order_blocks_nb,
    crosses_nb,
    ols_tval_nb,
)
from metalib.metafvg_backtest import (
    Trade,
    trades_to_dataframe,
    build_vbt_portfolio,
    connect_mt5,
    _fetch_ltf_rates,
)

LTF_TIMEFRAME = mt5.TIMEFRAME_M15
TREND_TIMEFRAME = mt5.TIMEFRAME_D1


@dataclass
class BacktestParams:
    symbol: str
    start: datetime
    end: datetime
    pivot_window: int = 60
    breakout_lookback: int = 3
    sma_long_hours: int = 200  # only used by "old_sharpe" (its rolling-Sharpe window)
    atr_period: int = 14
    sl_atr_mult: float = 5.0
    tp_atr_mult: float = 10.0
    trend_window: int = 200
    trend_t_threshold: float = 2.0
    ltf_timeframe: int = LTF_TIMEFRAME
    trend_timeframe: int = TREND_TIMEFRAME
    trend_filter: str = "new"  # "none" | "old_sharpe" | "new"
    # MetaOB's own signals() has its are_positions_with_tag_open guard commented
    # out - live has no concurrency cap, a position opens on every bar the signal
    # holds. Set generously high so vbt slot assignment never blocks an entry
    # (it auto-expands past this anyway - see _assign_slots).
    limit_number_position: int = 30


# =========================================================================
# Data fetching
# =========================================================================

def fetch_candles(
    symbol: str, start: datetime, end: datetime,
    ltf_timeframe: int, trend_timeframe: int, trend_window: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch LTF (entry) candles plus enough higher-timeframe history before
    `start` for the trend regression's warm-up window."""
    ltf_df = _fetch_ltf_rates(symbol, ltf_timeframe, start, end)[["open", "high", "low", "close"]]

    trend_start = start - timedelta(days=int(trend_window * 1.7) + 20)
    trend_df = _fetch_ltf_rates(symbol, trend_timeframe, trend_start, end)[["close"]]

    return ltf_df, trend_df


# =========================================================================
# Trend filters
# =========================================================================

def _asof_align(value_series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Point-in-time forward-fill of a sparse series onto target_index - only
    ever uses values whose own timestamp is <= the target timestamp, no
    lookahead."""
    src = value_series.dropna().sort_index()
    if src.empty:
        return pd.Series(0.0, index=target_index)
    # MT5 rates decode to second resolution while pd.Timedelta-shifted index arithmetic
    # can promote to microsecond resolution -- merge_asof requires matching dtypes.
    left = pd.DataFrame({"time": pd.DatetimeIndex(target_index).astype("datetime64[ns]")})
    right = pd.DataFrame({"time": pd.DatetimeIndex(src.index).astype("datetime64[ns]"), "val": src.to_numpy()})
    merged = pd.merge_asof(left.sort_values("time"), right.sort_values("time"), on="time", direction="backward")
    merged = merged.set_index("time").reindex(pd.DatetimeIndex(target_index).astype("datetime64[ns]"))
    return pd.Series(merged["val"].fillna(0.0).to_numpy(), index=target_index)


def compute_new_trend_t(trend_df: pd.DataFrame, ltf_index: pd.DatetimeIndex, trend_window: int) -> pd.Series:
    trend_t = trend_df["close"].rolling(trend_window).apply(ols_tval_nb, engine="numba", raw=True)
    available = trend_t.dropna().copy()
    available.index = available.index + pd.Timedelta(days=1)
    return _asof_align(available, ltf_index)


def compute_old_sharpe_trend(ltf_close: pd.Series, sma_long_hours: int) -> Tuple[pd.Series, pd.Series]:
    rets = ltf_close.pct_change()
    rolling_sh = rets.rolling(sma_long_hours).apply(
        lambda x: np.mean(x) / np.std(x) if np.std(x) > 0 else np.nan, raw=True
    )

    thr_long = pd.Series(np.nan, index=ltf_close.index)
    thr_short = pd.Series(np.nan, index=ltf_close.index)
    for day in ltf_close.index.normalize().unique():
        lookback = rolling_sh[(rolling_sh.index >= day - pd.Timedelta(days=66)) & (rolling_sh.index < day)].dropna()
        if len(lookback) < 30:
            continue
        mask = (ltf_close.index >= day) & (ltf_close.index < day + pd.Timedelta(days=1))
        thr_long[mask] = lookback.quantile(0.75)
        thr_short[mask] = lookback.quantile(0.25)
    thr_long = thr_long.ffill()
    thr_short = thr_short.ffill()

    uptrend = (rolling_sh > thr_long).fillna(False)
    downtrend = (rolling_sh < thr_short).fillna(False)
    return uptrend, downtrend


# =========================================================================
# Indicators / signals (vectorized replica of MetaOB.calculate_indicators)
# =========================================================================

def compute_indicators(ltf_df: pd.DataFrame, trend_df: pd.DataFrame, params: BacktestParams) -> pd.DataFrame:
    """Every underlying indicator fn (rolling_min_shift1_nb, order_blocks_nb,
    ...) is causal/shift-based by construction (verified against
    indicators.py), so this vectorized single pass over the whole window is
    point-in-time safe without a bar-by-bar recompute loop."""
    o = ltf_df["open"].to_numpy(dtype=np.float64)
    h = ltf_df["high"].to_numpy(dtype=np.float64)
    l = ltf_df["low"].to_numpy(dtype=np.float64)
    c = ltf_df["close"].to_numpy(dtype=np.float64)

    ind = pd.DataFrame(index=ltf_df.index)
    ind["close"] = c

    w = int(params.pivot_window)
    pivot_low = rolling_min_shift1_nb(l, w)
    pivot_high = rolling_max_shift1_nb(h, w)
    ind["pivot_low"] = pivot_low
    ind["pivot_high"] = pivot_high

    tr = true_range_nb(h, l, c)
    ind["atr"] = rolling_mean_nb(tr, int(params.atr_period))

    bull_ob, bear_ob = order_blocks_nb(o, h, l, c)
    ind["bull_ob"] = bull_ob
    ind["bear_ob"] = bear_ob

    cross_below, cross_above = crosses_nb(l, h, pivot_low, pivot_high)
    ind["cross_below_pivot"] = cross_below
    ind["cross_above_pivot"] = cross_above

    if params.trend_filter == "none":
        ind["uptrend"] = True
        ind["downtrend"] = True
    elif params.trend_filter == "old_sharpe":
        uptrend, downtrend = compute_old_sharpe_trend(ltf_df["close"], params.sma_long_hours)
        ind["uptrend"] = uptrend
        ind["downtrend"] = downtrend
    elif params.trend_filter == "new":
        trend_t = compute_new_trend_t(trend_df, ltf_df.index, params.trend_window)
        ind["trend_t"] = trend_t
        ind["uptrend"] = trend_t > params.trend_t_threshold
        ind["downtrend"] = trend_t < -params.trend_t_threshold
    else:
        raise ValueError(f"Unknown trend_filter: {params.trend_filter}")

    return ind


def compute_signals(ind: pd.DataFrame, breakout_lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    cross_pivot_below = (
        ind["cross_below_pivot"].rolling(breakout_lookback).apply(lambda x: np.max(x), raw=True).astype(bool)
    )
    cross_pivot_above = (
        ind["cross_above_pivot"].rolling(breakout_lookback).apply(lambda x: np.max(x), raw=True).astype(bool)
    )
    long_signal = (cross_pivot_below & ind["bull_ob"].astype(bool) & ind["uptrend"].astype(bool)).to_numpy()
    short_signal = (cross_pivot_above & ind["bear_ob"].astype(bool) & ind["downtrend"].astype(bool)).to_numpy()
    return long_signal, short_signal


# =========================================================================
# Simulation
# =========================================================================

def simulate_ob_trades(
    ltf_df: pd.DataFrame, ind: pd.DataFrame, long_signal: np.ndarray, short_signal: np.ndarray,
    params: BacktestParams,
) -> List[Trade]:
    opens = ltf_df["open"].to_numpy(dtype=np.float64)
    highs = ltf_df["high"].to_numpy(dtype=np.float64)
    lows = ltf_df["low"].to_numpy(dtype=np.float64)
    times = ltf_df.index
    atr = ind["atr"].to_numpy(dtype=np.float64)
    close = ind["close"].to_numpy(dtype=np.float64)
    n = len(ltf_df)

    open_trades: List[Trade] = []
    closed_trades: List[Trade] = []

    for i in range(1, n):
        ts = times[i]
        bar_high = highs[i]
        bar_low = lows[i]

        still_open = []
        for tr in open_trades:
            if tr.direction == 1:
                sl_hit = bar_low <= tr.sl
                tp_hit = bar_high >= tr.tp
            else:
                sl_hit = bar_high >= tr.sl
                tp_hit = bar_low <= tr.tp
            if sl_hit:
                tr.exit_time, tr.exit_price, tr.exit_reason, tr.status = ts, tr.sl, "sl", "closed"
                closed_trades.append(tr)
            elif tp_hit:
                tr.exit_time, tr.exit_price, tr.exit_reason, tr.status = ts, tr.tp, "tp", "closed"
                closed_trades.append(tr)
            else:
                still_open.append(tr)
        open_trades = still_open

        prev = i - 1
        if np.isnan(atr[prev]) or np.isnan(close[prev]):
            continue

        if long_signal[prev]:
            sl = close[prev] - atr[prev] * params.sl_atr_mult
            tp = close[prev] + atr[prev] * params.tp_atr_mult
            open_trades.append(Trade(direction=1, created_time=times[prev], entry=opens[i], sl=sl, tp=tp,
                                      entry_time=ts, exit_reason="filled", status="open"))
        if short_signal[prev]:
            sl = close[prev] + atr[prev] * params.sl_atr_mult
            tp = close[prev] - atr[prev] * params.tp_atr_mult
            open_trades.append(Trade(direction=-1, created_time=times[prev], entry=opens[i], sl=sl, tp=tp,
                                      entry_time=ts, exit_reason="filled", status="open"))

    for tr in open_trades:
        tr.status = "open"
        tr.exit_reason = "still_open"

    all_trades = closed_trades + open_trades
    all_trades.sort(key=lambda t: t.entry_time)
    return all_trades


# =========================================================================
# Orchestrator
# =========================================================================

@dataclass
class BacktestResult:
    params: BacktestParams
    ltf_df: pd.DataFrame
    trades: List[Trade]
    trades_df: pd.DataFrame
    portfolio: "vbt.Portfolio"
    stats: pd.Series


def run_backtest_with_data(ltf_df: pd.DataFrame, trend_df: pd.DataFrame, params: BacktestParams) -> BacktestResult:
    ind = compute_indicators(ltf_df, trend_df, params)
    long_signal, short_signal = compute_signals(ind, params.breakout_lookback)
    trades = simulate_ob_trades(ltf_df, ind, long_signal, short_signal, params)
    trades_df = trades_to_dataframe(trades)

    portfolio = build_vbt_portfolio(ltf_df, trades, params.limit_number_position)
    stats = portfolio.stats()

    return BacktestResult(
        params=params, ltf_df=ltf_df, trades=trades, trades_df=trades_df,
        portfolio=portfolio, stats=stats,
    )


def run_backtest(params: BacktestParams) -> BacktestResult:
    connect_mt5()
    ltf_df, trend_df = fetch_candles(
        params.symbol, params.start, params.end,
        params.ltf_timeframe, params.trend_timeframe, params.trend_window,
    )
    return run_backtest_with_data(ltf_df, trend_df, params)


if __name__ == "__main__":
    p = BacktestParams(
        symbol="EURUSD",
        start=datetime.utcnow() - timedelta(days=365 * 2),
        end=datetime.utcnow(),
        trend_filter="new",
    )
    result = run_backtest(p)
    print(f"LTF bars: {len(result.ltf_df)}")
    print(f"Trades: {len(result.trades_df)}")
    print(result.trades_df.tail(10))
    print(result.stats)
