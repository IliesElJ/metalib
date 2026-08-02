"""
"Back to basics" MetaFVG variant: keeps the HTF FVG zone detection + direction
call exactly as-is (now on a finer HTF -- 1h instead of 4h, to raise signal
frequency), but replaces the LTF momentum-FVG-pattern confirmation entirely
with a walk-forward, statistically-gated rolling OLS bias on the LTF (M15)
close series -- direction is only acted on when the rolling slope is
statistically significant (|t-stat| >= threshold), not just same-window-
positive. Entry is immediate market entry (no pending limit order / EOD
expiry -- there's no LTF gap price left to place a limit at once the
momentum pattern is gone).

Risk management is otherwise unchanged from the validated engine: SL = last
confirmed swing pivot (same PIVOT_WINDOW=7 causal, trailing-window pivot
detector used everywhere else in this codebase), TP = ATR-scaled target.

Nothing here touches production metafvg.py or metafvg_backtest.py -- separate,
explicitly experimental fork, same convention as metafvg_variants.py.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit

warnings.filterwarnings("ignore")

from metalib.fastfinance import atr as _atr_fn
from metalib.indicators import retrieve_high_pivot_point, retrieve_low_pivot_point
from metalib.metafvg import MetaFVG, HTFFVGResult
from metalib.metafvg_backtest import BacktestParams, Trade


@dataclass
class BasicsParams(BacktestParams):
    ltf_bias_window: int = 20
    ltf_bias_tstat_threshold: float = 1.0


@njit(cache=True)
def _rolling_slope_tstat(close: np.ndarray, window: int):
    """
    Same closed-form (sum-based, no matrix ops -- BLAS-safe) rolling OLS as
    _rolling_slope_r2 in metafvg_variants.py, extended with the slope's
    t-statistic: t = slope / SE(slope), SE(slope) = sqrt(resid_var / ss_xx).
    Purely descriptive same-window R^2 answers "does this window fit a line
    well"; the t-stat instead answers "is this slope distinguishable from
    zero given its own uncertainty" -- the statistical-significance framing
    requested for the LTF bias gate.
    """
    n = len(close)
    slopes = np.full(n, np.nan)
    tstats = np.full(n, np.nan)
    if window < 4:  # need window-2 > 1 degrees of freedom for a sane SE
        return slopes, tstats

    x = np.arange(window).astype(np.float64)
    x_mean = x.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx <= 0:
        return slopes, tstats
    dof = window - 2

    for i in range(window - 1, n):
        y = close[i - window + 1: i + 1]
        y_mean = y.mean()
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        slope = ss_xy / ss_xx
        y_pred = y_mean + slope * (x - x_mean)
        ss_res = np.sum((y - y_pred) ** 2)
        resid_var = ss_res / dof
        if resid_var <= 0:
            slopes[i] = slope
            continue
        se_slope = np.sqrt(resid_var / ss_xx)
        slopes[i] = slope
        tstats[i] = slope / se_slope if se_slope > 0 else np.nan
    return slopes, tstats


def _rolling_pivots(ohlc: pd.DataFrame, pivot_window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precomputed, forward-filled "last confirmed pivot as of bar i" arrays --
    vectorized equivalent of MetaFVG._retrieve_last_pivots's
    .rolling(window).apply(...).dropna().iloc[-1] pattern, but producing a
    full per-bar array instead of a single "last" value (calling the
    original per-bar in a loop would be O(n^2) on the LTF's much larger bar
    count now that it's the primary decision timeframe). retrieve_low/high_
    pivot_point confirm a swing point at the *trailing* window's midpoint --
    pandas' default (non-centered) .rolling() only ever sees bars up to and
    including i, so this is causal by construction (verified by reading the
    function: mid_point = window//2+1 lands 2 bars behind i, using only
    already-elapsed bars to confirm it -- not a lookahead risk).
    """
    low_pivot = ohlc["low"].rolling(pivot_window).apply(retrieve_low_pivot_point, engine="numba", raw=True).ffill()
    high_pivot = ohlc["high"].rolling(pivot_window).apply(retrieve_high_pivot_point, engine="numba", raw=True).ffill()
    return low_pivot.values, high_pivot.values


def simulate_fvg_trades_basics(
    ltf_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    params: BasicsParams,
) -> Tuple[List[Trade], Optional[HTFFVGResult], Optional[HTFFVGResult]]:
    strat = MetaFVG(
        symbols=[params.symbol],
        timeframe=params.ltf_timeframe,
        size_position=1.0,
        tag="backtest",
        limit_number_position=params.limit_number_position,
    )
    strat.risk_reward = params.risk_reward
    strat.atr_sensitivity = params.atr_sensitivity
    strat.htf_fill_pct = params.htf_fill_pct
    strat.max_htf_number_crossings = params.max_htf_number_crossings
    strat.ATR_PERIOD = params.atr_period
    strat.PIVOT_WINDOW = params.pivot_window
    strat.debug = True
    strat._log = lambda *a, **k: None

    n = len(ltf_df)
    closes = ltf_df["close"].values
    highs = ltf_df["high"].values
    lows = ltf_df["low"].values
    opens = ltf_df["open"].values
    times = ltf_df.index

    slopes, tstats = _rolling_slope_tstat(closes, params.ltf_bias_window)
    ltf_atr = _atr_fn(opens, highs, lows, params.atr_period)
    pivot_low, pivot_high = _rolling_pivots(ltf_df, params.pivot_window)

    bullish_result: Optional[HTFFVGResult] = None
    bearish_result: Optional[HTFFVGResult] = None
    last_n_htf = 0

    open_trades: List[Trade] = []
    closed_trades: List[Trade] = []

    warmup = max(params.ltf_bias_window, params.pivot_window, params.atr_period) + 1

    for i in range(n):
        ts = times[i]
        bar_high = highs[i]
        bar_low = lows[i]

        # 1) manage open trades: fixed SL/TP, same mechanics as the validated engine
        still_open = []
        for tr in open_trades:
            if tr.direction == 1:
                sl_hit = bar_low <= tr.sl
                tp_hit = bar_high >= tr.tp
            else:
                sl_hit = bar_high >= tr.sl
                tp_hit = bar_low <= tr.tp

            if sl_hit:
                tr.exit_time = ts
                tr.exit_price = tr.sl
                tr.exit_reason = "sl"
                tr.status = "closed"
                closed_trades.append(tr)
            elif tp_hit:
                tr.exit_time = ts
                tr.exit_price = tr.tp
                tr.exit_reason = "tp"
                tr.status = "closed"
                closed_trades.append(tr)
            else:
                still_open.append(tr)
        open_trades = still_open

        # 2) recompute HTF FVG zones once a new HTF bar is available
        avail_htf = htf_df[htf_df.index <= ts]
        if len(avail_htf) >= 3 and len(avail_htf) != last_n_htf:
            avail_htf_w = avail_htf.iloc[-params.htf_lookback_bars:]
            bullish_patterns, bearish_patterns = strat.detect_fvg_htf(avail_htf_w)
            bullish_result = strat._process_htf_fvg_patterns(
                bullish_patterns, avail_htf_w["low"], strat.htf_fill_pct, "Bullish"
            )
            bearish_result = strat._process_htf_fvg_patterns(
                bearish_patterns, avail_htf_w["high"], 1 - strat.htf_fill_pct, "Bearish"
            )
            last_n_htf = len(avail_htf)

        # 3) look for a new market entry: HTF zone direction + statistically
        # significant LTF OLS bias agreeing with it
        if i >= warmup and len(open_trades) < params.limit_number_position and bullish_result is not None:
            last_price = closes[i]
            direction = strat._determine_direction(
                last_price, bullish_result.filtered_patterns, bearish_result.filtered_patterns
            )
            if direction is not None:
                is_bullish = direction == 1
                tstat = tstats[i]
                slope = slopes[i]
                bias_ok = (
                    not np.isnan(tstat)
                    and abs(tstat) >= params.ltf_bias_tstat_threshold
                    and (slope > 0) == is_bullish
                )
                if bias_ok:
                    sl = pivot_low[i] if is_bullish else pivot_high[i]
                    atr_value = ltf_atr[i]
                    # sl must land on the loss-making side of entry -- unlike the
                    # original design (entry = a momentum-FVG gap boundary, which
                    # by construction of that pattern sits past the recent swing),
                    # entry here is just the current market price, so there's no
                    # implicit guarantee the last confirmed pivot is still on the
                    # correct side. Without this check ~20% of "sl" exits were
                    # silently profitable (stop above entry on a short, etc.) --
                    # caught empirically, not theoretically, before trusting any
                    # results from this engine.
                    sl_valid = (is_bullish and sl < last_price) or (not is_bullish and sl > last_price)
                    if not (np.isnan(sl) or np.isnan(atr_value)) and sl_valid:
                        direction_mult = 1 if is_bullish else -1
                        tp = last_price + atr_value * params.atr_sensitivity * params.risk_reward * direction_mult
                        open_trades.append(
                            Trade(
                                direction=direction_mult,
                                created_time=ts,
                                entry_time=ts,
                                entry=last_price,
                                sl=sl,
                                tp=tp,
                                status="open",
                                exit_reason="filled",
                            )
                        )

    all_trades = closed_trades + open_trades
    all_trades.sort(key=lambda t: t.created_time)
    return all_trades, bullish_result, bearish_result
