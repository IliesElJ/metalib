"""
Backtest-only A/B variants of the MetaFVG simulation, layering interpretable
statistical trend-confirmation filters and an ATR trailing-stop exit on top of
the validated baseline engine in metalib/metafvg_backtest.py. Nothing here
touches production metafvg.py or metafvg_backtest.py — this is a separate,
explicitly experimental fork for comparison purposes.

Variants:
  - "baseline"   : unmodified MetaFVG signal/exit logic (delegates to the
                   validated simulate_fvg_trades for a true apples-to-apples
                   baseline, not a re-derivation of it).
  - "adx"        : additionally requires HTF ADX(14) >= adx_threshold at the
                   moment of signal (trend-strength gate).
  - "regression" : additionally requires a rolling OLS fit of HTF closes vs.
                   bar index to have slope sign matching the signal direction
                   and R^2 >= regression_r2_threshold (clean-trend gate).
  - "spearman"   : same idea as "regression" but fully rank-based -- rolling
                   Spearman rank correlation of HTF closes vs. bar index takes
                   the place of both OLS slope (direction, via sign) and R^2
                   (cleanliness, via magnitude >= regression_r2_threshold).
                   Only requires monotonicity, not linearity, so a nonlinear
                   (accelerating/decelerating) but consistently directional
                   move isn't penalized the way it is under OLS R^2.
  - "theilsen"   : same gate as "regression" (OLS R^2 >= regression_r2_threshold)
                   but direction comes from the Theil-Sen slope (median of
                   pairwise slopes) instead of the OLS slope -- robust to a
                   single outlier/tail bar flipping the read.
  - "atr_trailing": replaces the fixed ATR take-profit with a chandelier-style
                   ATR trailing stop on the LTF (risk-management variant; can
                   be combined with either signal filter above).
  - breakeven_r_multiple: risk-management variant compatible with exit_style
                   "fixed" only (leaves the ATR target in place). Once a trade's
                   favorable excursion clears breakeven_r_multiple * original
                   risk, moves the stop to entry -- protects against a trade
                   running deep into profit and giving all of it back before
                   hitting the fixed target. Can combine with any signal filter.
  - session_filter: only allow NEW entries during [session_start_hour,
                   session_end_hour) (OHLC index hour, ~UTC) -- default 7-17
                   is roughly London open through NY afternoon, skipping the
                   thinner Asian-only window. Doesn't affect management of
                   already-pending/open trades. Composable with any trend
                   filter (applied as an extra AND on trend_ok).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from numba import njit

warnings.filterwarnings("ignore")

import vectorbt as vbt

from metalib.fastfinance import adx as _adx_fn, atr as _atr_fn
from metalib.metafvg import MetaFVG, HTFFVGResult
from metalib.metafvg_backtest import (
    BacktestParams,
    BacktestResult,
    Trade,
    _assign_slots,
    _end_of_day,
    build_vbt_portfolio,
    connect_mt5,
    fetch_ltf_htf_candles,
    simulate_fvg_trades,
    trades_to_dataframe,
)


@dataclass
class VariantParams(BacktestParams):
    trend_filter: str = "none"  # none | adx | regression | spearman | theilsen
    adx_period: int = 14
    adx_threshold: float = 20.0
    regression_window: int = 20
    regression_r2_threshold: float = 0.5
    exit_style: str = "fixed"  # fixed | atr_trailing
    trail_atr_mult: float = 3.0
    trail_atr_period: int = 14
    breakeven_r_multiple: Optional[float] = None  # fixed-exit only; None = disabled
    session_filter: bool = False  # only allow NEW entries within [session_start_hour, session_end_hour)
    session_start_hour: int = 7   # London/NY session, OHLC index hour (~UTC)
    session_end_hour: int = 17
    invert_direction: bool = False  # mean-reversion probe: require the regression/Lasso
    # sign to DISAGREE with the FVG-implied direction instead of agreeing (sign(beta_1) = 1-d
    # in the 0/1 direction encoding _determine_direction uses, i.e. -d in +-1 terms) -- fades
    # the trend-following bias reading rather than confirming it. Applies uniformly to every
    # trend_filter branch (regression, spearman, theilsen, lasso/tree).


# =========================================================================
# Rolling regression (manual closed-form OLS, no scipy.linalg -- this env's
# BLAS/LAPACK is documented-unstable for library linear-algebra calls; a
# hand-rolled sum-based fit avoids that path entirely, same workaround used
# elsewhere in this repo, e.g. dashboard.py's OU fit).
# =========================================================================
@njit(cache=True)
def _rolling_slope_r2(close: np.ndarray, window: int):
    n = len(close)
    slopes = np.full(n, np.nan)
    r2s = np.full(n, np.nan)
    if window < 2:
        return slopes, r2s

    x = np.arange(window).astype(np.float64)
    x_mean = x.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx <= 0:
        return slopes, r2s

    for i in range(window - 1, n):
        y = close[i - window + 1: i + 1]
        y_mean = y.mean()
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        slope = ss_xy / ss_xx
        y_pred = y_mean + slope * (x - x_mean)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        slopes[i] = slope
        r2s[i] = r2
    return slopes, r2s


@njit(cache=True)
def _rolling_spearman(close: np.ndarray, window: int):
    """
    Spearman rank correlation of close vs. bar index, rolling over `window`.
    Sign gives trend direction, magnitude gives trend "cleanliness" -- but
    unlike Pearson R^2 (_rolling_slope_r2), it only requires *monotonicity*,
    not linearity, so a nonlinear (accelerating/decelerating) but consistently
    directional move still scores close to +-1. No tie-correction (float
    prices essentially never tie exactly).
    """
    n = len(close)
    rho = np.full(n, np.nan)
    if window < 2:
        return rho
    denom = window * (window ** 2 - 1)
    if denom <= 0:
        return rho

    for i in range(window - 1, n):
        y = close[i - window + 1: i + 1]
        order = np.argsort(y)
        ranks = np.empty(window)
        for k in range(window):
            ranks[order[k]] = k
        d2_sum = 0.0
        for k in range(window):
            d = ranks[k] - k
            d2_sum += d * d
        rho[i] = 1.0 - 6.0 * d2_sum / denom
    return rho


@njit(cache=True)
def _rolling_theil_sen_slope(close: np.ndarray, window: int):
    """
    Theil-Sen slope: median of all pairwise slopes (close[b]-close[a])/(b-a)
    within the window. Robust analog of the OLS slope -- a single outlier bar
    (tail move) shifts the median by at most one rank instead of dominating a
    squared-error fit, so trend *direction* is less prone to being flipped by
    one large countertrend spike inside an otherwise clean trend.
    """
    n = len(close)
    slopes = np.full(n, np.nan)
    if window < 2:
        return slopes
    n_pairs = window * (window - 1) // 2

    for i in range(window - 1, n):
        y = close[i - window + 1: i + 1]
        pair_slopes = np.empty(n_pairs)
        idx = 0
        for a in range(window):
            for b in range(a + 1, window):
                pair_slopes[idx] = (y[b] - y[a]) / (b - a)
                idx += 1
        pair_slopes.sort()
        m = n_pairs // 2
        if n_pairs % 2 == 1:
            slopes[i] = pair_slopes[m]
        else:
            slopes[i] = 0.5 * (pair_slopes[m - 1] + pair_slopes[m])
    return slopes


# =========================================================================
# Variant simulation
# =========================================================================
def simulate_fvg_trades_variant(
    ltf_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    params: VariantParams,
    lasso_pred: Optional[pd.Series] = None,
    lasso_threshold: Optional[pd.Series] = None,
) -> tuple[List[Trade], Optional[HTFFVGResult], Optional[HTFFVGResult]]:
    """
    lasso_pred/lasso_threshold: precomputed once per symbol -- by
    metafvg_lasso_gate.compute_lasso_gate_series when trend_filter=="lasso",
    or metafvg_tree_gate.compute_tree_gate_series when trend_filter=="tree"
    (same (signed prediction, causal threshold) shape either way, so both
    fitted-model gates share this one lookup branch below). Passed in rather
    than computed here because a walk-forward refit is expensive relative to
    a per-bar lookup -- computing it inline in this loop would refit on every
    simulated bar instead of once per symbol.
    """
    if params.trend_filter == "none" and params.exit_style == "fixed" and params.breakeven_r_multiple is None:
        return simulate_fvg_trades(ltf_df, htf_df, params)

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
    strat.MOMENTUM_BODY_RATIO_THRESHOLD = params.momentum_body_ratio_threshold
    strat.debug = True
    strat._log = lambda *a, **k: None

    bullish_result: Optional[HTFFVGResult] = None
    bearish_result: Optional[HTFFVGResult] = None
    last_n_weekly = 0
    htf_adx = None  # recomputed alongside the HTF zone recompute, same cadence
    htf_slope = htf_r2 = None
    htf_rho = None
    htf_ts_slope = None

    pending: List[Trade] = []
    open_trades: List[Trade] = []
    closed_trades: List[Trade] = []
    expired_trades: List[Trade] = []

    n = len(ltf_df)
    closes = ltf_df["close"].values
    highs = ltf_df["high"].values
    lows = ltf_df["low"].values
    opens = ltf_df["open"].values
    times = ltf_df.index

    # LTF ATR series for the trailing-stop variant (computed once, no lookahead
    # since atr() at position i only uses bars up to i).
    ltf_atr = _atr_fn(opens, highs, lows, params.trail_atr_period) if params.exit_style == "atr_trailing" else None

    for i in range(n):
        ts = times[i]
        bar_high = highs[i]
        bar_low = lows[i]

        # 1) expire stale pending orders
        still_pending = []
        for tr in pending:
            if ts >= _end_of_day(tr.created_time):
                tr.status = "expired"
                tr.exit_reason = "expired"
                expired_trades.append(tr)
            else:
                still_pending.append(tr)
        pending = still_pending

        # 2) check pending order fills
        still_pending = []
        for tr in pending:
            filled = (bar_low <= tr.entry <= bar_high)
            if filled:
                tr.entry_time = ts
                tr.status = "open"
                tr.exit_reason = "filled"
                if params.exit_style == "atr_trailing":
                    tr.trail_extreme = tr.entry
                open_trades.append(tr)
            else:
                still_pending.append(tr)
        pending = still_pending

        # 3) check open positions for SL/TP hit, or update+check trailing stop
        still_open = []
        for tr in open_trades:
            if params.exit_style == "atr_trailing" and not np.isnan(ltf_atr[i]):
                # chandelier: trail behind the best price reached since entry
                if tr.direction == 1:
                    tr.trail_extreme = max(tr.trail_extreme, bar_high)
                    new_sl = tr.trail_extreme - params.trail_atr_mult * ltf_atr[i]
                    tr.sl = max(tr.sl, new_sl)  # only ever tighten favorably
                else:
                    tr.trail_extreme = min(tr.trail_extreme, bar_low)
                    new_sl = tr.trail_extreme + params.trail_atr_mult * ltf_atr[i]
                    tr.sl = min(tr.sl, new_sl)

            # Breakeven stop: once favorable excursion clears breakeven_r_multiple
            # * original risk, move the *effective* stop to entry. tr.sl itself is
            # left untouched (only "fixed" exit_style reaches this branch, so it's
            # never mutated elsewhere either) so Trade.r_multiple's risk
            # denominator still reflects the trade's real original risk, not a
            # collapsed-to-zero one -- a breakeven-stopped trade scores a genuine
            # 0R scratch instead of silently vanishing from avg-winner-R stats.
            if params.exit_style == "fixed" and params.breakeven_r_multiple is not None \
                    and not getattr(tr, "breakeven_triggered", False):
                original_risk = abs(tr.entry - tr.sl)
                if original_risk > 0:
                    favorable = (bar_high - tr.entry) if tr.direction == 1 else (tr.entry - bar_low)
                    if favorable >= params.breakeven_r_multiple * original_risk:
                        tr.effective_sl = tr.entry
                        tr.breakeven_triggered = True

            effective_sl = getattr(tr, "effective_sl", tr.sl)
            if tr.direction == 1:
                sl_hit = bar_low <= effective_sl
                tp_hit = (not params.exit_style == "atr_trailing") and bar_high >= tr.tp
            else:
                sl_hit = bar_high >= effective_sl
                tp_hit = (not params.exit_style == "atr_trailing") and bar_low <= tr.tp

            if sl_hit:
                tr.exit_time = ts
                tr.exit_price = effective_sl
                if params.exit_style == "atr_trailing":
                    tr.exit_reason = "trail_sl"
                elif getattr(tr, "breakeven_triggered", False):
                    tr.exit_reason = "breakeven_sl"
                else:
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

        # 4) recompute weekly FVG zones (+ trend filters) once a new HTF bar is available
        avail_weekly = htf_df[htf_df.index <= ts]
        if len(avail_weekly) >= 3 and len(avail_weekly) != last_n_weekly:
            avail_weekly_w = avail_weekly.iloc[-params.htf_lookback_bars:]
            bullish_patterns, bearish_patterns = strat.detect_fvg_htf(avail_weekly_w)
            bullish_result = strat._process_htf_fvg_patterns(
                bullish_patterns, avail_weekly_w["low"], strat.htf_fill_pct, "Bullish"
            )
            bearish_result = strat._process_htf_fvg_patterns(
                bearish_patterns, avail_weekly_w["high"], 1 - strat.htf_fill_pct, "Bearish"
            )
            last_n_weekly = len(avail_weekly)

            if params.trend_filter == "adx":
                htf_adx = _adx_fn(
                    avail_weekly["open"].values, avail_weekly["high"].values,
                    avail_weekly["low"].values, params.adx_period, params.adx_period,
                )
            elif params.trend_filter == "regression":
                htf_slope, htf_r2 = _rolling_slope_r2(avail_weekly["close"].values, params.regression_window)
            elif params.trend_filter == "spearman":
                htf_rho = _rolling_spearman(avail_weekly["close"].values, params.regression_window)
            elif params.trend_filter == "theilsen":
                htf_ts_slope = _rolling_theil_sen_slope(avail_weekly["close"].values, params.regression_window)
                _, htf_r2 = _rolling_slope_r2(avail_weekly["close"].values, params.regression_window)

        # 5) look for a new setup if under the concurrency limit and enough history
        n_active = len(open_trades) + len(pending)
        if i >= 3 and n_active < params.limit_number_position and bullish_result is not None:
            last_price = closes[i]
            direction = strat._determine_direction(
                last_price, bullish_result.filtered_patterns, bearish_result.filtered_patterns
            )
            if direction is not None:
                is_bullish = direction == 1
                # Mean-reversion probe: require the regression/Lasso sign to DISAGREE with
                # the FVG-implied direction instead of agreeing (sign(beta_1) = 1-d rather
                # than = d, in _determine_direction's 0/1 encoding). target_bullish is what
                # every trend_filter branch below actually checks its sign against, so
                # flipping it once here flips the whole family uniformly.
                target_bullish = is_bullish if not params.invert_direction else not is_bullish
                trend_ok = True
                if params.trend_filter == "adx" and htf_adx is not None and len(htf_adx):
                    cur_adx = htf_adx[-1]
                    trend_ok = not np.isnan(cur_adx) and cur_adx >= params.adx_threshold
                elif params.trend_filter == "regression" and htf_slope is not None and len(htf_slope):
                    cur_slope, cur_r2 = htf_slope[-1], htf_r2[-1]
                    if np.isnan(cur_slope) or np.isnan(cur_r2):
                        trend_ok = False
                    else:
                        slope_matches = (cur_slope > 0) == target_bullish
                        trend_ok = slope_matches and cur_r2 >= params.regression_r2_threshold
                elif params.trend_filter == "spearman" and htf_rho is not None and len(htf_rho):
                    cur_rho = htf_rho[-1]
                    if np.isnan(cur_rho):
                        trend_ok = False
                    else:
                        rho_matches = (cur_rho > 0) == target_bullish
                        trend_ok = rho_matches and abs(cur_rho) >= params.regression_r2_threshold
                elif params.trend_filter == "theilsen" and htf_ts_slope is not None and len(htf_ts_slope):
                    cur_ts_slope, cur_r2 = htf_ts_slope[-1], htf_r2[-1]
                    if np.isnan(cur_ts_slope) or np.isnan(cur_r2):
                        trend_ok = False
                    else:
                        slope_matches = (cur_ts_slope > 0) == target_bullish
                        trend_ok = slope_matches and cur_r2 >= params.regression_r2_threshold
                elif params.trend_filter in ("lasso", "tree"):
                    cur_htf_ts = avail_weekly.index[-1] if len(avail_weekly) else None
                    if (
                        lasso_pred is None or cur_htf_ts is None
                        or cur_htf_ts not in lasso_pred.index
                        or pd.isna(lasso_pred.loc[cur_htf_ts])
                    ):
                        trend_ok = False
                    elif lasso_threshold is None or pd.isna(lasso_threshold.loc[cur_htf_ts]):
                        # threshold not yet available (walk-forward + causal-quantile
                        # warmup): fail closed rather than let every sign-matching
                        # prediction through unfiltered, which would defeat the point
                        # of requiring an unusually confident signal.
                        trend_ok = False
                    else:
                        pred = lasso_pred.loc[cur_htf_ts]
                        thr = lasso_threshold.loc[cur_htf_ts]
                        trend_ok = ((pred > 0) == target_bullish) and (abs(pred) >= thr)

                if trend_ok and params.session_filter:
                    # New-entry-only gate: London/NY session hours (OHLC index is
                    # ~UTC), skip the thin/choppy Asian-only window. Only blocks
                    # NEW pending orders -- existing pending/open trades still
                    # manage (fill checks, SL/TP) regardless of the hour.
                    trend_ok = params.session_start_hour <= ts.hour < params.session_end_hour

                if trend_ok:
                    # target_bullish, not is_bullish: in the non-inverted case they're
                    # identical (backward compatible), but under invert_direction=True,
                    # target_bullish equals the regression/Lasso sign itself whenever the
                    # gate passed (the gate condition literally checked slope_sign ==
                    # target_bullish) -- so the trade actually taken, and the LTF momentum
                    # required to confirm it, both follow the disagreeing slope's own
                    # direction, not the original HTF-zone direction. Trading the fade
                    # means fading the zone, not just requiring disagreement and then
                    # trading the zone's side anyway.
                    momentum_window = ltf_df.iloc[i - 3:i]
                    momentum_patterns = strat.detect_fvg_momentum_tres_strong(momentum_window, target_bullish)
                    if momentum_patterns:
                        current_momentum = momentum_patterns[0]
                        momentum_is_bullish = current_momentum.direction == "bullish"
                        if momentum_is_bullish == target_bullish:
                            strat._calculate_trade_parameters(
                                ltf_df.iloc[: i + 1], current_momentum, target_bullish
                            )
                            pending.append(
                                Trade(
                                    direction=strat.state,
                                    created_time=ts,
                                    entry=strat.entry,
                                    sl=strat.sl,
                                    tp=strat.tp,
                                )
                            )

    for tr in pending:
        tr.status = "expired"
        tr.exit_reason = "backtest_end"
        expired_trades.append(tr)
    for tr in open_trades:
        tr.status = "open"
        tr.exit_reason = "still_open"

    all_trades = closed_trades + open_trades + expired_trades
    all_trades.sort(key=lambda t: t.created_time)

    return all_trades, bullish_result, bearish_result


def run_variant_backtest(params: VariantParams) -> BacktestResult:
    connect_mt5()
    ltf_df, htf_df = fetch_ltf_htf_candles(
        params.symbol, params.start, params.end,
        ltf_timeframe=params.ltf_timeframe, htf_resample_rule=params.htf_resample_rule,
    )

    trades, bullish_result, bearish_result = simulate_fvg_trades_variant(ltf_df, htf_df, params)
    trades_df = trades_to_dataframe(trades)

    portfolio = build_vbt_portfolio(ltf_df, trades, params.limit_number_position)
    stats = portfolio.stats()

    return BacktestResult(
        params=params,
        ltf_df=ltf_df,
        htf_df=htf_df,
        trades=trades,
        trades_df=trades_df,
        bullish_htf_result=bullish_result,
        bearish_htf_result=bearish_result,
        portfolio=portfolio,
        stats=stats,
    )


# =========================================================================
# Risk-normalized position sizing (same trades, different vbt Portfolio)
# =========================================================================
def build_vbt_portfolio_risk_sized(
    ltf_df: pd.DataFrame,
    trades: List[Trade],
    n_slots: int,
    risk_fraction: float = 0.00002,
    max_size_pct: float = 0.30,
) -> "vbt.Portfolio":
    """
    Structurally identical to metafvg_backtest.build_vbt_portfolio (same
    event-grid construction, same collision-nudging, same annualization) --
    the only difference is a per-entry `size` array instead of a scalar.

    The validated baseline sizes every trade at a fixed 2% of equity
    notional, regardless of stop distance -- so risk-per-trade (equity lost
    if SL hits) varies by however much stop distance varies, which empirically
    spans a 100x+ range within a single symbol (e.g. EURUSD 0.008%-0.978% of
    price). This sizes each trade instead so a stop-out always costs
    risk_fraction of equity: size_pct = risk_fraction / stop_distance_pct,
    capped at max_size_pct to avoid degenerate leverage on near-zero-distance
    stops. risk_fraction defaults to whatever makes a *median*-stop-distance
    EURUSD trade size out at ~2%, matching the baseline's average exposure --
    so this isolates the effect of reducing size dispersion, not a change in
    overall leverage.
    """
    slots = _assign_slots(trades, max(n_slots, 1))
    cols = [f"slot_{i}" for i in range(len(slots))]

    col_events: dict = {}
    extra_ts: List[pd.Timestamp] = []
    for col, trs in zip(cols, slots):
        last_ts = None
        events = []
        for tr in trs:
            entry_ts = tr.entry_time
            if last_ts is not None and entry_ts <= last_ts:
                entry_ts = last_ts + pd.Timedelta(microseconds=1)
                extra_ts.append(entry_ts)
            events.append((entry_ts, "entry", tr))
            last_ts = entry_ts

            if tr.exit_time is not None and tr.exit_price is not None:
                exit_ts = tr.exit_time
                if exit_ts <= last_ts:
                    exit_ts = last_ts + pd.Timedelta(microseconds=1)
                    extra_ts.append(exit_ts)
                events.append((exit_ts, "exit", tr))
                last_ts = exit_ts
        col_events[col] = events

    idx = ltf_df.index.union(pd.DatetimeIndex(extra_ts)) if extra_ts else ltf_df.index

    entries = pd.DataFrame(False, index=idx, columns=cols)
    exits = pd.DataFrame(False, index=idx, columns=cols)
    short_entries = pd.DataFrame(False, index=idx, columns=cols)
    short_exits = pd.DataFrame(False, index=idx, columns=cols)
    price = pd.DataFrame(np.nan, index=idx, columns=cols)
    size = pd.DataFrame(np.nan, index=idx, columns=cols)

    for col, events in col_events.items():
        for ts, kind, tr in events:
            if kind == "entry":
                if tr.direction == 1:
                    entries.loc[ts, col] = True
                else:
                    short_entries.loc[ts, col] = True
                price.loc[ts, col] = tr.entry
                stop_distance_pct = abs(tr.entry - tr.sl) / tr.entry
                if stop_distance_pct > 0:
                    size.loc[ts, col] = min(risk_fraction / stop_distance_pct, max_size_pct)
                else:
                    size.loc[ts, col] = max_size_pct
            else:
                if tr.direction == 1:
                    exits.loc[ts, col] = True
                else:
                    short_exits.loc[ts, col] = True
                price.loc[ts, col] = tr.exit_price

    close_s = ltf_df["close"].reindex(idx, method="ffill")
    open_s = ltf_df["open"].reindex(idx, method="ffill")
    high_s = ltf_df["high"].reindex(idx, method="ffill")
    low_s = ltf_df["low"].reindex(idx, method="ffill")
    price = price.apply(lambda s: s.fillna(close_s))
    size = size.fillna(0.02)  # non-entry rows: vbt ignores size except at order-placing bars

    close_df = pd.DataFrame({c: close_s for c in cols})
    open_df = pd.DataFrame({c: open_s for c in cols})
    high_df = pd.DataFrame({c: high_s for c in cols})
    low_df = pd.DataFrame({c: low_s for c in cols})

    bar_freq = ltf_df.index.to_series().diff().median()

    portfolio = vbt.Portfolio.from_signals(
        close=close_df,
        open=open_df,
        high=high_df,
        low=low_df,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        price=price,
        size=size,
        size_type="percent",
        fees=0.0,
        freq=bar_freq,
        group_by=True,
        cash_sharing=True,
    )
    return portfolio
