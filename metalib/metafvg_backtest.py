"""
Standalone backtest engine for MetaFVG (metalib/metafvg.py), with both scales
upscaled one level: LTF M1 -> H4, HTF 4h -> Weekly.

Reuses MetaFVG's actual detection/parameter methods (detect_fvg_htf,
detect_fvg_momentum_tres_strong, _process_htf_fvg_patterns, _determine_direction,
_retrieve_last_pivots, _calculate_trade_parameters) by instantiating the class in
a detached (no MT5 connection) mode, so signal logic never drifts from production.

Data: fetches H4 candles from MT5 directly (the new LTF), then derives Weekly
candles (the new HTF) via the same resample the class itself uses internally.

Simulation walks the H4 series bar by bar, replicating the live cadence:
  - weekly FVG zones are recomputed once a new weekly bar becomes available
    (mirrors the daily fit() recompute), using only weekly bars closed as of
    that point (point-in-time correct, no lookahead in the crossing filter).
  - pending limit orders / open positions are marked-to-market against each
    H4 bar's high/low, with day-end expiry for unfilled pending orders
    (mirrors execute(..., is_limit=True, is_eod=True)).

vectorbt is used to turn the realized trades into a Portfolio (multi-column
"slot" model so `limit_number_position` concurrent trades are representable)
for equity-curve/Sharpe/drawdown/win-rate stats; the OHLC+markers chart itself
is drawn with plain plotly in the metadash visual style, driven by the same
trade records.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import MetaTrader5 as mt5
import vectorbt as vbt

import metalib.metafvg as metafvg_module
from metalib.metafvg import MetaFVG, FVGPattern, FVGPatternCollection, HTFFVGResult

# The class's detection methods are reused verbatim (see simulate_fvg_trades),
# but they're called dozens/hundreds of times over a multi-year walk-forward —
# their live-mode tqdm bars and per-call logging would otherwise flood stdout
# and add real overhead. Silenced only within this module's usage.
metafvg_module.tqdm = lambda iterable, *a, **k: iterable

# Default mode: both scales upscaled one level from production (M1->H4, 4h->Weekly).
LTF_TIMEFRAME = mt5.TIMEFRAME_H4
LTF_LABEL = "H4"
HTF_RESAMPLE_RULE = "W"
HTF_LABEL = "Weekly"

# Production mode (M1->4h, matching config/prod/metafvg.yaml) for cross-validating
# the engine against real MT5 trade history from the live bot.
PROD_LTF_TIMEFRAME = mt5.TIMEFRAME_M1
PROD_LTF_LABEL = "M1"
PROD_HTF_RESAMPLE_RULE = "4h"
PROD_HTF_LABEL = "4h"
PROD_HTF_LOOKBACK_BARS = 42  # 7 days of 4h bars, matches MetaFVG.DEFAULT_LOOKBACK_DAYS

# Selectable LTF/HTF options (e.g. for a UI timeframe picker), with approximate bar
# duration in minutes for validating a chosen pair isn't nonsensical (HTF must be
# strictly coarser than LTF for the 3-candle FVG pattern to mean anything).
LTF_TIMEFRAME_OPTIONS = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}
LTF_TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}

HTF_RESAMPLE_OPTIONS = {
    "1h": "1h",
    "4h": "4h",
    "12h": "12h",
    "1 Day": "D",
    "1 Week": "W",
}
HTF_RESAMPLE_MINUTES = {"1h": 60, "4h": 240, "12h": 720, "1 Day": 1440, "1 Week": 10080}


# =========================================================================
# Data fetching
# =========================================================================

def connect_mt5() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed, error = {mt5.last_error()}")


def _timeframe_to_minutes(tf: int) -> int:
    """Decode an mt5.TIMEFRAME_* constant into its bar duration in minutes."""
    if tf & 0x8000:  # week-based timeframes (W1, ...)
        return (tf & 0x7FFF) * 7 * 1440
    if tf & 0x4000:  # hour/day-based timeframes (H1, H4, D1, ...)
        return (tf & 0x3FFF) * 60
    return tf  # minute-based timeframes (M1, M5, ...) encode minutes directly


# MT5's terminal caps any single copy_rates_* call at just under terminal_info().maxbars
# (100_000 by default) -- asking for a date range that would span more bars at the
# chosen timeframe doesn't truncate gracefully, it fails the WHOLE request outright
# (error -2 "Invalid params"). Fine timeframes (M1/M5) over anything longer than ~2
# months hit this. Stay safely under the cap and page through history in chunks.
_MAX_BARS_PER_REQUEST = 90_000


def _fetch_ltf_rates(symbol: str, ltf_timeframe, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch LTF candles from MT5, paging through [start, end] in request-size-safe
    chunks. Returns whatever MT5 actually has available in that window -- if real
    history is shorter than requested (common for M1/M5 going back years), the
    result simply starts later than `start` rather than the whole fetch failing.
    """
    bar_minutes = _timeframe_to_minutes(ltf_timeframe)
    chunk_span = timedelta(minutes=_MAX_BARS_PER_REQUEST * bar_minutes)

    frames = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + chunk_span, end)
        rates = mt5.copy_rates_range(symbol, ltf_timeframe, chunk_start, chunk_end)
        if rates is not None and len(rates):
            frames.append(pd.DataFrame(rates))
        chunk_start = chunk_end

    if not frames:
        raise RuntimeError(f"Failed to load LTF data for {symbol}: {mt5.last_error()}")

    ltf_df = pd.concat(frames, ignore_index=True)
    ltf_df["time"] = pd.to_datetime(ltf_df["time"], unit="s")
    # Chunk boundaries can land on the same bar twice (copy_rates_range is inclusive).
    ltf_df = ltf_df.drop_duplicates(subset="time").set_index("time").sort_index()
    return ltf_df


def fetch_ltf_htf_candles(
    symbol: str,
    start: datetime,
    end: datetime,
    ltf_timeframe=LTF_TIMEFRAME,
    htf_resample_rule: str = HTF_RESAMPLE_RULE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch LTF candles from MT5 (in as many chunks as needed to get the max history
    available for the requested window) and derive HTF candles by resampling.

    label='right', closed='left': an HTF bucket covering [04:00, 08:00) is labeled
    "08:00", not "04:00". pandas' default (label='left') labels it "04:00" -- its
    *open* time -- while its close/high/low aggregate the entire window, which
    isn't actually known until ~08:00. The simulation loop gates visibility with
    `htf_df.index <= ts`, so under the default labeling the bucket's fully-realized
    close leaks in the instant ts reaches 04:00, up to one full HTF-bar-width
    (4h, or up to a week for HTF="1 Week") before that data would exist in real
    time. Live trading is naturally immune (MT5 can't return future bars: a
    MetaFVG.fit() at 05:00 only ever sees LTF data up to 05:00, so its "04:00"
    HTF bucket is genuinely partial) -- this is a backtest-only artifact of
    resampling the complete historical series in one batch, upfront, before the
    walk-forward loop starts. closed='left' (pandas' default, kept explicit here)
    preserves the exact same bar-to-bucket grouping as before -- only the label
    shifts to the bucket's true completion time; closed='right' would instead
    shift *which* LTF bars land in which bucket, corrupting the OHLC values
    themselves (verified empirically before choosing this combination).
    """
    ltf_df = _fetch_ltf_rates(symbol, ltf_timeframe, start, end)[["open", "high", "low", "close"]]

    # No .dropna(): MetaFVG._resample_to_htf doesn't drop empty buckets either, and in
    # PROD_* mode (M1->4h) weekend gaps produce real empty 4h buckets that live's own
    # detect_fvg_htf sees as no-op rows preserving true candle spacing — dropping them
    # here would let the backtest compare Friday/Monday candles as direct neighbours,
    # a different (and wrong) set of triples than live ever evaluates.
    htf_df = ltf_df.resample(htf_resample_rule, label="right", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )

    return ltf_df, htf_df


# =========================================================================
# Params / results
# =========================================================================

@dataclass
class BacktestParams:
    symbol: str
    start: datetime
    end: datetime
    limit_number_position: int = 1
    risk_reward: float = 2.0
    atr_sensitivity: float = 4.0
    htf_fill_pct: float = 1.0
    max_htf_number_crossings: int = 3
    atr_period: int = 14
    pivot_window: int = 7
    momentum_body_ratio_threshold: float = 0.7
    htf_lookback_bars: int = 52  # rolling weekly lookback window for FVG detection
    ltf_timeframe: int = LTF_TIMEFRAME  # mt5 timeframe constant for the LTF fetch
    htf_resample_rule: str = HTF_RESAMPLE_RULE  # pandas resample rule for the HTF


@dataclass
class Trade:
    direction: int  # 1 long, -1 short
    created_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    entry_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = "pending"  # pending -> filled -> tp/sl/expired
    status: str = "pending"  # pending, open, closed, expired

    @property
    def pnl(self) -> Optional[float]:
        if self.exit_price is None or self.entry is None or self.status != "closed":
            return None
        return (self.exit_price - self.entry) * self.direction

    @property
    def r_multiple(self) -> Optional[float]:
        pnl = self.pnl
        if pnl is None:
            return None
        risk = abs(self.entry - self.sl)
        if risk == 0:
            return None
        return pnl / risk


@dataclass
class BacktestResult:
    params: BacktestParams
    ltf_df: pd.DataFrame
    htf_df: pd.DataFrame
    trades: List[Trade]
    trades_df: pd.DataFrame
    bullish_htf_result: Optional[HTFFVGResult]
    bearish_htf_result: Optional[HTFFVGResult]
    portfolio: "vbt.Portfolio"
    stats: pd.Series


# =========================================================================
# Simulation
# =========================================================================

def _end_of_day(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.normalize() + pd.Timedelta(days=1)


def simulate_fvg_trades(
    ltf_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    params: BacktestParams,
) -> tuple[List[Trade], Optional[HTFFVGResult], Optional[HTFFVGResult]]:
    """
    Bar-by-bar walk-forward simulation over ltf_df (H4), using htf_df (Weekly)
    for zone detection. Reuses MetaFVG's own detection/parameter methods.
    """
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

    pending: List[Trade] = []
    open_trades: List[Trade] = []
    closed_trades: List[Trade] = []
    expired_trades: List[Trade] = []  # never filled: day-end expiry or backtest ended first

    n = len(ltf_df)
    closes = ltf_df["close"].values
    highs = ltf_df["high"].values
    lows = ltf_df["low"].values
    times = ltf_df.index

    for i in range(n):
        ts = times[i]
        bar_high = highs[i]
        bar_low = lows[i]

        # 1) expire stale pending orders (day-end expiry, mirrors is_eod=True)
        still_pending = []
        for tr in pending:
            if ts >= _end_of_day(tr.created_time):
                tr.status = "expired"
                tr.exit_reason = "expired"
                expired_trades.append(tr)
            else:
                still_pending.append(tr)
        pending = still_pending

        # 2) check pending order fills against this bar's range
        still_pending = []
        for tr in pending:
            filled = (bar_low <= tr.entry <= bar_high)
            if filled:
                tr.entry_time = ts
                tr.status = "open"
                tr.exit_reason = "filled"
                open_trades.append(tr)
            else:
                still_pending.append(tr)
        pending = still_pending

        # 3) check open positions for SL/TP hit (conservative: SL first if both hit)
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

        # 4) recompute weekly FVG zones once a new weekly bar is available (point-in-time)
        avail_weekly = htf_df[htf_df.index <= ts]
        if len(avail_weekly) >= 3 and len(avail_weekly) != last_n_weekly:
            avail_weekly = avail_weekly.iloc[-params.htf_lookback_bars:]
            bullish_patterns, bearish_patterns = strat.detect_fvg_htf(avail_weekly)
            bullish_result = strat._process_htf_fvg_patterns(
                bullish_patterns, avail_weekly["low"], strat.htf_fill_pct, "Bullish"
            )
            bearish_result = strat._process_htf_fvg_patterns(
                bearish_patterns, avail_weekly["high"], 1 - strat.htf_fill_pct, "Bearish"
            )
            last_n_weekly = len(htf_df[htf_df.index <= ts])

        # 5) look for a new setup if under the concurrency limit and enough history
        n_active = len(open_trades) + len(pending)
        if i >= 3 and n_active < params.limit_number_position and bullish_result is not None:
            last_price = closes[i]
            direction = strat._determine_direction(
                last_price, bullish_result.filtered_patterns, bearish_result.filtered_patterns
            )
            if direction is not None:
                is_bullish = direction == 1
                momentum_window = ltf_df.iloc[i - 3:i]
                momentum_patterns = strat.detect_fvg_momentum_tres_strong(momentum_window, is_bullish)
                if momentum_patterns:
                    current_momentum = momentum_patterns[0]
                    momentum_is_bullish = current_momentum.direction == "bullish"
                    if momentum_is_bullish == is_bullish:
                        strat._calculate_trade_parameters(
                            ltf_df.iloc[: i + 1], current_momentum, is_bullish
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

    # anything left pending/open at the end of history is unresolved
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


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    rows = []
    for tr in trades:
        rows.append(
            {
                "direction": "Long" if tr.direction == 1 else "Short",
                "created_time": tr.created_time,
                "entry_time": tr.entry_time,
                "entry": tr.entry,
                "sl": tr.sl,
                "tp": tr.tp,
                "exit_time": tr.exit_time,
                "exit_price": tr.exit_price,
                "exit_reason": tr.exit_reason,
                "status": tr.status,
                "pnl": tr.pnl,
                "r_multiple": tr.r_multiple,
            }
        )
    cols = [
        "direction", "created_time", "entry_time", "entry", "sl", "tp",
        "exit_time", "exit_price", "exit_reason", "status", "pnl", "r_multiple",
    ]
    return pd.DataFrame(rows, columns=cols)


# =========================================================================
# vectorbt Portfolio construction
# =========================================================================

def _assign_slots(trades: List[Trade], n_slots: int) -> List[List[Trade]]:
    """Greedy interval scheduling of filled trades into n_slots concurrent columns."""
    filled = [t for t in trades if t.entry_time is not None]
    filled.sort(key=lambda t: t.entry_time)

    slot_free_at = [pd.Timestamp.min] * n_slots
    slots: List[List[Trade]] = [[] for _ in range(n_slots)]

    for tr in filled:
        end = tr.exit_time if tr.exit_time is not None else pd.Timestamp.max
        placed = False
        for s in range(n_slots):
            if slot_free_at[s] <= tr.entry_time:
                slots[s].append(tr)
                slot_free_at[s] = end
                placed = True
                break
        if not placed:
            slots.append([tr])
            slot_free_at.append(end)

    return slots


def build_vbt_portfolio(ltf_df: pd.DataFrame, trades: List[Trade], n_slots: int) -> "vbt.Portfolio":
    """
    vbt.Portfolio.from_signals resolves at most one order per bar per column, so
    reusing the raw OHLC bar grid as the entries/exits index is unsafe whenever
    two events land on the same bar in the same slot column: a trade that fills
    and hits SL/TP within its own fill bar has entries==exits==True on one cell
    and simply vanishes from vbt's orders/trades entirely (confirmed empirically,
    not just theoretically); two different trades back-to-back in the same slot
    with one's exit_time == the next's entry_time silently clobber each other's
    price via last-write-wins .loc assignment. Both are common, not edge cases,
    for an SL/TP strategy on wide bars.

    Fix: keep the full bar grid (so Start/End/Period and vbt's annualization stay
    calendar-accurate, and open positions still mark-to-market bar by bar), but
    nudge any event timestamp that would collide with the previous event in the
    same slot column forward by a microsecond, inserting that handful of extra
    rows into the index (union, not replacement). Every entry/exit then lands on
    its own distinct row without losing bar-by-bar granularity everywhere else.
    `trades_df` remains the source of truth for per-trade prices/PnL regardless;
    this index only feeds vbt's aggregate stats (Sharpe/drawdown/win-rate) and
    equity curve.

    An earlier version of this fix replaced the bar grid outright with a sparse
    event-only index — that also resolves the collision, but vbt's "Period" stat
    (and everything annualized from it, e.g. Sharpe/Calmar) is `len(index) * freq`,
    not `index[-1] - index[0]`, so a sparse index silently understates elapsed
    time by orders of magnitude. Do not reintroduce that.
    """
    slots = _assign_slots(trades, max(n_slots, 1))
    cols = [f"slot_{i}" for i in range(len(slots))]

    col_events: dict = {}
    extra_ts: List[pd.Timestamp] = []
    for col, trs in zip(cols, slots):
        last_ts = None
        events = []
        for tr in trs:  # _assign_slots already sorted these by entry_time
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

    for col, events in col_events.items():
        for ts, kind, tr in events:
            if kind == "entry":
                if tr.direction == 1:
                    entries.loc[ts, col] = True
                else:
                    short_entries.loc[ts, col] = True
                price.loc[ts, col] = tr.entry
            else:
                if tr.direction == 1:
                    exits.loc[ts, col] = True
                else:
                    short_exits.loc[ts, col] = True
                price.loc[ts, col] = tr.exit_price

    # Fallback price for non-event rows (the vast majority — real bars and the
    # handful of ffilled nudge rows): that bar's own OHLC, shared across slots
    # since all slots trade the same symbol.
    close_s = ltf_df["close"].reindex(idx, method="ffill")
    open_s = ltf_df["open"].reindex(idx, method="ffill")
    high_s = ltf_df["high"].reindex(idx, method="ffill")
    low_s = ltf_df["low"].reindex(idx, method="ffill")
    price = price.apply(lambda s: s.fillna(close_s))

    close_df = pd.DataFrame({c: close_s for c in cols})
    open_df = pd.DataFrame({c: open_s for c in cols})
    high_df = pd.DataFrame({c: high_s for c in cols})
    low_df = pd.DataFrame({c: low_s for c in cols})

    # Annualization factor must reflect the real bar cadence (H4, M1, ...), not the
    # handful of inserted microsecond-nudge rows.
    bar_freq = ltf_df.index.to_series().diff().median()

    # size_type='percent' allocates a fixed fraction of current equity per trade,
    # rather than a fixed 1-unit notional (`size=1, size_type='amount'`, vbt's
    # default). A raw 1-unit position is a rounding error against 100 starting
    # cash on EURUSD (~1.10/unit) but ~200x the entire portfolio on GER40
    # (~20,000/unit) — every percentage-based stat (Total Return, Sharpe, Max
    # Drawdown) becomes meaningless, and can even exceed 100% drawdown, once a
    # symbol's price scale is far from 1. Fixed-fractional sizing keeps these
    # stats comparable across instruments regardless of price scale; it is not
    # a claim about real position sizing (SL/TP levels are independent of it).
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
        size=0.02,
        size_type="percent",
        fees=0.0,
        freq=bar_freq,
        group_by=True,
        cash_sharing=True,
    )
    return portfolio


# =========================================================================
# Orchestrator
# =========================================================================

def run_backtest(params: BacktestParams) -> BacktestResult:
    connect_mt5()
    ltf_df, htf_df = fetch_ltf_htf_candles(
        params.symbol, params.start, params.end,
        ltf_timeframe=params.ltf_timeframe, htf_resample_rule=params.htf_resample_rule,
    )

    trades, bullish_result, bearish_result = simulate_fvg_trades(ltf_df, htf_df, params)
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


if __name__ == "__main__":
    p = BacktestParams(
        symbol="EURUSD",
        start=datetime.utcnow() - timedelta(days=365 * 3),
        end=datetime.utcnow(),
    )
    result = run_backtest(p)
    print(f"LTF ({LTF_LABEL}) bars: {len(result.ltf_df)}   HTF ({HTF_LABEL}) bars: {len(result.htf_df)}")
    print(f"Trades: {len(result.trades_df)}")
    print(result.trades_df.tail(10))
    print(result.stats)
