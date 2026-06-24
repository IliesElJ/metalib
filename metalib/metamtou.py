"""
metamtou.py — MTO OU Quorum Mean-Reversion (QMR) live strategy.

Implements the QMR strategy as a MetaStrategy subclass for live MT5 trading.

Strategy rules (mirrors backtest_ou.py logic exactly):
  Entry  : Q >= k_min AND n_true_open >= 1 AND R/R >= rr_min
  Stop   : entry * exp(± stop_mult * w_vol_1d)
  Target : weighted OU attractor (significance-weighted mean in log-space)
  Trail  : every check_interval bars, if actual_speed >= trail_threshold * expected_speed
           → ratchet stop, upgrade target if better attractor exists
  Exit   : target hit / stop hit / max_hold_days elapsed

Recommended config (from sweep analysis across 10 symbols):
  k_min=3, rr_min=1.0, horizon=3, stop_mult=1.5, max_hold_days=30
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pytz

try:
    from metalib.metastrategy import MetaStrategy
except ImportError:
    _pkg = os.path.dirname(os.path.abspath(__file__))
    if _pkg not in sys.path:
        sys.path.insert(0, _pkg)
    from metastrategy import MetaStrategy


# ─── pure helpers (mirrored from backtest_ou.py) ───────────────────────────

def _pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol.upper() else 0.0001


def _compute_quorum(snap_h: pd.DataFrame) -> int:
    """Max of (#lags with z>0, #lags with z<0) over active lags 0–2."""
    sub = snap_h[snap_h["lag"].isin([0, 1, 2])].copy()
    act = sub[sub["regime_active"] == True]
    if act.empty:
        return 0
    zs = act["z_score"].fillna(0)
    return max(int((zs > 0).sum()), int((zs < 0).sum()))


def _compute_wz(snap_h: pd.DataFrame) -> float:
    """Significance-weighted z-score across all active levels."""
    act = snap_h[snap_h["regime_active"] == True].copy()
    if act.empty or act["z_score"].isna().all():
        return float("nan")
    w = -np.log10(act["pval_kappa"].clip(lower=1e-15))
    W = w.sum()
    if W == 0:
        return float("nan")
    w = w / W
    return float((w * act["z_score"].fillna(0)).sum())


def _weighted_target_and_vol(snap_h: pd.DataFrame) -> Optional[tuple[float, float]]:
    """Return (target_log_price, weighted_daily_vol) from active levels."""
    act = snap_h[snap_h["regime_active"] == True].copy()
    if act.empty:
        return None
    w = -np.log10(act["pval_kappa"].clip(lower=1e-15))
    W = w.sum()
    if W == 0:
        return None
    w = w / W
    target_log = float((w * np.log(act["level_price"])).sum())
    w_vol = float((w * act["resid_vol_daily"]).sum())
    return target_log, w_vol


def _true_open_count(snap_h: pd.DataFrame, close_price: float, direction: int) -> int:
    """Count active levels still on the correct side of current price."""
    act = snap_h[snap_h["regime_active"] == True]
    if act.empty:
        return 0
    if direction == -1:   # SHORT: attractor must be below price
        return int((act["level_price"] < close_price).sum())
    else:                 # LONG: attractor must be above price
        return int((act["level_price"] > close_price).sum())


def _median_half_life(snap_h: pd.DataFrame) -> float:
    """Weighted median half-life for PnL speed expectation."""
    hl = snap_h["half_life_days"].dropna()
    return float(hl.median()) if len(hl) > 0 else 20.0


# ─── strategy class ────────────────────────────────────────────────────────


class MetaMTOU(MetaStrategy):
    """
    MTO OU Quorum Mean-Reversion (QMR) live trading strategy.

    One symbol per instance. fit() computes the rolling OU grid daily.
    signals() reads from the precomputed grid and sets self.state / sl / tp.
    check_conditions() enters or manages the open position.
    """

    def __init__(
        self,
        symbols: list[str],
        timeframe,
        tag: str,
        size_position: float,
        active_hours=None,
        k_min: int = 3,
        rr_min: float = 1.0,
        horizon: int = 3,
        stop_mult: float = 1.5,
        max_hold_days: int = 30,
        check_interval: int = 5,
        trail_threshold: float = 0.8,
        from_date: Optional[datetime] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(symbols, timeframe, tag, size_position, active_hours)

        # OU model parameters
        self.k_min = k_min
        self.rr_min = rr_min
        self.horizon = horizon
        self.stop_mult = stop_mult
        self.max_hold_days = max_hold_days
        self.check_interval = check_interval
        self.trail_threshold = trail_threshold
        self.from_date = from_date or datetime(2000, 1, 1)
        # Where to look for cache_{symbol}_roll.parquet and dashboard.py
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "..")

        # State from fit()
        self.roll_df: Optional[pd.DataFrame] = None

        # State from signals()
        self.sl: Optional[float] = None
        self.tp: Optional[float] = None
        self._signal_wz: float = float("nan")
        self._signal_q: int = 0
        self._signal_rr: float = 0.0
        self._dist_pips_initial: float = 0.0

        # Open trade tracking (persists across run() calls)
        self.trade_entry_date: Optional[datetime] = None
        self.trade_entry_px: Optional[float] = None
        self.trade_direction: Optional[int] = None
        self.trade_stop: Optional[float] = None
        self.trade_target: Optional[float] = None
        self.trade_bars_in: int = 0

    # ─── fit ───────────────────────────────────────────────────────────────

    def fit(self):
        """Load full OHLC history and compute the rolling OU grid. Called daily."""
        symbol = self.symbols[0]
        utc = pytz.utc
        now = datetime.now(utc)

        # Check parquet cache (valid for 23 h — regenerated by daily fit)
        cache_path = os.path.join(self.cache_dir, f"cache_{symbol}_roll.parquet")
        if os.path.exists(cache_path):
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_path), tz=utc)
            if (now - mtime).total_seconds() < 23 * 3600:
                try:
                    self.roll_df = pd.read_parquet(cache_path)
                    print(f"{self.tag}:: roll_df loaded from cache "
                          f"({len(self.roll_df):,} rows, last eval="
                          f"{self.roll_df['eval_date'].max()})")
                    return
                except Exception as e:
                    print(f"{self.tag}:: Cache load failed ({e}), recomputing...")

        # Load full OHLC via MT5
        ohlc = self._load_ohlc(symbol)
        if ohlc is None or len(ohlc) < 200:
            print(f"{self.tag}:: Insufficient data for {symbol}, skipping fit")
            return

        # Import OU grid builder (same as research_base.py does)
        _dir = os.path.abspath(self.cache_dir)
        if _dir not in sys.path:
            sys.path.insert(0, _dir)
        from dashboard import compute_rolling_grid, get_all_second_monday_opens

        X_log = np.log(ohlc["close"])
        sm = get_all_second_monday_opens(ohlc)

        print(f"{self.tag}:: Computing rolling OU grid for {symbol} ...")
        self.roll_df = compute_rolling_grid(ohlc, X_log, sm)

        if self.roll_df is not None and len(self.roll_df) > 0:
            self.roll_df.to_parquet(cache_path)
            print(f"{self.tag}:: Grid ready - {len(self.roll_df):,} rows, "
                  f"saved to {os.path.basename(cache_path)}")
        else:
            print(f"{self.tag}:: Empty grid - check MT5 data quality")

    # ─── signals ───────────────────────────────────────────────────────────

    def signals(self):
        """
        Evaluate the QMR signal at the latest eval_date in roll_df.
        Sets self.state (+1/-1/0), self.sl, self.tp.
        """
        symbol = self.symbols[0]
        self.state = 0
        self.sl = self.tp = None

        if self.roll_df is None or len(self.roll_df) == 0:
            print(f"{self.tag}:: No roll_df - run fit() first")
            self._set_signal_data(symbol)
            return

        # Latest snapshot for target horizon
        snap = self._latest_snap()
        if snap is None or snap.empty:
            self._set_signal_data(symbol)
            return

        # Current price from MT5 tick
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"{self.tag}:: MT5 tick unavailable for {symbol}")
            self._set_signal_data(symbol)
            return
        close = (tick.bid + tick.ask) / 2

        # ── Entry condition 1: quorum Q >= k_min ──────────────────────────
        q = _compute_quorum(snap)
        self._signal_q = q   # always record for logging even on early return
        if q < self.k_min:
            self._set_signal_data(symbol, q=q, close=close)
            return

        # ── wz for direction ─────────────────────────────────────────────
        wz = _compute_wz(snap)
        if math.isnan(wz) or wz == 0.0:
            self._set_signal_data(symbol, q=q, close=close)
            return
        direction = -1 if wz > 0 else 1

        # ── Entry condition 2: at least 1 true open level ─────────────────
        n_to = _true_open_count(snap, close, direction)
        if n_to < 1:
            self._set_signal_data(symbol, q=q, wz=wz, close=close)
            return

        # ── Weighted target and vol ───────────────────────────────────────
        tv = _weighted_target_and_vol(snap)
        if tv is None:
            self._set_signal_data(symbol, q=q, wz=wz, close=close)
            return
        target_log, w_vol_1d = tv

        target_px = math.exp(target_log)
        log_close = math.log(close)

        if direction == -1:
            stop_px = math.exp(log_close + self.stop_mult * w_vol_1d)
        else:
            stop_px = math.exp(log_close - self.stop_mult * w_vol_1d)

        # ── Entry condition 3: R/R >= rr_min ─────────────────────────────
        dist_target = abs(target_px - close)
        dist_stop = abs(stop_px - close)
        if dist_stop == 0:
            self._set_signal_data(symbol, q=q, wz=wz, close=close)
            return
        rr = dist_target / dist_stop
        if rr < self.rr_min:
            self._set_signal_data(symbol, q=q, wz=wz, rr=rr, close=close)
            return

        # ── All conditions met ────────────────────────────────────────────
        self.state = direction
        self.sl = stop_px
        self.tp = target_px
        self._signal_wz = wz
        self._signal_q = q
        self._signal_rr = rr
        self._dist_pips_initial = dist_target / _pip_size(symbol)

        self._set_signal_data(symbol, q=q, wz=wz, rr=rr, close=close,
                              stop=stop_px, target=target_px)

        trade_type = "SHORT" if direction == -1 else "LONG"
        print(f"{self.tag}:: SIGNAL {trade_type}  Q={q}  wz={wz:+.3f}  "
              f"R/R={rr:.2f}  target={target_px:.5f}  stop={stop_px:.5f}")

    # ─── check_conditions ──────────────────────────────────────────────────

    def check_conditions(self):
        """Enter a new trade or manage the existing open position."""
        symbol = self.symbols[0]

        if self.are_positions_with_tag_open():
            self._manage_position(symbol)
        elif self.state in [1, -1]:
            self._enter_trade(symbol)

    # ─── private: entry ────────────────────────────────────────────────────

    def _enter_trade(self, symbol: str):
        """Place a market order with sl/tp from signals()."""
        if self.sl is None or self.tp is None:
            return

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            print(f"{self.tag}:: Cannot get symbol info for {symbol}")
            return

        digits = sym_info.digits
        sl = round(self.sl, digits)
        tp = round(self.tp, digits)

        result = self.execute(symbol=symbol, short=(self.state == -1), sl=sl, tp=tp)
        if result is None:
            print(f"{self.tag}:: execute() returned None")
            return

        # Record trade state for management
        tick = mt5.symbol_info_tick(symbol)
        entry_px = (tick.bid + tick.ask) / 2 if tick else self.tp  # fallback
        now_utc = datetime.now(pytz.utc)

        self.trade_entry_date = now_utc
        self.trade_entry_px = entry_px
        self.trade_direction = self.state
        self.trade_stop = sl
        self.trade_target = tp
        self.trade_bars_in = 0

        side = "SELL" if self.state == -1 else "BUY"
        print(f"{self.tag}:: Entered {side} {symbol}  "
              f"entry={entry_px:.5f}  sl={sl:.5f}  tp={tp:.5f}  "
              f"Q={self._signal_q}  wz={self._signal_wz:+.3f}  R/R={self._signal_rr:.2f}")

    # ─── private: management ───────────────────────────────────────────────

    def _manage_position(self, symbol: str):
        """Manage the open position: max-hold exit, trail stop, target upgrade."""
        # Reconstruct state from MT5 if process restarted
        self._reconstruct_trade_state(symbol)
        if self.trade_direction is None:
            return

        self.trade_bars_in += 1

        # Current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return
        close = (tick.bid + tick.ask) / 2

        now_utc = datetime.now(pytz.utc)
        cal_days = (now_utc - self.trade_entry_date).days if self.trade_entry_date else 0

        # ── Max hold exit ─────────────────────────────────────────────────
        if cal_days >= self.max_hold_days:
            print(f"{self.tag}:: Max hold {self.max_hold_days}d reached - closing position")
            self.close_all_positions()
            pnl_pips = ((close - self.trade_entry_px) * self.trade_direction
                        / _pip_size(symbol))
            print(f"{self.tag}:: Closed at max_hold  pnl={pnl_pips:+.1f} pips")
            self._reset_trade_state()
            return

        # ── PnL speed check ───────────────────────────────────────────────
        if self.trade_bars_in > 0 and self.trade_bars_in % self.check_interval == 0:
            d = self.trade_direction
            current_pnl_pips = ((close - self.trade_entry_px) * d
                                / _pip_size(symbol))

            if current_pnl_pips > 0 and self._dist_pips_initial > 0:
                actual_rate = current_pnl_pips / self.trade_bars_in

                snap = self._latest_snap()
                med_hl = _median_half_life(snap) if snap is not None else 20.0
                if med_hl <= 0 or math.isnan(med_hl):
                    med_hl = 20.0

                expected_rate = self._dist_pips_initial / med_hl

                if expected_rate > 0 and actual_rate >= self.trail_threshold * expected_rate:
                    self._trail_stop(symbol, close)
                    self._try_upgrade_target(symbol, snap, close)

    def _trail_stop(self, symbol: str, close: float):
        """Ratchet the stop loss toward profit using recent bar extremes."""
        d = self.trade_direction
        recent = self.data.get(symbol)
        if recent is None or recent.empty:
            return

        tail = recent.tail(self.check_interval)
        sym_info = mt5.symbol_info(symbol)
        digits = sym_info.digits if sym_info else 5

        if d == 1:   # LONG: trail stop up via recent lows
            candidate = float(tail["low"].min()) * 0.9995
            new_stop = max(self.trade_stop, candidate)
        else:        # SHORT: trail stop down via recent highs
            candidate = float(tail["high"].max()) * 1.0005
            new_stop = min(self.trade_stop, candidate)

        new_stop = round(new_stop, digits)
        if new_stop != self.trade_stop:
            if self._modify_sl_tp(symbol, new_stop, round(self.trade_target, digits)):
                print(f"{self.tag}:: Stop trailed to {new_stop:.5f}")
                self.trade_stop = new_stop

    def _try_upgrade_target(
        self, symbol: str, snap: Optional[pd.DataFrame], close: float
    ):
        """Upgrade TP to a better (further) attractor if one exists."""
        if snap is None or snap.empty:
            return
        d = self.trade_direction
        act = snap[snap["regime_active"] == True]
        if act.empty:
            return

        sym_info = mt5.symbol_info(symbol)
        digits = sym_info.digits if sym_info else 5

        if d == 1:   # LONG: look for higher level above current target AND above close
            candidates = act[
                (act["level_price"] > self.trade_target) &
                (act["level_price"] > close)
            ]
            if not candidates.empty:
                new_target = round(float(candidates["level_price"].min()), digits)
                if new_target != self.trade_target:
                    if self._modify_sl_tp(symbol, round(self.trade_stop, digits), new_target):
                        print(f"{self.tag}:: Target upgraded {self.trade_target:.5f}"
                              f" → {new_target:.5f}")
                        self.trade_target = new_target
        else:        # SHORT: look for lower level below current target AND below close
            candidates = act[
                (act["level_price"] < self.trade_target) &
                (act["level_price"] < close)
            ]
            if not candidates.empty:
                new_target = round(float(candidates["level_price"].max()), digits)
                if new_target != self.trade_target:
                    if self._modify_sl_tp(symbol, round(self.trade_stop, digits), new_target):
                        print(f"{self.tag}:: Target upgraded {self.trade_target:.5f}"
                              f" → {new_target:.5f}")
                        self.trade_target = new_target

    def _modify_sl_tp(self, symbol: str, sl: float, tp: float) -> bool:
        """Send TRADE_ACTION_SLTP to modify SL and TP on the open position."""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return False
        for pos in positions:
            if pos.comment != self.tag[:16]:
                continue
            req = {
                "action":   mt5.TRADE_ACTION_SLTP,
                "symbol":   symbol,
                "sl":       sl,
                "tp":       tp,
                "position": pos.ticket,
            }
            res = mt5.order_send(req)
            if res.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"{self.tag}:: SLTP modify failed: {res.comment}")
                return False
            return True
        return False

    # ─── private: utilities ────────────────────────────────────────────────

    def _load_ohlc(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load full daily OHLC from MT5 since from_date."""
        try:
            rates = mt5.copy_rates_range(
                symbol, mt5.TIMEFRAME_D1, self.from_date, datetime.now()
            )
            if rates is None or len(rates) == 0:
                raise RuntimeError(f"copy_rates_range returned empty for {symbol}")
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = (df.set_index("time")[["open", "high", "low", "close"]]
                  .asfreq("B").ffill())
            print(f"{self.tag}:: Loaded {len(df):,} bars "
                  f"({df.index[0].date()} -> {df.index[-1].date()})")
            return df
        except Exception as e:
            print(f"{self.tag}:: OHLC load failed: {e}")
            return None

    def _latest_snap(self) -> Optional[pd.DataFrame]:
        """Return the latest eval_date slice from roll_df for self.horizon."""
        if self.roll_df is None:
            return None
        rdf = self.roll_df[self.roll_df["horizon_months"] == self.horizon]
        if rdf.empty:
            return None
        latest = rdf["eval_date"].max()
        return rdf[rdf["eval_date"] == latest]

    def _reconstruct_trade_state(self, symbol: str):
        """Recover trade state from MT5 positions if instance was restarted."""
        if self.trade_entry_date is not None:
            return  # state already in memory
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return
        for pos in positions:
            if pos.comment != self.tag[:16]:
                continue
            self.trade_entry_date = datetime.fromtimestamp(pos.time, tz=pytz.utc)
            self.trade_entry_px = pos.price_open
            self.trade_direction = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            self.trade_stop = pos.sl if pos.sl > 0 else None
            self.trade_target = pos.tp if pos.tp > 0 else None
            self.trade_bars_in = 0  # unknown after restart
            print(f"{self.tag}:: Reconstructed trade state from MT5 position "
                  f"{pos.ticket}  dir={self.trade_direction}  "
                  f"entry={pos.price_open:.5f}")
            return

    def _reset_trade_state(self):
        """Clear all open-trade tracking variables."""
        self.trade_entry_date = None
        self.trade_entry_px = None
        self.trade_direction = None
        self.trade_stop = None
        self.trade_target = None
        self.trade_bars_in = 0
        self._dist_pips_initial = 0.0

    def _set_signal_data(
        self, symbol: str,
        q: int = 0, wz: float = float("nan"),
        rr: float = float("nan"), close: float = float("nan"),
        stop: float = float("nan"), target: float = float("nan"),
    ):
        """Populate self.signalData for DB logging."""
        self.signalData = pd.Series({
            "timestamp": datetime.now(pytz.utc),
            "symbol":    symbol,
            "state":     self.state,
            "quorum_k":  q,
            "wz":        wz,
            "rr":        rr,
            "close":     close,
            "stop":      stop,
            "target":    target,
            "horizon":   self.horizon,
            "k_min":     self.k_min,
        })
