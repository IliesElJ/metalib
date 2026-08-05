"""
MetaFVG v2: Spearman mean-reversion gate.

Adds a rolling Spearman rank-correlation gate on HTF closes on top of the
base MetaFVG HTF-zone + LTF-momentum pipeline, and inverts the traded
direction whenever the gate disagrees with the zone (mean-reversion, not
trend-following). Ported from the research/backtest fork
(metalib/research/scripts/metafvg_variants.py, trend_filter="spearman",
invert_direction=True) -- see metalib/research/reports/
metafvg_research_report.pdf for the full derivation, backtest results, and
known caveats (in-sample parameter selection, no out-of-sample validation
yet, single data source).

Combined-mix production parameters (research report Sec. 10, the best
Sharpe found by combining the two parameter-sensitivity sweep results):
    SPEARMAN_WINDOW = 25       (research default: 20)
    SPEARMAN_THRESHOLD = 0.4   (research default: 0.5)
    atr_sensitivity = 2.0      (base class default: 4)
Backtested equal-weight 14-instrument portfolio Sharpe 0.910 on this exact
config (vs. 0.536 baseline on the same data snapshot). risk_reward is left
at the base class default (2) -- confirmed redundant with atr_sensitivity
in the TP formula (both only ever appear as a product), see report Sec. 9.2.

Two things this class deliberately does NOT inherit unchanged from v1:

1. check_conditions() -- v1 never passes short=True for state==-1, so live
   v1 shorts are placed as BUY orders (a pre-existing bug, out of scope to
   fix in v1 itself since it's currently live and this file must not change
   its behavior). v2 trades both directions roughly symmetrically by design
   (mean-reversion, not directional), so this bug cannot carry over -- it's
   fixed here.

2. Position sizing -- v1 uses a static per-instance lot size
   (self.size_position) set in YAML, unrelated to account equity. v2 uses
   fixed-fraction risk sizing: self.size_position is reinterpreted as the
   fraction of account balance to LOSE if the stop-loss is hit. The
   implementation lives on the base class as
   MetaStrategy._fixed_fraction_position_size (extracted so metamtou and
   metamlp can share it); _resolve_position_size below delegates to it.
   Combined-mix production risk is 0.5% (0.005), a smaller fraction of the
   backtest's 2% -- a conservative live rollout, not a claim that 0.5% was
   itself backtested.

   NOTE on metascal.py's MetaScale: dashboard-triggered risk-budgeting tool
   that periodically rewrites size_position for the strategies listed in
   its DEFAULT_STRATEGY_PARAMS (metafvg, metago, metane, metaga, metaob)
   via CVXPY. It uses tick_value/tick_size (P&L-per-tick, for its covariance
   calc) -- a different quantity from the base helper's contract_size*price
   (notional per lot, for equity-percent sizing); the two aren't
   interchangeable. metafvg_v2, metamtou and metamlp are all deliberately
   NOT in DEFAULT_STRATEGY_PARAMS -- they manage their own sizing and
   MetaScale has no visibility into them, by design. If that changes,
   either _resolve_position_size on those classes must be dropped (falling
   back to the base default that returns raw size_position) or MetaScale
   needs to write fractions in the fixed-fraction convention.
"""
import MetaTrader5 as mt5
import numpy as np
from numba import njit

from metalib.metafvg import MetaFVG


@njit(cache=True)
def _rolling_spearman(close: np.ndarray, window: int) -> np.ndarray:
    """
    Spearman rank correlation of close vs. bar index, rolling over `window`.
    Identical implementation to metafvg_variants.py's version, ported here
    directly (not imported from research/) so production has no dependency
    on the research tree. No tie-correction (float prices essentially never
    tie exactly).
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


class MetaFVGv2(MetaFVG):
    """MetaFVG + Spearman mean-reversion gate, combined-mix parameters."""

    SPEARMAN_WINDOW = 25
    SPEARMAN_THRESHOLD = 0.4

    def __init__(
        self,
        symbols,
        timeframe,
        size_position: float,
        tag: str,
        limit_number_position: int = 1,
        active_hours=None,
    ):
        """
        size_position is the fraction of account balance to risk if the stop
        loss is hit (e.g. 0.005 = 0.5% of balance). It is also used as a
        fallback lot size if the dynamic computation cannot run (MT5 info
        unavailable or no SL provided).
        """
        super().__init__(symbols, timeframe, size_position, tag, limit_number_position, active_hours)
        self.atr_sensitivity = 2.0  # combined-mix override; base class default is 4
        # risk_reward intentionally left at the base class default (2).

    # =========================================================================
    # Signal generation: base pipeline + Spearman mean-reversion gate
    # =========================================================================

    def signals(self) -> None:
        """
        Same pipeline as MetaFVG.signals() through HTF-zone direction
        determination, then adds the Spearman mean-reversion gate before LTF
        momentum confirmation. Mirrors metafvg_variants.py's spearman +
        invert_direction branch exactly -- target_bullish (not the zone's
        own is_bullish) is what everything downstream (momentum confirmation,
        trade parameterization, self.state) actually follows.
        """
        self.state = 0
        self._log("Generating trading signals (v2: Spearman mean-reversion gate)")

        _, current_open_position = self.get_positions_info()
        self._log(f"Current open positions: {current_open_position}")
        if current_open_position >= self.limit_number_position:
            self._log(f"Maximum open position reached: {current_open_position}")
            self._save_signal()
            return

        if self.bullish_htf_result is None or self.bearish_htf_result is None:
            self._log("HTF FVG patterns not initialized. Run fit() first.")
            self._save_signal()
            return

        bullish_filtered = self.bullish_htf_result.filtered_patterns
        bearish_filtered = self.bearish_htf_result.filtered_patterns
        if bullish_filtered.is_empty() and bearish_filtered.is_empty():
            self._log("No HTF FVG patterns detected, no action required")
            self._save_signal()
            return

        ohlc_ltf = self.data[self.symbols[0]]
        last_price = ohlc_ltf["close"].iloc[-1]
        self._log(f"Pulled data for symbol: {self.symbols[0]}")
        self._log(f"Last Price: ${last_price}")

        direction = self._determine_direction(last_price, bullish_filtered, bearish_filtered)
        if direction is None:
            self._log("Price not in any Bullish OR Bearish FVG H4, no action required")
            self._save_signal(last_price)
            return

        is_bullish = direction == 1
        target_bullish = not is_bullish  # mean-reversion: always the disagreeing direction

        # --- Spearman mean-reversion gate ---
        ohlc_htf = self._resample_to_htf(ohlc_ltf, self.DEFAULT_HTF_TIMEFRAME)
        htf_close = ohlc_htf["close"].values
        if len(htf_close) < self.SPEARMAN_WINDOW:
            self._log(f"Not enough HTF bars ({len(htf_close)}) for Spearman window "
                      f"({self.SPEARMAN_WINDOW}), no action required")
            self._save_signal(last_price)
            return

        rho = _rolling_spearman(htf_close, self.SPEARMAN_WINDOW)
        cur_rho = rho[-1]
        if np.isnan(cur_rho):
            self._log("Spearman rho is NaN, no action required")
            self._save_signal(last_price)
            return

        rho_matches = (cur_rho > 0) == target_bullish
        gate_ok = rho_matches and abs(cur_rho) >= self.SPEARMAN_THRESHOLD
        self._log(f"Spearman gate: rho={cur_rho:.4f} target_bullish={target_bullish} "
                  f"rho_matches={rho_matches} |rho|>=thr={abs(cur_rho) >= self.SPEARMAN_THRESHOLD} "
                  f"gate_ok={gate_ok}")
        if not gate_ok:
            self._log("Spearman gate rejected signal, no action required")
            self._save_signal(last_price)
            return

        # --- LTF momentum confirmation, in the gate's (mean-reversion) direction ---
        momentum_patterns = self.detect_fvg_momentum_tres_strong(ohlc_ltf.iloc[-4:-1], target_bullish)
        if not momentum_patterns:
            self._log("No LTF FVG patterns detected, no action required")
            self._save_signal(last_price)
            return

        current_momentum = momentum_patterns[0]
        momentum_is_bullish = current_momentum.direction == "bullish"
        if momentum_is_bullish != target_bullish:
            self._log("LTF FVG direction doesn't match gate target, no action required")
            self._save_signal(last_price)
            return

        self._log(f"Mean-reversion setup confirmed (zone was "
                  f"{'bullish' if is_bullish else 'bearish'}, trading "
                  f"{'bullish' if target_bullish else 'bearish'})")
        self._calculate_trade_parameters(ohlc_ltf, current_momentum, target_bullish)
        self._save_signal(last_price)

    # =========================================================================
    # Order placement -- fixes v1's missing short=True (see module docstring)
    # =========================================================================

    def check_conditions(self) -> None:
        self._log(f"Checking conditions with state: {self.state}")
        if self.state == 0:
            self._log("State is 0, no action required")
            return

        symbol = self.symbols[0]
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self._log(f"Failed to get symbol info for {symbol}")
            return

        digits = symbol_info.digits + 1
        tp = round(self.tp, digits)
        sl = round(self.sl, digits)
        entry = round(self.entry, digits)

        if self.state == 1:
            self._log(f"Long setup valid - SL: ${sl}, ENTRY: ${entry}, TP: ${tp}")
            self.execute(symbol=symbol, sl=sl, tp=tp, entry=entry, is_limit=True, is_eod=True, short=False)
        elif self.state == -1:
            self._log(f"Short setup valid - SL: ${sl}, ENTRY: ${entry}, TP: ${tp}")
            self.execute(symbol=symbol, sl=sl, tp=tp, entry=entry, is_limit=True, is_eod=True, short=True)
        else:
            self._log(f"Unknown state: {self.state}")

    # =========================================================================
    # Position sizing -- delegates to shared fixed-fraction helper on MetaStrategy
    # (moved to base class so metamtou, metamlp, etc. can reuse it).
    # =========================================================================

    def _resolve_position_size(self, symbol, price=None, sl=None):
        return self._fixed_fraction_position_size(symbol, price=price, sl=sl)
