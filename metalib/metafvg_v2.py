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
   the same sizing *convention* as its own backtest instead: vectorbt's
   size_type='percent' there means "N% of current equity, converted to
   units at the trade's price" -- _resolve_position_size below reimplements
   that against real MT5 account equity and symbol contract specs, since
   nothing in this codebase computed real position sizing dynamically
   before now (see metastrategy.py's _resolve_position_size hook, added
   alongside this class, default-preserving for every other strategy).
   RISK_FRACTION here is 1% (0.01), deliberately more conservative than
   the backtest's 2% (0.02, see report Sec. 5/6) for initial live rollout --
   a smaller fraction of the exact backtested config, not a claim that 1%
   was itself backtested.

   NOTE on metascal.py's MetaScale: that's a separate, dashboard-triggered
   (not automated) portfolio-wide risk-budgeting tool that periodically
   rewrites size_position for the strategies listed in its
   DEFAULT_STRATEGY_PARAMS (metafvg, metago, metane, metaga, metaob) via a
   correlation-aware CVXPY optimization across the whole book. It uses
   tick_value/tick_size (P&L-per-tick sensitivity, for its covariance calc)
   -- a different quantity from this class's contract_size*price (notional
   value per lot, for replicating the backtest's percent-of-equity
   convention); the two aren't interchangeable for their respective
   purposes. metafvg_v2 is deliberately NOT in DEFAULT_STRATEGY_PARAMS: it
   manages its own sizing independently and MetaScale's optimizer has no
   visibility into it, by design decision (not an oversight) -- if that
   changes, _resolve_position_size below would need to defer to
   size_position instead of overriding it, or MetaScale would need to
   account for v2's risk contribution some other way.
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
    RISK_FRACTION = 0.01  # equity-percent sizing, same convention as the
                           # backtest's vectorbt size_type='percent' but at
                           # 1% instead of the backtested 2% -- more
                           # conservative for initial live rollout

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
        size_position here is a FALLBACK ONLY: the static lot size used if
        _resolve_position_size can't compute a dynamic one (e.g. MT5
        account_info()/symbol_info() unavailable). Normal operation ignores
        it in favor of equity-percent sizing.
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
    # Position sizing -- equity-percent, matching the backtest convention
    # =========================================================================

    def _resolve_position_size(self, symbol, price=None):
        """
        volume = (RISK_FRACTION * account_equity) / (contract_size * price),
        converted to account currency if the symbol's profit currency
        differs, then rounded to the broker's volume_step/min/max. Falls
        back to self.size_position (static, from YAML) on any failure --
        never raises, never blocks a trade over a sizing error.
        """
        try:
            account_info = mt5.account_info()
            if account_info is None or price is None or price <= 0:
                self._log(f"{symbol}: sizing fallback (account_info unavailable or bad price)")
                return self.size_position

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self._log(f"{symbol}: sizing fallback (symbol_info unavailable)")
                return self.size_position

            equity = account_info.equity
            account_currency = account_info.currency
            contract_size = symbol_info.trade_contract_size
            profit_currency = symbol_info.currency_profit

            value_per_lot = contract_size * price
            if profit_currency != account_currency:
                conv_rate = self._get_conversion_rate(profit_currency, account_currency)
                if conv_rate is None:
                    self._log(f"{symbol}: sizing fallback (no {profit_currency}->{account_currency} "
                              f"conversion rate found)")
                    return self.size_position
                value_per_lot *= conv_rate

            if value_per_lot <= 0:
                self._log(f"{symbol}: sizing fallback (non-positive value_per_lot)")
                return self.size_position

            notional_target = self.RISK_FRACTION * equity
            raw_volume = notional_target / value_per_lot

            volume_min = symbol_info.volume_min
            volume_max = symbol_info.volume_max
            volume_step = symbol_info.volume_step
            stepped = round(raw_volume / volume_step) * volume_step if volume_step > 0 else raw_volume
            volume = round(max(volume_min, min(volume_max, stepped)), 2)

            self._log(f"{symbol}: equity={equity:.2f} {account_currency} "
                      f"notional_target={notional_target:.2f} value_per_lot={value_per_lot:.4f} "
                      f"raw_volume={raw_volume:.4f} -> volume={volume}")
            return volume
        except Exception as e:
            self._log(f"{symbol}: sizing fallback (exception: {e})")
            return self.size_position

    @staticmethod
    def _get_conversion_rate(from_ccy: str, to_ccy: str):
        """Best-effort spot conversion rate from from_ccy to to_ccy via a
        directly-tradeable MT5 symbol (either FROMTO or TOFROM), or None if
        no such symbol is found/visible."""
        if from_ccy == to_ccy:
            return 1.0
        for pair, invert in ((f"{from_ccy}{to_ccy}", False), (f"{to_ccy}{from_ccy}", True)):
            info = mt5.symbol_info(pair)
            if info is None:
                continue
            if not info.visible:
                mt5.symbol_select(pair, True)
            tick = mt5.symbol_info_tick(pair)
            if tick is None or tick.bid <= 0:
                continue
            return (1.0 / tick.bid) if invert else tick.bid
        return None
