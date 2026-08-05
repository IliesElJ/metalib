from abc import ABC, abstractmethod
import MetaTrader5 as mt5
import pandas as pd
from requests import get
import sys
from datetime import datetime
from metalib.constants import SIGNALS_FILE, LOG_EXTENSION, ensure_directories
import os
from uuid import uuid4
import traceback

MT5_MAX_TAG_LENGTH = 16


class MetaStrategy(ABC):
    """
    Abstract base class for quantitative trading strategies.
    """

    def __init__(
        self,
        symbols,
        timeframe,
        tag,
        size_position,
        active_hours=None,
        long_only=False,
        short_only=False,
    ):
        """
        Initializes the strategy with multiple symbols and a timeframe.

        :param symbols: List of symbols (str) to trade.
        :param timeframe: Timeframe for the strategy.
        :param tag: Tag to be used when taking the positions
        """
        self.symbols = symbols
        self.timeframe = timeframe
        self.data = {}
        self.tag = tag
        self.size_position = size_position
        self.active_hours = active_hours if isinstance(active_hours, list) else None
        self.state = 0
        self.long_only = long_only
        self.short_only = short_only
        self.signalData = None
        self.debug = False

    def connect(self):
        """
        Establishes a connection to the MetaTrader 5 terminal.
        """
        if not mt5.initialize():
            print("Initialize() failed, error code =", mt5.last_error())
            mt5.shutdown()

    def loadData(self, start_date=None, end_date=None, timeframe=None):
        """
        Loads historical data for the specified symbols and timeframe.
        """

        if timeframe is None:
            timeframe = self.timeframe

        for symbol in self.symbols:
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

            if rates is None:
                print(
                    f"Failed to load data for {symbol}, error code =", mt5.last_error()
                )
                return

            self.data[symbol] = pd.DataFrame(rates)
            self.data[symbol]["time"] = pd.to_datetime(
                self.data[symbol]["time"], unit="s"
            )
            self.data[symbol] = self.data[symbol].set_index("time")
            print(f"Last time in the index: {self.data[symbol].index[-1]}")

    def save_signal_data_to_db(self):
        """
        Appends one signal row to signals.hdf5 under a per-tag key.
        Uses table format so each write is O(1) — no read-modify-write.
        """
        signal_line = self.signalData
        if not isinstance(signal_line, pd.Series):
            return
        if "timestamp" not in signal_line or pd.isna(signal_line["timestamp"]):
            return

        row_df = signal_line.to_frame().T
        row_df["timestamp"] = pd.to_datetime(row_df["timestamp"])
        if "tag" not in row_df.columns:
            row_df["tag"] = self.tag

        file_name = SIGNALS_FILE
        key = "/" + self.tag.replace("-", "_").replace(".", "_")
        ensure_directories()

        # Pad all string/object columns to 100 chars so the schema is stable across runs.
        min_itemsize = {
            col: 100 for col in row_df.select_dtypes(include="object").columns
        }

        try:
            with pd.HDFStore(file_name, mode="a") as store:
                store.append(
                    key,
                    row_df,
                    format="table",
                    data_columns=True,
                    min_itemsize=min_itemsize or None,
                )
        except Exception as e:
            print(f"Error saving signal data: {str(e)}")

    def get_vol_prediction(self, symbol=None, max_age_minutes=300):
        """
        Latest MetaHAR vol prediction for a symbol from the shared store
        (metalib/volstore.py), or None if absent/stale. Prediction is the
        expected change in log realized variance over the next model bar
        ((ts, ts + horizon_min] — see volstore docstring).
        """
        from metalib.volstore import get_latest

        target = symbol if symbol is not None else self.symbols[0]
        try:
            return get_latest(target, max_age_minutes=max_age_minutes)
        except Exception as e:
            print(f"{self.tag}::: volstore read failed: {e}")
            return None

    @abstractmethod
    def signals(self):
        """
        Generates trading signals. Must be implemented by each strategy.
        """
        pass

    @abstractmethod
    def check_conditions(self):
        """
        Generates trading signals. Must be implemented by each strategy.
        """
        pass

    @abstractmethod
    def fit(self):
        """
        Fit potential ML/Stat models before running run()
        """
        pass

    def _resolve_position_size(self, symbol, price=None, sl=None):
        """
        Returns the order volume (MT5 lots) for `symbol` at `price`.

        Default behavior, unchanged for every existing strategy: the static
        per-instance self.size_position set at construction. Subclasses that
        want dynamic sizing override this method. `sl` is passed through so
        subclasses can implement SL-distance-based risk sizing.
        """
        return self.size_position

    # =========================================================================
    # Shared fixed-fraction risk sizing (used by SL-based strategies)
    # =========================================================================

    NOTIONAL_CAP_MULT = 5.0
    FX_MIN_SL_PIPS = 5

    def _fixed_fraction_position_size(self, symbol, price=None, sl=None):
        """
        Sizes so that hitting the stop loss costs exactly self.size_position *
        balance in account currency:

            volume = (size_position * balance) / (sl_distance * vpu)

        where vpu = contract_size * fx_conv(profit_ccy -> account_ccy) is the
        P&L per lot per 1.0 unit of price move.

        Safety layers:
          - FX minimum SL floor: for FX pairs (trade_calc_mode == 0), sl_distance
            used for sizing is raised to FX_MIN_SL_PIPS * pip_size when smaller.
            The order's actual SL price is unchanged.
          - Notional cap: volume * contract_size * price (in account ccy) <=
            NOTIONAL_CAP_MULT * balance. Trims volume down when tight SLs would
            otherwise blow past this.

        Falls back to self.size_position as a raw lot count on any failure
        (missing MT5 info, missing SL, missing FX conversion). Never raises.
        """
        try:
            account_info = mt5.account_info()
            if account_info is None or price is None or price <= 0:
                self._log(f"{symbol}: sizing fallback (account_info unavailable or bad price)")
                return self.size_position

            if not sl or sl <= 0:
                self._log(f"{symbol}: sizing fallback (no SL provided)")
                return self.size_position

            sl_distance = abs(price - sl)
            if sl_distance <= 0:
                self._log(f"{symbol}: sizing fallback (SL == entry price)")
                return self.size_position

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self._log(f"{symbol}: sizing fallback (symbol_info unavailable)")
                return self.size_position

            balance = account_info.balance
            account_currency = account_info.currency
            contract_size = symbol_info.trade_contract_size
            profit_currency = symbol_info.currency_profit

            value_per_price_unit = contract_size
            if profit_currency != account_currency:
                conv_rate = self._get_conversion_rate(profit_currency, account_currency)
                if conv_rate is None:
                    self._log(f"{symbol}: sizing fallback (no {profit_currency}->{account_currency} "
                              f"conversion rate found)")
                    return self.size_position
                value_per_price_unit *= conv_rate

            if value_per_price_unit <= 0:
                self._log(f"{symbol}: sizing fallback (non-positive value_per_price_unit)")
                return self.size_position

            # Minimum SL width for FX pairs (sizing only, SL level unchanged)
            if symbol_info.trade_calc_mode == 0:
                pip_size = 0.01 if "JPY" in symbol else 0.0001
                min_sl = self.FX_MIN_SL_PIPS * pip_size
                if sl_distance < min_sl:
                    self._log(
                        f"{symbol}: SL distance {sl_distance:.6f} below "
                        f"{self.FX_MIN_SL_PIPS}-pip minimum ({min_sl:.6f}), "
                        f"clamping for sizing"
                    )
                    sl_distance = min_sl

            risk_amount = self.size_position * balance
            raw_volume = risk_amount / (sl_distance * value_per_price_unit)

            volume_min = symbol_info.volume_min
            volume_max = symbol_info.volume_max
            volume_step = symbol_info.volume_step
            stepped = round(raw_volume / volume_step) * volume_step if volume_step > 0 else raw_volume
            volume = round(max(volume_min, min(volume_max, stepped)), 2)

            # Hard notional cap
            max_notional = self.NOTIONAL_CAP_MULT * balance
            notional_per_lot = contract_size * price
            if profit_currency != account_currency:
                notional_per_lot *= value_per_price_unit / contract_size
            notional = volume * notional_per_lot
            if notional > max_notional and notional_per_lot > 0:
                volume_capped = round(max_notional / notional_per_lot, 2)
                volume_capped = round(max(volume_min, min(volume_max, volume_capped)), 2)
                self._log(
                    f"{symbol}: notional cap triggered "
                    f"({notional:.0f} > {max_notional:.0f}) -> volume {volume} -> {volume_capped}"
                )
                volume = volume_capped

            self._log(
                f"{symbol}: balance={balance:.2f} {account_currency} "
                f"risk={self.size_position*100:.2f}% -> risk_amount={risk_amount:.2f} "
                f"sl_dist={sl_distance:.6f} sl%={sl_distance/price*100:.3f}% "
                f"vpu={value_per_price_unit:.4f} raw={raw_volume:.4f} -> volume={volume}"
            )
            return volume
        except Exception as e:
            self._log(f"{symbol}: sizing fallback (exception: {e})")
            return self.size_position

    @staticmethod
    def _get_conversion_rate(from_ccy: str, to_ccy: str):
        """
        Best-effort spot conversion rate from from_ccy to to_ccy via a
        directly-tradeable MT5 symbol (either FROMTO or TOFROM), or None if
        no such symbol is found/visible.
        """
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

    def _log(self, message: str) -> None:
        """
        Default no-op-friendly logger --- most subclasses override this with
        a tag-prefixing print. Provided here so _fixed_fraction_position_size
        can log even from a strategy that hasn't defined its own _log.
        """
        print(f"{self.tag}::    {message}")

    def execute(
        self,
        symbol,
        sl=None,
        tp=None,
        short=False,
        is_stop=False,
        is_limit=False,
        entry=None,
        expiration_date=None,
        is_eod=False,
    ):
        """
        Places an order based on the given parameters.

        :param symbol: Symbol for the trading pair (str).
        :param sl: Stop loss value (float).
        :param tp: Take profit value (float).
        :param volume: Volume of the order (float).
        :param short: Whether the order is a short sale (bool).
        :param is_stop: Whether the order is a stop order (bool).
        :param is_limit: Whether the order is a limit order (bool).
        :param entry: Entry price for limit or stop orders (float), required if is_limit or is_stop is True.
        :param expiration_date: Expiration date of the order, optional (datetime).
        :param is_eod: Order expire at the end of days (bool).
        """
        # Determine order type
        if short:
            order_type = (
                mt5.ORDER_TYPE_SELL_LIMIT
                if is_limit
                else mt5.ORDER_TYPE_SELL_STOP if is_stop else mt5.ORDER_TYPE_SELL
            )
        else:
            order_type = (
                mt5.ORDER_TYPE_BUY_LIMIT
                if is_limit
                else mt5.ORDER_TYPE_BUY_STOP if is_stop else mt5.ORDER_TYPE_BUY
            )

        if (is_limit or is_stop) and entry is None:
            raise ValueError("Entry price must be provided for limit or stop orders.")

        if not entry:
            price = mt5.symbol_info_tick(symbol).ask

        # sizing_price is independent of the `price` var above (only ever set
        # when entry is falsy) so this never touches that existing branch.
        sizing_price = entry if entry is not None else price
        volume = self._resolve_position_size(symbol, sizing_price, sl=sl)

        # Construct the request dictionary
        request = {
            "action": (
                mt5.TRADE_ACTION_DEAL if not is_limit else mt5.TRADE_ACTION_PENDING
            ),
            "symbol": symbol,
            "sl": sl if sl else 0.0,
            "tp": tp if tp else 0.0,
            "volume": float(volume),
            "type": order_type,
            "price": entry if is_limit or is_stop else price,
            "comment": self.tag,
            "deviation": 0,
            "magic": 0,
            "type_time": mt5.ORDER_TIME_GTC,  # Good till cancelled or End of Day
            "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or cancel
        }

        # Add expiration date if provided
        if expiration_date is not None:
            request["expiration"] = expiration_date

        if is_eod:
            request["type_time"] = mt5.ORDER_TIME_DAY

        # Execute the order
        result = mt5.order_send(request)

        print(f"Order: {request}")
        print(f"Order sent: {result} for strategy: {self.tag}")

        return result

    def run(self, start_date, end_date):
        """
        Main method to run the strategy.

        Parameters:
        -----------
        start_date : datetime
            The start date for data retrieval
        end_date : datetime
            The end date for data retrieval

        Returns:
        --------
        bool
            True if the strategy executed successfully, False otherwise
        """
        log_file = None

        try:
            # Save original stdout to restore later
            original_stdout = sys.stdout
            if not self.debug:
                # Setup logging
                today = datetime.today().strftime("%Y-%m-%d")
                log_path = f"logs/output_{self.tag}_{today}{LOG_EXTENSION}"

                # Ensure logs directory exists
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                log_file = open(log_path, "a")
                sys.stdout = log_file

            # Data preparation phase
            try:
                self.loadData(start_date, end_date)
            except Exception as e:
                print(f"Error loading data: {str(e)}")
                return False

            try:
                self.signals()
            except Exception as e:
                err_id = uuid4().hex[:8]
                err_ctx = {
                    "err_id": err_id,
                    "exc_type": type(e).__name__,
                    "message": str(e),
                    "stack": traceback.format_exc(),  # full traceback
                    # --- domain/context you care about ---
                    "stage": "generate_signals",
                }

                print(f"Error generating signals: {err_ctx}")
                return False

            try:
                self.save_signal_data_to_db()
            except Exception as e:
                print(f"Error saving signal data: {str(e)}")

            # Get current hour from the last data point
            try:
                first_symbol = next(iter(self.data))
                if not self.data or not self.data[first_symbol].index.size:
                    print("No data available for analysis")
                    return False

                current_hour = self.data[first_symbol].index[-1].hour
            except (StopIteration, IndexError, KeyError) as e:
                print(f"Error accessing data timestamp: {str(e)}")
                return False

            # Check if we should execute the strategy
            if not self.are_positions_with_tag_open():
                # Time-based filtering (hours are in MT5 server time, typically UTC+2/+3)
                if (
                    self.active_hours is not None
                    and current_hour not in self.active_hours
                    and self.state != -2
                ):
                    print(
                        f"Current MT5 server hour ({current_hour}) not in active_hours {self.active_hours}. Skipping."
                    )
                    return False

                # Direction-based filtering
                if self.long_only and self.state == -1:
                    print(
                        "Long only strategy, but current state indicates short. Strategy will not run."
                    )
                    return False
                elif self.short_only and self.state == 1:
                    print(
                        "Short only strategy, but current state indicates long. Strategy will not run."
                    )
                    return False

            # Execute strategy logic
            try:
                self.check_conditions()
                return True
            except Exception as e:
                print(f"Error executing strategy conditions: {str(e)}")
                return False

        except Exception as e:
            # Catch any unexpected exceptions
            if sys.stdout != sys.__stdout__:
                print(f"Unexpected error in strategy execution: {str(e)}")
            else:
                # If stdout redirection failed, print to console
                print(f"Critical error in strategy execution: {str(e)}")
            return False

        finally:
            self.data = {}
            # Ensure resources are properly closed
            if log_file:
                sys.stdout = sys.__stdout__  # Restore original stdout
                try:
                    log_file.close()
                except:
                    pass

    def are_positions_with_tag_open(self, position_type=None):
        # Retrieve all open positions
        open_positions = mt5.positions_get()
        if open_positions is None:
            print("No positions,", mt5.last_error())
            return False
        else:
            # Check each position for the tag and optionally the position type
            for position in open_positions:
                if position.comment != self.tag[:MT5_MAX_TAG_LENGTH]:
                    continue  # Skip if the tag does not match

                # If position_type is specified, further filter by position type
                if position_type is not None:
                    if (
                        position_type.lower() == "buy"
                        and position.type == mt5.ORDER_TYPE_BUY
                    ):
                        return True
                    elif (
                        position_type.lower() == "sell"
                        and position.type == mt5.ORDER_TYPE_SELL
                    ):
                        return True
                    # If position type does not match, continue to the next position
                    continue

                # If no position_type specified or position type matches
                return True
            return False

    def close_all_positions(self, position_type=None):
        """Close all open positions with a given tag, optionally filtered by position type (buy/sell)."""
        open_positions = mt5.positions_get()

        if len(open_positions) == 0:
            print("No open positions found.")
            return

        for position in open_positions:
            # Check for tag
            if position.comment != self.tag:
                print(
                    f"Position with ticket {position.ticket} does not have the tag '{self.tag}', skipping."
                )
                continue

            # Check for position type, if specified
            if position_type is not None:
                if (
                    position_type.lower() == "buy"
                    and position.type != mt5.ORDER_TYPE_BUY
                ):
                    print(
                        f"Position with ticket {position.ticket} is not a buy position, skipping."
                    )
                    continue
                elif (
                    position_type.lower() == "sell"
                    and position.type != mt5.ORDER_TYPE_SELL
                ):
                    print(
                        f"Position with ticket {position.ticket} is not a sell position, skipping."
                    )
                    continue

            # Close position
            print(
                f"Closing position with ticket {position.ticket} and tag '{self.tag}'"
            )
            ticket = position.ticket
            self.close_position(ticket)

    def close_position(self, deal_id):
        """Close a position by deal ID."""
        open_positions = mt5.positions_get(ticket=deal_id)

        if not open_positions:
            print(f"Position with deal ID {deal_id} not found.")
            return

        position = open_positions[0]
        symbol = position.symbol
        volume = position.volume

        # Determine the order type (BUY or SELL) based on the position type
        order_type = (
            mt5.ORDER_TYPE_SELL
            if position.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )

        # Get the current bid or ask price
        if order_type == mt5.ORDER_TYPE_BUY:
            price = mt5.symbol_info_tick(symbol).bid
        else:
            price = mt5.symbol_info_tick(symbol).ask

        # Create a close request
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "position": deal_id,
            "price": price,
            "magic": 234000,
            "comment": "Close trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send the close request
        result = mt5.order_send(close_request)

        # Check if the close order was successful
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to close order {deal_id}: {result.comment}")
        else:
            print(f"Order {deal_id} successfully closed!")

    def send_telegram_message(self, message):
        bot_token = "6879531129:AAFwzA9vu2xt8-2zMvPKTyyTgfKMdZlpBWw"
        chat_ids = ("295737723", "895011343", "5797648513")
        response = None

        for chat_id in chat_ids:
            send_text = f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={message}"
            response = get(send_text)

        return response.json()

    def get_positions_info(self):
        # Ensure connected to MT5
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            return None, None

        # Retrieve all positions
        positions = mt5.positions_get()
        if positions is None:
            print("No positions found, error code =", mt5.last_error())
            return None, None

        # Filter positions based on the comment
        filtered_positions = [pos for pos in positions if pos.comment == self.tag]

        # Calculate mean entry price and count positions
        if filtered_positions:
            total_volume = sum(pos.volume for pos in filtered_positions)
            mean_entry_price = (
                sum(pos.price_open * pos.volume for pos in filtered_positions)
                / total_volume
            )
            num_positions = len(filtered_positions)
        else:
            mean_entry_price = 0
            num_positions = 0

        # Return the mean entry price and number of positions
        return mean_entry_price, num_positions
