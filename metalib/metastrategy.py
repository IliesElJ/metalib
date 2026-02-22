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

    def loadData(self, start_date=None, end_date=None):
        """
        Loads historical data for the specified symbols and timeframe.
        """
        for symbol in self.symbols:
            rates = mt5.copy_rates_range(symbol, self.timeframe, start_date, end_date)

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
        Saves the signal data to a single HDF5 file ("signals.hdf5"), organized by tag.
        """
        signal_line = self.signalData
        if not isinstance(signal_line, pd.Series):
            return
        if "timestamp" not in signal_line or pd.isna(signal_line["timestamp"]):
            return

        row_df = signal_line.to_frame().T
        row_df["timestamp"] = pd.to_datetime(row_df["timestamp"])

        file_name = SIGNALS_FILE
        # Sanitize tag for HDF5 key (replace hyphens, dots)
        key = "/" + self.tag.replace("-", "_").replace(".", "_")

        ensure_directories()

        with pd.HDFStore(file_name, mode="a") as store:
            if key in store:
                existing = store[key]
                updated = pd.concat([existing, row_df], ignore_index=True)
                store.put(key, updated)
            else:
                store.put(key, row_df)

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

        # Construct the request dictionary
        request = {
            "action": (
                mt5.TRADE_ACTION_DEAL if not is_limit else mt5.TRADE_ACTION_PENDING
            ),
            "symbol": symbol,
            "sl": sl if sl else 0.0,
            "tp": tp if tp else 0.0,
            "volume": float(self.size_position),
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
