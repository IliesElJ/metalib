from abc import ABC, abstractmethod
import MetaTrader5 as mt5
import pandas as pd
import requests
from metalib.utils import load_hist_data
# from sqlalchemy import create_engine

# POSTGRESQL Connexion
database_price = "price"
database_signal = "signal"

username = "postgres"
password = "toor"
host = "localhost"
port = "5432"


class MetaStrategy(ABC):
    """
    Abstract base class for quantitative trading strategies.
    """

    def __init__(self,
                 symbols,
                 timeframe,
                 tag, 
                 active_hours=None,
                 long_only=False,
                 short_only=False,
                 save_to_sql_db=False, 
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
        self.active_hours = active_hours
        self.long_only = long_only
        self.short_only = short_only
        self.save_to_sql_db = save_to_sql_db

        if self.save_to_sql_db:
            self.engine_signal = create_engine(
                f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database_signal}")

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
                print(f"Failed to load data for {symbol}, error code =", mt5.last_error())
                return

            self.data[symbol] = pd.DataFrame(rates)
            self.data[symbol]['time'] = pd.to_datetime(self.data[symbol]['time'], unit='s')
            self.data[symbol] = self.data[symbol].set_index('time')

        print(f"Last time in the index: {self.data[symbol].index[-1]}")

    def save_price_data_to_db(self):
        """
        Saves the price data from self.data to the database, creating one table per symbol.
        """
        for symbol, df in self.data.items():
            # Ensure the DataFrame has the 'symbol' column, if it's not already present
            if 'symbol' not in df.columns:
                df['symbol'] = symbol

            for column, dtype in df.dtypes.items():
                if dtype == 'uint64':
                    df[column] = df[column].astype('int64')  # or 'float64' if necessary

            table_name = f"prices_{symbol.lower()}"  # Define the table name based on the symbol
            # df.reset_index().to_sql(table_name, self.engine_price, if_exists='append', index=False)

    def save_signal_data_to_db(self):
        """
        Saves the price data from self.data to the database, creating one table per symbol.
        """

        df = self.signals_data
        table_name = f"signals_{self.tag}"  # Define the table name based on the symbol
        df.reset_index().to_sql(table_name, self.engine_signal, if_exists='append', index=False)

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

    def execute(self, symbol, volume, sl=None, tp=None, short=False, is_stop=False, is_limit=False, entry=None,
                expiration_date=None):
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
        """
        # Determine order type
        if short:
            order_type = mt5.ORDER_TYPE_SELL_LIMIT if is_limit else mt5.ORDER_TYPE_SELL_STOP if is_stop else mt5.ORDER_TYPE_SELL
        else:
            order_type = mt5.ORDER_TYPE_BUY_LIMIT if is_limit else mt5.ORDER_TYPE_BUY_STOP if is_stop else mt5.ORDER_TYPE_BUY

        if (is_limit or is_stop) and entry is None:
            raise ValueError("Entry price must be provided for limit or stop orders.")

        if not entry:
            price = mt5.symbol_info_tick(symbol).ask

        # Construct the request dictionary
        request = {
            "action": mt5.TRADE_ACTION_DEAL if not is_limit else mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "sl": sl if sl else 0.0,
            "tp": tp if tp else 0.0,
            "volume": float(volume),
            "type": order_type,
            "price": entry if is_limit or is_stop else price,
            "comment": self.tag,
            "deviation": 0,
            "magic": 0,
            "type_time": mt5.ORDER_TIME_GTC,  # Good till cancelled
            "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or cancel
        }

        # Add expiration date if provided
        if expiration_date is not None:
            request["expiration"] = expiration_date

        # Execute the order
        result = mt5.order_send(request)

        print(f"Order: {request}")
        print(f"Order sent: {result} for strategy: {self.tag}")

        return result

    def run(self, start_date, end_date):
        """
        Main method to run the strategy.
        """

        self.loadData(start_date, end_date)
        self.signals()

        if self.save_to_sql_db:
            self.save_signal_data_to_db()

        current_hour = self.data[next(iter(self.data))].index[-1].hour

        if not self.are_positions_with_tag_open():
            if self.active_hours is not None and current_hour not in self.active_hours and self.state != -2:
                print(f"Current hour ({current_hour}) is not within active hours. Strategy will not run.")
                return

            # Check if long_only
            if (self.long_only and self.state == -1) or (self.short_only and self.state == 1):
                print("Long only or short only strategy, and the current state is not in the direction.")
                return

        self.check_conditions()

    def are_positions_with_tag_open(self, position_type=None):
        # Retrieve all open positions
        open_positions = mt5.positions_get()
        if open_positions is None:
            print("No positions,", mt5.last_error())
            return False
        else:
            # Check each position for the tag and optionally the position type
            for position in open_positions:
                if position.comment != self.tag:
                    continue  # Skip if the tag does not match

                # If position_type is specified, further filter by position type
                if position_type is not None:
                    if position_type.lower() == "buy" and position.type == mt5.ORDER_TYPE_BUY:
                        return True
                    elif position_type.lower() == "sell" and position.type == mt5.ORDER_TYPE_SELL:
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
                print(f"Position with ticket {position.ticket} does not have the tag '{self.tag}', skipping.")
                continue

            # Check for position type, if specified
            if position_type is not None:
                if position_type.lower() == "buy" and position.type != mt5.ORDER_TYPE_BUY:
                    print(f"Position with ticket {position.ticket} is not a buy position, skipping.")
                    continue
                elif position_type.lower() == "sell" and position.type != mt5.ORDER_TYPE_SELL:
                    print(f"Position with ticket {position.ticket} is not a sell position, skipping.")
                    continue

            # Close position
            print(f"Closing position with ticket {position.ticket} and tag '{self.tag}'")
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
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

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
        bot_token = '6879531129:AAFwzA9vu2xt8-2zMvPKTyyTgfKMdZlpBWw'
        chat_id = '295737723'

        send_text = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={message}'
        response = requests.get(send_text)

        chat_id = '895011343'

        send_text = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={message}'
        response = requests.get(send_text)

        chat_id = '5797648513'

        send_text = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={message}'
        response = requests.get(send_text)

        return response.json()
