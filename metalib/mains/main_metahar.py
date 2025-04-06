from datetime import timedelta, datetime
from metalib.metahar import MetaHAR
import pytz
import time
import schedule
import csv
import logging
import MetaTrader5 as mt5
import warnings

warnings.filterwarnings("ignore")


def run_strategies():
    end_time = datetime.now(pytz.utc) + timedelta(hours=3)
    start_time = end_time - timedelta(days=10)
    for metahar in metahar_list:
        try:
            metahar.run(start_time, end_time)
        except Exception as e:
            logging.error(f"An error occurred while running the strategy {metahar.tag}: {e}")
            continue


def fit_strategies():
    for metahar in metahar_list:
        try:
            metahar.fit()
        except Exception as e:
            logging.error(f"An error occurred while fitting the strategy {metahar.tag}: {e}")
            continue


def main():
    global metahar_list
    metahar_list = []

    with open('../config/metahar_args.csv', mode='r') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            # convert parameter values to appropriate types
            symbols = eval(row['symbols'])  # Should be a list of symbols
            predicted_symbol = row['predicted_symbol']
            timeframe = eval(row['timeframe'])
            tag = row['tag']
            ah = eval(row['active_hours'])

            # Optional parameters with defaults
            short_factor = eval(row.get('short_factor', '60'))
            long_factor = eval(row.get('long_factor', '480'))  # 8*60

            # initialize metahar objects with the retrieved parameters
            metahar = MetaHAR(
                symbols=symbols,
                predicted_symbol=predicted_symbol,
                timeframe=timeframe,
                tag=tag,
                active_hours=ah,
                short_factor=short_factor,
                long_factor=long_factor)

            metahar_list.append(metahar)

    # Connect to MetaTrader 5 terminal and fit strategies
    for metahar in metahar_list:
        try:
            metahar.connect()
            metahar.fit()
        except Exception as e:
            logging.error(f"An error occurred while initializing the strategy {metahar.tag}: {e}")
            continue

    # Schedule the fit method to run once every day
    schedule.every().day.do(fit_strategies)

    # Schedule the strategy runs to execute every 15 minutes at :15, :30, :45, :00
    schedule.every().hour.at(":00").do(run_strategies)
    schedule.every().hour.at(":15").do(run_strategies)
    schedule.every().hour.at(":30").do(run_strategies)
    schedule.every().hour.at(":45").do(run_strategies)

    # Run the scheduling loop
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logging.error(f"An error occurred in the scheduling loop: {e}")
        time.sleep(1)


if __name__ == "__main__":
    main()
