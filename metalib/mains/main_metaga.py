from datetime import timedelta, datetime
from metalib.metaga import MetaGA

import pytz
import time
import schedule
import csv

import MetaTrader5 as mt5

import warnings

warnings.filterwarnings("ignore")


def run_strategies():
    end_time = datetime.now(pytz.utc) + timedelta(hours=3)
    start_time = end_time - timedelta(days=40)
    for metaga in metaga_list:
        metaga.run(start_time, end_time)

def fit_strategies():
    for metaga in metaga_list:
        metaga.fit()


def main():
    global metaga_list
    metaga_list = []

    with open('../config/metaga_args.csv', mode='r') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            # convert parameter values to appropriate types
            symbol = row['symbol']
            timeframe = eval(row['timeframe'])  # Evaluate the string representation to get the actual constant
            tag = row['tag']
            print(row['active_hours'])
            ah = eval(row['active_hours'])
            rf = eval(row['risk_factor'])  # Evaluate the string representation to get the actual value

            # initialize metaga objects with the retrieved parameters
            metaga = MetaGA(
                symbols=[symbol],
                timeframe=timeframe,
                tag=tag,
                active_hours=ah,
                risk_factor=rf)

            metaga_list.append(metaga)

    # Connect to MetaTrader 5 terminal and fit strategies
    for metaga in metaga_list:
        metaga.connect()
        metaga.fit()

    # Schedule the fit method to run once every day
    schedule.every().day.do(fit_strategies)

    # Schedule the strategy runs to execute every hour
    schedule.every().hour.at(":00").do(run_strategies)
    # schedule.every().minute.do(run_strategies)

    # Run the scheduling loop
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()