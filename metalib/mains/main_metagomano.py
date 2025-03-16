from datetime import timedelta, datetime
from metagomano import MetaGO
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
    for metago in metago_list:
        metago.run(start_time, end_time)

def fit_strategies():
    for metago in metago_list:
        metago.fit()


def main():
    global metago_list
    metago_list = []

    with open('metago_args.csv', mode='r') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            # convert parameter values to appropriate types
            symbol = row['symbol']
            timeframe = eval(row['timeframe'])  # Evaluate the string representation to get the actual constant
            tag = row['tag']
            print(row['active_hours'])
            ah = eval(row['active_hours'])
            rf = eval(row['risk_factor'])  # Evaluate the string representation to get the actual value

            # initialize metago objects with the retrieved parameters
            metago = MetaGO(
                symbols=[symbol],
                timeframe=timeframe,
                tag=tag,
                active_hours=ah,
                risk_factor=rf)

            metago_list.append(metago)

    # Connect to MetaTrader 5 terminal and fit strategies
    for metago in metago_list:
        metago.connect()
        metago.fit()

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