"""
MT5 integration test for MetaFVGv2. No dry-run/paper-trading mode exists
anywhere in this codebase (confirmed by research before writing this) --
check_conditions() calling execute() hits mt5.order_send() for real if a
signal fires, full stop, on whatever account the local MT5 terminal happens
to be logged into. This script's FIRST and only unconditional action is a
read-only mt5.account_info() check to confirm that account is a demo
account before doing anything else that could place an order.

If a genuine Spearman-gate signal fires during the test, this WILL place a
real (demo-money) limit order -- that's the point, to verify the full
pipeline end to end including order routing. If no signal fires (entirely
possible/likely depending on real-time market conditions), the test still
verifies connect -> fit -> signals -> check_conditions runs cleanly, just
without exercising the execute() path that tick.

Usage:
    PYTHONPATH=. "<adonys python>" metalib/mains/test_metafvg_v2_mt5.py <SYMBOL>
"""
import sys
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pytz

from metalib.metafvg_v2 import MetaFVGv2

EXPECTED_DEMO_SERVER_SUBSTRING = "Demo"


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"

    print("=" * 70)
    print("STEP 1: connect + verify account is a demo account (read-only)")
    print("=" * 70)
    if not mt5.initialize():
        print(f"mt5.initialize() FAILED: {mt5.last_error()}")
        sys.exit(1)

    account_info = mt5.account_info()
    if account_info is None:
        print(f"mt5.account_info() FAILED: {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    print(f"Login: {account_info.login}")
    print(f"Server: {account_info.server}")
    print(f"Name: {account_info.name}")
    print(f"Currency: {account_info.currency}")
    print(f"Balance: {account_info.balance}")
    print(f"Equity: {account_info.equity}")
    print(f"Trade mode (0=demo,1=contest,2=real): {account_info.trade_mode}")

    is_demo = account_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
    if not is_demo:
        print()
        print("!" * 70)
        print(f"ABORTING: account trade_mode is NOT demo (trade_mode={account_info.trade_mode}, "
              f"server={account_info.server}). Refusing to proceed with an order-capable test "
              f"against a non-demo account. No further action taken.")
        print("!" * 70)
        mt5.shutdown()
        sys.exit(1)

    print(f"\nConfirmed DEMO account (server={account_info.server}). Proceeding.")
    mt5.shutdown()  # MetaFVGv2.connect() will re-initialize; keeping the flow identical to production

    print()
    print("=" * 70)
    print(f"STEP 2: construct MetaFVGv2({symbol}), connect, fit")
    print("=" * 70)
    strat = MetaFVGv2(
        symbols=[symbol],
        timeframe=mt5.TIMEFRAME_M15,
        size_position=0.01,  # fallback only; _resolve_position_size computes the real size
        tag="metafvg_v2_test",
        limit_number_position=2,
    )
    strat.debug = True  # keep output on stdout instead of redirecting to a log file
    strat.connect()
    strat.fit()

    print()
    print("=" * 70)
    print("STEP 3: load data + generate signals")
    print("=" * 70)
    end_time = datetime.now(pytz.utc) + timedelta(hours=3)
    start_time = end_time - timedelta(days=30)
    strat.loadData(start_time, end_time)
    strat.signals()

    print()
    print(f"Resulting state: {strat.state} (1=long, -1=short, 0=no signal)")
    print(f"Entry: {strat.entry}")
    print(f"SL: {strat.sl}")
    print(f"TP: {strat.tp}")

    print()
    print("=" * 70)
    print("STEP 4: check_conditions() -- WILL place a real demo order if state != 0")
    print("=" * 70)
    strat.check_conditions()

    print()
    print("Test complete.")


if __name__ == "__main__":
    main()
