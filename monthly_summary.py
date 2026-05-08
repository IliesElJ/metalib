"""
Trading activity summary for the past two weeks using metalib metadash functions.
"""

import sys
sys.path.insert(0, r"C:\Users\Hermes\PycharmProjects\metalib\metalib\metadash")

from datetime import datetime, timedelta
import pandas as pd

from utils.mt5_utils import (
    initialize_mt5,
    get_historical_data,
    process_deals_data,
    get_account_info,
)
from utils.metrics import (
    strategy_metrics,
    calculate_daily_performance,
    calculate_strategy_type_metrics,
)

# ── Connect ──────────────────────────────────────────────────────────────────
ok, msg = initialize_mt5()
print(msg)
if not ok:
    raise SystemExit(1)

# ── Date range: past 14 days ─────────────────────────────────────────────────
to_date   = datetime.utcnow()
from_date = to_date - timedelta(days=14)
print(f"\nPeriod : {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')} (UTC)\n")

# ── Account snapshot ─────────────────────────────────────────────────────────
acct = get_account_info()
print("=" * 55)
print("ACCOUNT")
print("=" * 55)
for k in ("login", "name", "server", "currency", "balance", "equity", "profit", "margin"):
    if k in acct:
        print(f"  {k:<10}: {acct[k]}")

# ── Fetch history ─────────────────────────────────────────────────────────────
orders, deals, err = get_historical_data(from_date, to_date)
if err:
    print(f"\nError fetching history: {err}")
    raise SystemExit(1)

print(f"\nRaw deals fetched : {len(deals) if deals else 0}")
print(f"Raw orders fetched: {len(orders) if orders else 0}")

# ── Process deals ─────────────────────────────────────────────────────────────
merged = process_deals_data(deals)

if merged is None or merged.empty:
    print("\nNo closed bot trades found in the period.")
    raise SystemExit(0)

print(f"Matched bot trades : {len(merged)}\n")

# ── Overall metrics ───────────────────────────────────────────────────────────
print("=" * 55)
print("OVERALL METRICS  (past 30 days)")
print("=" * 55)
overall = strategy_metrics(merged, account_size=acct.get("balance", 100_000))
for label, val in overall.items():
    if isinstance(val, float):
        print(f"  {label:<30}: {val:>10.2f}")
    else:
        print(f"  {label:<30}: {val:>10}")

# ── Per-strategy-type breakdown ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("BY STRATEGY TYPE")
print("=" * 55)
st_metrics = calculate_strategy_type_metrics(merged, account_size=acct.get("balance", 100_000))
cols = ["Number of Trades", "Total Profit", "Win Rate (%)", "Profit Factor", "RRR", "Max Drawdown (%)"]
available_cols = [c for c in cols if c in st_metrics.columns]
print(st_metrics[available_cols].to_string())

# ── Daily P&L ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("DAILY P&L  (past 30 days)")
print("=" * 55)
daily = calculate_daily_performance(merged)
daily["cumulative"] = daily["profit"].cumsum()
daily.columns = ["Date", "Daily P&L", "Cumulative P&L"]
print(daily.to_string(index=False))

# ── Symbol breakdown ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("BY SYMBOL")
print("=" * 55)
merged["total_profit"] = merged["profit_open"] + merged["profit_close"]
sym_summary = (
    merged.groupby("symbol_open")
    .agg(
        trades=("total_profit", "count"),
        total_pnl=("total_profit", "sum"),
        win_rate=("total_profit", lambda x: 100 * (x > 0).mean()),
        avg_pnl=("total_profit", "mean"),
    )
    .sort_values("total_pnl", ascending=False)
)
print(sym_summary.to_string())
