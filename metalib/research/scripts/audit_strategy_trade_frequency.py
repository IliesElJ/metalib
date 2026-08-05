"""
Live MT5 audit: trades/month per SL-based strategy tag over the last N days.

Distinguishes strategies by comment prefix on OPEN deals (entry==0). Emits
a pickle at metalib/research/data/strategy_trade_frequency.pkl consumed by
generate_metafvg_v2_sizing_report.py.

Run from the metalib repo root:
    PYTHONPATH=. python metalib/research/scripts/audit_strategy_trade_frequency.py
"""
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd


# ----- config -----
LOOKBACK_DAYS = 180
# canonical (strategy_name -> tag prefixes we consider to belong to it)
STRATEGY_PREFIXES = {
    "metafvg":     ["metafvg_"],
    "metafvg_v2":  ["fvg2_"],
    "metamtou":    ["mtou-", "test-mtou-"],
    "metaob":      ["metaob_"],
    "metamlp":     ["metamlp_"],
    "metago":      ["metago_"],
    "metane":      ["metane_"],
    "metaga":      ["metaga_"],
}
# ------------------


def _classify(comment: str) -> str:
    if not isinstance(comment, str):
        return "unknown"
    c = comment.lower()
    for name, prefixes in STRATEGY_PREFIXES.items():
        if any(c.startswith(p) for p in prefixes):
            return name
    if c.startswith("meta"):
        return "unknown_meta"
    return "other"


def main():
    if not mt5.initialize():
        print(f"MT5 initialize() failed: {mt5.last_error()}")
        return 1

    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS)
    print(f"Fetching deals from {start:%Y-%m-%d} to {end:%Y-%m-%d}")

    deals = mt5.history_deals_get(start, end)
    if deals is None or len(deals) == 0:
        print(f"No deals returned. last_error={mt5.last_error()}")
        mt5.shutdown()
        return 1

    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    df["time"] = pd.to_datetime(df["time"], unit="s")
    print(f"Total deals fetched: {len(df)}")

    # Open deals only (entry == 0 => IN)
    opens = df[df["entry"] == 0].copy()
    print(f"Open deals: {len(opens)}")

    opens["strategy"] = opens["comment"].apply(_classify)

    # Trades / month per strategy
    per_strategy = defaultdict(dict)
    months = LOOKBACK_DAYS / 30.44
    for strat, grp in opens.groupby("strategy"):
        if strat in ("other",):
            continue
        n = len(grp)
        n_tags = grp["comment"].nunique()
        n_symbols = grp["symbol"].nunique() if "symbol" in grp.columns else None
        per_strategy[strat] = {
            "total_open_deals": int(n),
            "trades_per_month_total": n / months,
            "trades_per_month_per_tag": (n / months / n_tags) if n_tags else 0,
            "distinct_tags": int(n_tags),
            "distinct_symbols": int(n_symbols) if n_symbols else None,
            "lookback_days": LOOKBACK_DAYS,
            "period_start": start.strftime("%Y-%m-%d"),
            "period_end": end.strftime("%Y-%m-%d"),
        }

    # Also compute per-tag frequency for sanity
    per_tag = (opens.groupby(["strategy", "comment"])
                    .size()
                    .reset_index(name="n_trades")
                    .sort_values(["strategy", "n_trades"], ascending=[True, False]))
    per_tag["trades_per_month"] = per_tag["n_trades"] / months

    print("\n=== Trades per strategy (last {} days) ===".format(LOOKBACK_DAYS))
    for strat, info in sorted(per_strategy.items()):
        print(f"  {strat:12s}: {info['total_open_deals']:5d} trades "
              f"({info['trades_per_month_total']:6.1f}/mo across "
              f"{info['distinct_tags']} tags, "
              f"{info['trades_per_month_per_tag']:5.1f}/mo per tag)")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_pkl = os.path.join(out_dir, "strategy_trade_frequency.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "per_strategy": dict(per_strategy),
            "per_tag": per_tag,
            "lookback_days": LOOKBACK_DAYS,
            "period_start": start,
            "period_end": end,
        }, f)
    print(f"\nWrote frequency data to {out_pkl}")
    mt5.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
