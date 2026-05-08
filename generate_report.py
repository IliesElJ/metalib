"""
Generate a Markdown trading report with embedded plots for the past two weeks.
"""

import sys
import os
sys.path.insert(0, r"C:\Users\Hermes\PycharmProjects\metalib\metalib\metadash")

from datetime import datetime, timedelta
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np

from utils.mt5_utils import initialize_mt5, get_historical_data, process_deals_data, get_account_info
from utils.metrics import (
    strategy_metrics,
    calculate_daily_performance,
    calculate_strategy_type_metrics,
)

# ── Output dirs ───────────────────────────────────────────────────────────────
REPORT_DIR = r"C:\Users\Hermes\PycharmProjects\metalib"
IMG_DIR    = os.path.join(REPORT_DIR, "report_imgs")
os.makedirs(IMG_DIR, exist_ok=True)

STYLE = {
    "bg":     "#0f1117",
    "panel":  "#1a1d27",
    "green":  "#00c853",
    "red":    "#ff3d3d",
    "blue":   "#4a9eff",
    "orange": "#ff9800",
    "purple": "#ab47bc",
    "text":   "#e0e0e0",
    "muted":  "#666677",
    "grid":   "#2a2d3a",
}

plt.rcParams.update({
    "figure.facecolor":  STYLE["bg"],
    "axes.facecolor":    STYLE["panel"],
    "axes.edgecolor":    STYLE["grid"],
    "axes.labelcolor":   STYLE["text"],
    "xtick.color":       STYLE["muted"],
    "ytick.color":       STYLE["muted"],
    "text.color":        STYLE["text"],
    "grid.color":        STYLE["grid"],
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         10,
})

def savefig(name):
    path = os.path.join(IMG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    return path

# ── Connect & fetch ───────────────────────────────────────────────────────────
ok, msg = initialize_mt5()
if not ok:
    raise SystemExit(msg)

to_date   = datetime.utcnow()
from_date = to_date - timedelta(days=14)

acct    = get_account_info()
_, deals, err = get_historical_data(from_date, to_date)
if err:
    raise SystemExit(err)

merged = process_deals_data(deals)
if merged is None or merged.empty:
    raise SystemExit("No closed bot trades found.")

merged["total_profit"] = merged["profit_open"] + merged["profit_close"]
overall  = strategy_metrics(merged, account_size=acct.get("balance", 100_000))
st_stats = calculate_strategy_type_metrics(merged, account_size=acct.get("balance", 100_000))
daily    = calculate_daily_performance(merged)
daily.columns = ["date", "daily_pnl"]
daily["cumulative"] = daily["daily_pnl"].cumsum()
daily["date"] = pd.to_datetime(daily["date"])

# ── 1. Equity curve (cumulative P&L) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
color = STYLE["green"] if daily["cumulative"].iloc[-1] >= 0 else STYLE["red"]
ax.fill_between(daily["date"], daily["cumulative"], alpha=0.18, color=color)
ax.plot(daily["date"], daily["cumulative"], color=color, linewidth=2)
ax.axhline(0, color=STYLE["muted"], linewidth=0.8, linestyle="--")
ax.set_title("Cumulative P&L  (past 2 weeks)", fontsize=13, pad=12)
ax.set_ylabel("USD")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.grid(True)
fig.tight_layout()
savefig("equity_curve.png")

# ── 2. Daily P&L bar chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
colors = [STYLE["green"] if v >= 0 else STYLE["red"] for v in daily["daily_pnl"]]
bars = ax.bar(daily["date"], daily["daily_pnl"], color=colors, width=0.6, zorder=3)
ax.axhline(0, color=STYLE["muted"], linewidth=0.8)
ax.set_title("Daily P&L", fontsize=13, pad=12)
ax.set_ylabel("USD")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
for bar, val in zip(bars, daily["daily_pnl"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.3 if val >= 0 else -1.2),
            f"{val:+.1f}", ha="center", va="bottom", fontsize=8, color=STYLE["text"])
ax.grid(True, axis="y")
fig.tight_layout()
savefig("daily_pnl.png")

# ── 3. Strategy type breakdown (pie + bar) ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

st_trades = st_stats["Number of Trades"].astype(float)
palette = [STYLE["blue"], STYLE["orange"], STYLE["purple"], STYLE["green"], STYLE["red"]][:len(st_trades)]

wedges, texts, autotexts = axes[0].pie(
    st_trades, labels=st_trades.index, autopct="%1.0f%%",
    colors=palette, startangle=90,
    wedgeprops={"edgecolor": STYLE["bg"], "linewidth": 2},
    textprops={"color": STYLE["text"]},
)
for at in autotexts:
    at.set_color(STYLE["bg"])
    at.set_fontweight("bold")
axes[0].set_title("Trade Volume by Strategy", fontsize=12, pad=12)
axes[0].set_facecolor(STYLE["bg"])

pnl_vals  = st_stats["Total Profit"].astype(float)
bar_colors = [STYLE["green"] if v >= 0 else STYLE["red"] for v in pnl_vals]
bars = axes[1].bar(pnl_vals.index, pnl_vals, color=bar_colors, zorder=3)
axes[1].axhline(0, color=STYLE["muted"], linewidth=0.8)
axes[1].set_title("Total P&L by Strategy", fontsize=12, pad=12)
axes[1].set_ylabel("USD")
for bar, val in zip(bars, pnl_vals):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.3 if val >= 0 else -1.5),
                 f"{val:+.1f}", ha="center", va="bottom", fontsize=9, color=STYLE["text"])
axes[1].grid(True, axis="y")
fig.tight_layout()
savefig("strategy_breakdown.png")

# ── 4. Symbol P&L ─────────────────────────────────────────────────────────────
sym = (
    merged.groupby("symbol_open")["total_profit"]
    .sum()
    .sort_values()
)
fig, ax = plt.subplots(figsize=(9, max(3, len(sym) * 0.55)))
colors = [STYLE["green"] if v >= 0 else STYLE["red"] for v in sym]
bars = ax.barh(sym.index, sym.values, color=colors, zorder=3)
ax.axvline(0, color=STYLE["muted"], linewidth=0.8)
ax.set_title("P&L by Symbol", fontsize=13, pad=12)
ax.set_xlabel("USD")
for bar, val in zip(bars, sym.values):
    ax.text(val + (0.1 if val >= 0 else -0.1), bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}", va="center", ha="left" if val >= 0 else "right",
            fontsize=9, color=STYLE["text"])
ax.grid(True, axis="x")
fig.tight_layout()
savefig("symbol_pnl.png")

# ── 5. Win rate by strategy (horizontal bar) ──────────────────────────────────
wr = st_stats["Win Rate (%)"].astype(float)
fig, ax = plt.subplots(figsize=(8, max(3, len(wr) * 0.65)))
bar_colors = [STYLE["green"] if v >= 50 else STYLE["red"] for v in wr]
bars = ax.barh(wr.index, wr.values, color=bar_colors, zorder=3)
ax.axvline(50, color=STYLE["muted"], linewidth=1, linestyle="--")
ax.set_xlim(0, 110)
ax.set_title("Win Rate by Strategy (%)", fontsize=13, pad=12)
ax.set_xlabel("Win Rate (%)")
for bar, val in zip(bars, wr.values):
    ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9, color=STYLE["text"])
ax.grid(True, axis="x")
fig.tight_layout()
savefig("win_rate.png")

# ── Build Markdown ────────────────────────────────────────────────────────────
def fmt(v, decimals=2, prefix="$"):
    sign = "+" if v > 0 else ""
    return f"{sign}{prefix}{v:,.{decimals}f}" if prefix else f"{sign}{v:,.{decimals}f}"

total_pnl = overall["Total Profit"]
win_rate  = overall["Win Rate (%)"]
pf        = overall["Profit Factor"]
sharpe    = overall["Sharpe Ratio"]
rrr       = overall["RRR"]
mdd       = overall["Max Drawdown (%)"]
n_trades  = int(overall["Number of Trades"])

md = f"""# Trading Report — {from_date.strftime('%b %d')} to {to_date.strftime('%b %d, %Y')}

> Account **{acct.get('name', '')}** &nbsp;|&nbsp; Server: `{acct.get('server', '')}` &nbsp;|&nbsp; Currency: {acct.get('currency', 'USD')}

---

## Account Snapshot

| | |
|---|---|
| Balance | **${acct.get('balance', 0):,.2f}** |
| Equity  | **${acct.get('equity', 0):,.2f}** |
| Open P&L | **${acct.get('profit', 0):+,.2f}** |
| Margin Used | ${acct.get('margin', 0):,.2f} |

---

## Equity Curve

![Equity Curve](report_imgs/equity_curve.png)

---

## Daily P&L

![Daily P&L](report_imgs/daily_pnl.png)

| Date | Daily P&L | Cumulative |
|---|---:|---:|
"""
for _, row in daily.iterrows():
    sign_d = "+" if row["daily_pnl"] >= 0 else ""
    sign_c = "+" if row["cumulative"] >= 0 else ""
    md += f"| {row['date'].strftime('%b %d')} | {sign_d}${row['daily_pnl']:.2f} | {sign_c}${row['cumulative']:.2f} |\n"

md += f"""
---

## Overall Performance

| Metric | Value |
|---|---:|
| Closed Trades | {n_trades} |
| Total P&L | **{fmt(total_pnl)}** |
| Win Rate | {win_rate:.1f}% |
| Profit Factor | {pf:.2f} |
| Sharpe Ratio | {sharpe:.2f} |
| RRR (avg win / avg loss) | {rrr:.2f} |
| Max Drawdown | {mdd:.2f}% |

---

## Strategy Breakdown

![Strategy Breakdown](report_imgs/strategy_breakdown.png)

![Win Rate](report_imgs/win_rate.png)

"""

cols_show = ["Number of Trades", "Total Profit", "Win Rate (%)", "Profit Factor", "RRR", "Max Drawdown (%)"]
available = [c for c in cols_show if c in st_stats.columns]
md += "| Strategy |" + "".join(f" {c} |" for c in available) + "\n"
md += "|---|" + "".join("---:|" for _ in available) + "\n"
for strat, row in st_stats[available].iterrows():
    cells = []
    for c, v in row.items():
        if isinstance(v, float):
            if c == "Number of Trades":
                cells.append(str(int(v)))
            elif "%" in c:
                cells.append(f"{v:.1f}%")
            elif v == float("inf"):
                cells.append("inf")
            else:
                cells.append(f"{v:.2f}")
        else:
            cells.append(str(int(v)))
    md += f"| {strat} |" + "".join(f" {c} |" for c in cells) + "\n"

md += f"""
---

## Symbol P&L

![Symbol P&L](report_imgs/symbol_pnl.png)

"""

sym_df = (
    merged.groupby("symbol_open")
    .agg(trades=("total_profit", "count"),
         total_pnl=("total_profit", "sum"),
         win_rate=("total_profit", lambda x: 100*(x>0).mean()),
         avg_pnl=("total_profit", "mean"))
    .sort_values("total_pnl", ascending=False)
)
md += "| Symbol | Trades | Total P&L | Win Rate | Avg P&L |\n|---|---:|---:|---:|---:|\n"
for sym_name, row in sym_df.iterrows():
    sign = "+" if row["total_pnl"] >= 0 else ""
    md += f"| {sym_name} | {int(row['trades'])} | {sign}${row['total_pnl']:.2f} | {row['win_rate']:.1f}% | {'+' if row['avg_pnl']>=0 else ''}${row['avg_pnl']:.2f} |\n"

md += f"\n---\n*Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC*\n"

report_path = os.path.join(REPORT_DIR, "trading_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(md)

print(f"Report written to: {report_path}")
print(f"Images in:         {IMG_DIR}")
