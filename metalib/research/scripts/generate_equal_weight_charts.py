"""
Renders the equity curve, drawdown, rolling-correlation, and correlation-
heatmap charts for the equal-weight portfolio report using scienceplots.

MUST run under the base miniconda env, NOT adonys -- adonys's BLAS build
crashes matplotlib's Agg savefig()/canvas.draw() unconditionally (confirmed
by isolated smoke test), so all other charts this session were built as
reportlab-native vector drawings instead. The base env's matplotlib isn't
affected, so it's used here purely as a rendering sidecar: this script only
reads a plain pickle of already-computed numbers (no MT5, no strategy code,
no adonys-only deps) and writes PNGs, which are then embedded into the
reportlab PDF built separately in the adonys env.

Usage:
    "C:\\Users\\Hermes\\miniconda3\\python.exe" metalib/research/scripts/generate_equal_weight_charts.py
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import scienceplots  # noqa: F401

plt.style.use(["science", "no-latex"])

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PKL = os.path.join(RESEARCH_DIR, "data", "equal_weight_report_data.pkl")
OUT_DIR = os.path.join(RESEARCH_DIR, "reports", "equal_weight_full")
os.makedirs(OUT_DIR, exist_ok=True)

GREEN = "#1b7a3d"
RED = "#b3221a"
BLUE = "#2f6ca8"
PURPLE = "#7b4fa6"


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {path}")


def main():
    with open(DATA_PKL, "rb") as f:
        d = pickle.load(f)

    equity = d["equity"]
    drawdown = d["drawdown"]
    rolling_corr = d["rolling_corr"]
    corr_matrix = d["corr_matrix"]
    symbols = d["symbols"]

    # --- Equity curve ---
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(equity.index, equity.values, color=GREEN, linewidth=1.3)
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_ylabel("Equity (starting = 1.0)")
    ax.set_title("Equal-Weight Portfolio -- Cumulative Equity")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    save(fig, "equity_curve.png")

    # --- Drawdown (underwater) ---
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.fill_between(drawdown.index, drawdown.values * 100, 0, color=RED, alpha=0.35, linewidth=0)
    ax.plot(drawdown.index, drawdown.values * 100, color=RED, linewidth=0.9)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Equal-Weight Portfolio -- Underwater Plot")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    save(fig, "drawdown.png")

    # --- Rolling average pairwise correlation ---
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.plot(rolling_corr.index, rolling_corr.values, color=BLUE, linewidth=1.2, marker="o", markersize=2.2)
    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    static_avg = float(np.nanmean(corr_matrix.values[np.triu_indices(len(symbols), k=1)]))
    ax.axhline(static_avg, color=PURPLE, linewidth=0.9, linestyle=":", label=f"full-sample avg = {static_avg:.4f}")
    ax.set_ylabel("Avg. pairwise correlation")
    ax.set_title("Rolling 180-Day Average Pairwise Correlation (14 instruments)")
    ax.legend(loc="upper right", fontsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    save(fig, "rolling_correlation.png")

    # --- Static correlation heatmap ---
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(symbols)))
    ax.set_yticks(range(len(symbols)))
    ax.set_xticklabels(symbols, rotation=90, fontsize=7)
    ax.set_yticklabels(symbols, fontsize=7)
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            val = corr_matrix.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.3, color=color)
    ax.set_title("Full-Sample Pairwise Correlation Matrix")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=7)
    save(fig, "correlation_heatmap.png")


if __name__ == "__main__":
    main()
