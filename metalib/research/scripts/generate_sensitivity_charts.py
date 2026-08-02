"""
Renders the Spearman-gate (window x threshold) Sharpe heatmap and the
ATR-sensitivity Sharpe line chart for the research report's Parameter
Sensitivity Analysis section, using scienceplots. Must run under the base
miniconda env, not adonys (matplotlib crashes there -- see
generate_equal_weight_charts.py for the full explanation).

Usage:
    "C:\\Users\\Hermes\\miniconda3\\python.exe" metalib/research/scripts/generate_sensitivity_charts.py
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401

plt.style.use(["science", "no-latex"])

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(RESEARCH_DIR, "data")
SWEEP_DIR = os.path.join(DATA_DIR, "sensitivity_sweep")
OUT_DIR = os.path.join(RESEARCH_DIR, "reports", "equal_weight_full")
os.makedirs(OUT_DIR, exist_ok=True)

BLUE = "#2f6ca8"

ATR_RESULTS = {2.0: 0.8014, 3.0: 0.5414, 4.0: 0.5179, 5.0: 0.4452, 6.0: 0.1118}


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {path}")


def main():
    matrix = pd.read_pickle(os.path.join(SWEEP_DIR, "window_threshold_matrix.pkl"))
    windows = list(matrix.index)
    thresholds = list(matrix.columns)

    # --- Window x threshold Sharpe heatmap ---
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    vmax = float(np.nanmax(np.abs(matrix.values)))
    im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(thresholds)))
    ax.set_yticks(range(len(windows)))
    ax.set_xticklabels([f"{t:.1f}" for t in thresholds])
    ax.set_yticklabels([str(w) for w in windows])
    ax.set_xlabel(r"Threshold $\tau_\rho$")
    ax.set_ylabel(r"Window $w$")
    ax.set_title("Equal-Weight Portfolio Sharpe: Spearman Gate Sensitivity")
    for i in range(len(windows)):
        for j in range(len(thresholds)):
            val = matrix.values[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
    # mark the report's baseline cell (window=20, threshold=0.5)
    bi, bj = windows.index(20), thresholds.index(0.5)
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.2))
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Sharpe")
    save(fig, "sensitivity_window_threshold_heatmap.png")

    # --- ATR sensitivity line chart ---
    xs = sorted(ATR_RESULTS.keys())
    ys = [ATR_RESULTS[x] for x in xs]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(xs, ys, color=BLUE, linewidth=1.4, marker="o", markersize=5)
    baseline_x, baseline_y = 4.0, ATR_RESULTS[4.0]
    ax.scatter([baseline_x], [baseline_y], color="black", s=55, zorder=5, label="baseline (4.0)")
    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xlabel("ATR sensitivity")
    ax.set_ylabel("Equal-weight portfolio Sharpe")
    ax.set_title("Equal-Weight Portfolio Sharpe vs. ATR Sensitivity")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "sensitivity_atr_line.png")


if __name__ == "__main__":
    main()
