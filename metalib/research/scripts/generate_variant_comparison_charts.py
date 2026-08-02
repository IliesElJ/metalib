"""
Overlays equity curves and drawdown curves for the baseline config and the
three standout combos found in the parameter sensitivity sweep (best gate
alone, best ATR alone, and their combination) on the same plots, for the
"Exploring Outperforming Parameter Sets" section. All four variants were
backtested on the identical OHLC snapshot (backtest_param_variant.py), so
differences are purely parameter effects, not data-snapshot noise.

Must run under the base miniconda env, not adonys (matplotlib crashes there).

Usage:
    "C:\\Users\\Hermes\\miniconda3\\python.exe" metalib/research/scripts/generate_variant_comparison_charts.py
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scienceplots  # noqa: F401

plt.style.use(["science", "no-latex"])

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANT_DIR = os.path.join(RESEARCH_DIR, "data", "param_variants")
OUT_DIR = os.path.join(RESEARCH_DIR, "reports", "equal_weight_full")
os.makedirs(OUT_DIR, exist_ok=True)

VARIANTS = [
    ("baseline", "Baseline (w=20, tau=0.5, atr=4.0)", "#5a6472"),
    ("best_gate", "Best Gate (w=25, tau=0.4, atr=4.0)", "#2f6ca8"),
    ("best_atr", "Best ATR (w=20, tau=0.5, atr=2.0)", "#1b7a3d"),
    ("combined_mix", "Combined Mix (w=25, tau=0.4, atr=2.0)", "#b3221a"),
]


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {path}")


def main():
    data = {}
    for slug, label, color in VARIANTS:
        with open(os.path.join(VARIANT_DIR, f"{slug}.pkl"), "rb") as f:
            data[slug] = pickle.load(f)

    # --- Overlaid equity curves ---
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for slug, label, color in VARIANTS:
        eq = data[slug]["equity"]
        lw = 1.7 if slug == "combined_mix" else 1.1
        alpha = 1.0 if slug in ("combined_mix", "baseline") else 0.85
        ax.plot(eq.index, eq.values, color=color, linewidth=lw, alpha=alpha, label=label)
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_ylabel("Equity (starting = 1.0)")
    ax.set_title("Equal-Weight Portfolio Equity: Baseline vs. Outperforming Variants")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    save(fig, "variant_equity_comparison.png")

    # --- Overlaid drawdown curves ---
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for slug, label, color in VARIANTS:
        dd = data[slug]["drawdown"]
        lw = 1.7 if slug == "combined_mix" else 1.1
        alpha = 1.0 if slug in ("combined_mix", "baseline") else 0.85
        ax.plot(dd.index, dd.values * 100, color=color, linewidth=lw, alpha=alpha, label=label)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Equal-Weight Portfolio Drawdown: Baseline vs. Outperforming Variants")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    save(fig, "variant_drawdown_comparison.png")


if __name__ == "__main__":
    main()
