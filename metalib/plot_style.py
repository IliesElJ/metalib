"""Matplotlib style helpers for a consistent Metalib look."""

MINIMAL_MODERN_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "semibold",
    "axes.titlepad": 10,
    "figure.figsize": (10, 4.8),
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D9DEE8",
    "axes.labelcolor": "#1F2937",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.color": "#4B5563",
    "ytick.color": "#4B5563",
    "text.color": "#111827",
    "grid.color": "#E9EDF5",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "legend.frameon": False,
}


def use_metalib_style() -> None:
    """Apply Metalib's minimal modern Matplotlib style."""
    import matplotlib as mpl

    style = dict(MINIMAL_MODERN_STYLE)
    style["axes.prop_cycle"] = mpl.cycler(
        color=[
            "#2563EB",
            "#059669",
            "#DC2626",
            "#7C3AED",
            "#EA580C",
            "#0891B2",
            "#DB2777",
        ]
    )
    mpl.rcParams.update(style)
