"""
Custom reportlab candlestick chart with FVG-zone shading and a rolling
Spearman-rho subplot, for illustrating example trades in the mean-reversion
diversified reports. reportlab has no built-in candlestick chart type, so
this draws bars directly with Rect/Line primitives (same house pattern as
every other chart this session -- native vector drawing, no matplotlib).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from reportlab.lib.colors import HexColor
from reportlab.graphics.shapes import Drawing, Line, Rect, String

GREEN = HexColor("#00c853")
RED = HexColor("#ff3d3d")
BLUE = HexColor("#4a9eff")
TEXT = HexColor("#e0e0e0")
MUTED = HexColor("#8a8fa3")
ACCENT = HexColor("#2a2d3a")


def _price_axis_labels(pmin: float, pmax: float, n: int = 4) -> List[float]:
    return list(np.linspace(pmin, pmax, n))


def trade_example_drawing(
    ltf_window: pd.DataFrame,
    trade,
    fvg_zone: Optional[Tuple[float, float]],
    rho_window: pd.Series,
    rho_threshold: float,
    width: float = 480,
    height: float = 300,
    title: str = "",
) -> Drawing:
    """
    ltf_window: OHLC DataFrame (DatetimeIndex) spanning some bars before entry
        through some bars after exit.
    trade: the backtest Trade object (entry, sl, tp, exit_price, entry_time,
        exit_time, direction, status).
    fvg_zone: (gap_low, gap_high) of the HTF zone active at entry, or None.
    rho_window: rolling Spearman rho series (HTF cadence) covering the same
        time span, for the subplot underneath.
    """
    d = Drawing(width, height)
    d.add(String(2, height - 12, title, fillColor=TEXT, fontSize=10, fontName="Helvetica-Bold"))

    top_y = height - 22
    candle_h = height * 0.56
    rho_h = height * 0.20
    x0, x1 = 44, width - 8
    plot_w = x1 - x0

    n = len(ltf_window)
    if n < 2:
        d.add(String(width / 2 - 40, height / 2, "insufficient data", fillColor=MUTED, fontSize=9))
        return d
    bar_w = plot_w / n

    all_prices = list(ltf_window["high"].values) + list(ltf_window["low"].values)
    if trade.sl is not None:
        all_prices.append(trade.sl)
    if trade.tp is not None:
        all_prices.append(trade.tp)
    if fvg_zone:
        all_prices.extend(fvg_zone)
    pmin, pmax = min(all_prices), max(all_prices)
    prange = pmax - pmin
    ppad = prange * 0.08 if prange > 1e-12 else abs(pmax) * 0.01
    pmin -= ppad
    pmax += ppad

    candle_y0 = top_y - candle_h

    def price_to_y(p):
        return candle_y0 + (p - pmin) / (pmax - pmin) * candle_h

    def idx_to_x(i):
        return x0 + i * bar_w + bar_w / 2

    # FVG zone shading (drawn first, underneath candles). Label pinned to the
    # fixed top-left corner of the panel rather than inside the zone itself --
    # an in-zone label collides with the entry marker whenever entry (often
    # right at the zone boundary, by construction) lands nearby.
    if fvg_zone:
        glow, ghigh = min(fvg_zone), max(fvg_zone)
        y_lo, y_hi = price_to_y(glow), price_to_y(ghigh)
        zone_fill = HexColor("#4a9eff")
        d.add(Rect(x0, y_lo, plot_w, max(y_hi - y_lo, 1), fillColor=zone_fill, fillOpacity=0.15, strokeColor=None))
        d.add(String(x0 + 3, top_y - 8, "HTF FVG zone", fillColor=BLUE, fontSize=6.5, fontName="Helvetica-Oblique"))

    # axis frame
    d.add(Line(x0, candle_y0, x1, candle_y0, strokeColor=ACCENT, strokeWidth=0.6))
    d.add(Line(x0, candle_y0, x0, top_y, strokeColor=ACCENT, strokeWidth=0.6))
    for p in _price_axis_labels(pmin + ppad, pmax - ppad, 4):
        y = price_to_y(p)
        d.add(String(x0 - 4, y - 3, f"{p:.5g}", fillColor=MUTED, fontSize=6, textAnchor="end"))
        d.add(Line(x0 - 2, y, x0, y, strokeColor=ACCENT, strokeWidth=0.5))

    # candlesticks
    times = list(ltf_window.index)
    for i, (ts, row) in enumerate(ltf_window.iterrows()):
        x = idx_to_x(i)
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = GREEN if c >= o else RED
        d.add(Line(x, price_to_y(l), x, price_to_y(h), strokeColor=color, strokeWidth=0.8))
        body_top, body_bot = price_to_y(max(o, c)), price_to_y(min(o, c))
        d.add(Rect(x - bar_w * 0.35, body_bot, max(bar_w * 0.7, 0.5), max(body_top - body_bot, 0.6),
                    fillColor=color, strokeColor=color))

    # SL / TP levels. Labels pinned to the left edge, not the right: exit
    # markers land at the *same* price as SL/TP by definition whenever that's
    # the exit reason, and exit tends to sit toward the right of the window
    # (window_end is anchored a few bars past exit_time) -- a right-edge
    # label collides with it far more often than a left-edge one collides
    # with entry (a coincidence, not structural).
    if trade.sl is not None:
        y = price_to_y(trade.sl)
        d.add(Line(x0, y, x1, y, strokeColor=RED, strokeWidth=0.8, strokeDashArray=[3, 2]))
        d.add(String(x0 + 2, y + 2, "SL", fillColor=RED, fontSize=6.5, textAnchor="start"))
    if trade.tp is not None:
        y = price_to_y(trade.tp)
        d.add(Line(x0, y, x1, y, strokeColor=GREEN, strokeWidth=0.8, strokeDashArray=[3, 2]))
        d.add(String(x0 + 2, y + 2, "TP", fillColor=GREEN, fontSize=6.5, textAnchor="start"))

    # entry / exit markers -- nearest bar index
    def nearest_idx(ts):
        if ts is None:
            return None
        pos = np.searchsorted(np.array(times, dtype="datetime64[ns]"), np.datetime64(ts))
        return int(np.clip(pos, 0, n - 1))

    entry_idx = nearest_idx(trade.entry_time)
    if entry_idx is not None and trade.entry is not None:
        x, y = idx_to_x(entry_idx), price_to_y(trade.entry)
        d.add(Line(x - 5, y - 5, x + 5, y + 5, strokeColor=BLUE, strokeWidth=1.4))
        d.add(Line(x - 5, y + 5, x + 5, y - 5, strokeColor=BLUE, strokeWidth=1.4))
        d.add(String(x, y - 12, "entry", fillColor=BLUE, fontSize=6.5, textAnchor="middle"))

    exit_idx = nearest_idx(trade.exit_time)
    if exit_idx is not None and trade.exit_price is not None:
        won = trade.exit_reason == "tp"
        color = GREEN if won else RED
        x, y = idx_to_x(exit_idx), price_to_y(trade.exit_price)
        d.add(Line(x - 5, y - 5, x + 5, y + 5, strokeColor=color, strokeWidth=1.4))
        d.add(Line(x - 5, y + 5, x + 5, y - 5, strokeColor=color, strokeWidth=1.4))
        d.add(String(x, y + 6, "exit", fillColor=color, fontSize=6.5, textAnchor="middle"))

    # ---- Spearman rho subplot ----
    rho_y0 = candle_y0 - 24 - rho_h
    rho_y1 = rho_y0 + rho_h
    d.add(String(x0, rho_y1 + 3, "Rolling Spearman ρ (HTF)", fillColor=TEXT, fontSize=7.5, fontName="Helvetica-Bold"))
    d.add(Line(x0, rho_y0, x1, rho_y0, strokeColor=ACCENT, strokeWidth=0.6))
    d.add(Line(x0, rho_y0, x0, rho_y1, strokeColor=ACCENT, strokeWidth=0.6))

    rmin, rmax = -1.0, 1.0

    def rho_to_y(r):
        return rho_y0 + (r - rmin) / (rmax - rmin) * rho_h

    # zero line + threshold band
    d.add(Line(x0, rho_to_y(0), x1, rho_to_y(0), strokeColor=MUTED, strokeWidth=0.5))
    d.add(Line(x0, rho_to_y(rho_threshold), x1, rho_to_y(rho_threshold), strokeColor=HexColor("#ff9800"), strokeWidth=0.6, strokeDashArray=[2, 2]))
    d.add(Line(x0, rho_to_y(-rho_threshold), x1, rho_to_y(-rho_threshold), strokeColor=HexColor("#ff9800"), strokeWidth=0.6, strokeDashArray=[2, 2]))
    d.add(String(x1, rho_to_y(rho_threshold) + 1, f"±{rho_threshold:.2f}", fillColor=HexColor("#ff9800"), fontSize=6, textAnchor="end"))

    rho_clean = rho_window.dropna()
    if len(rho_clean) >= 2:
        rho_times = np.array(rho_clean.index, dtype="datetime64[ns]")
        ltf_times = np.array(times, dtype="datetime64[ns]")
        pts = []
        for ts, val in rho_clean.items():
            pos = np.searchsorted(ltf_times, np.datetime64(ts))
            pos = int(np.clip(pos, 0, n - 1))
            pts.append((idx_to_x(pos), rho_to_y(max(min(val, rmax), rmin))))
        for k in range(len(pts) - 1):
            d.add(Line(pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1], strokeColor=HexColor("#ab47bc"), strokeWidth=1.1))

    d.add(String(x0 - 4, rho_y0 - 2, "-1", fillColor=MUTED, fontSize=6, textAnchor="end"))
    d.add(String(x0 - 4, rho_y1 - 4, "+1", fillColor=MUTED, fontSize=6, textAnchor="end"))

    return d
