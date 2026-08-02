"""
Flowchart of the full MetaFVG + Regression Gate signal/risk pipeline (the
reference setup: HTF=4h / LTF=M15, original momentum-FVG confirmation,
Regression Gate as the trend filter -- the one config that survived the HTF
lookahead-bias fix with a real edge).

Built as a native reportlab vector drawing (this env's matplotlib crashes
natively, confirmed repeatedly this session -- see e.g. generate_metafvg_
backtest_report.py's line_chart_drawing docstring), rendered to a one-page
PDF, then rasterized to JPG via pypdfium2 (the same tool already used all
session to visually verify PDF report pages).

Usage:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/generate_regression_gate_diagram.py
"""
import os

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import portrait
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
OUT_PDF = os.path.join(REPORTS_DIR, "_regression_gate_diagram_tmp.pdf")
OUT_JPG = os.path.join(REPORTS_DIR, "regression_gate_diagram.jpg")

STYLE = {
    "bg": "#0f1117", "panel": "#1a1d27", "green": "#00c853", "red": "#ff3d3d",
    "blue": "#4a9eff", "orange": "#ff9800", "purple": "#ab47bc", "cyan": "#00bcd4",
    "text": "#e0e0e0", "muted": "#8a8fa3", "grid": "#2a2d3a",
}
BG = HexColor(STYLE["bg"])
PANEL = HexColor(STYLE["panel"])
GREEN = HexColor(STYLE["green"])
RED = HexColor(STYLE["red"])
BLUE = HexColor(STYLE["blue"])
ORANGE = HexColor(STYLE["orange"])
PURPLE = HexColor(STYLE["purple"])
CYAN = HexColor(STYLE["cyan"])
TEXT = HexColor(STYLE["text"])
MUTED = HexColor(STYLE["muted"])
GRID = HexColor(STYLE["grid"])

W, H = 900, 1500
FONT = "Helvetica"
FONT_B = "Helvetica-Bold"


def wrap_text(text, font, size, max_width):
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if stringWidth(trial, font, size) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_box(c, cx, cy, w, h, text, fill, text_color=TEXT, font_size=11.5, bold=True, stroke=None):
    c.setFillColor(fill)
    c.setStrokeColor(stroke or fill)
    c.roundRect(cx - w / 2, cy - h / 2, w, h, 8, fill=1, stroke=1)
    lines = wrap_text(text, FONT_B if bold else FONT, font_size, w - 20)
    c.setFont(FONT_B if bold else FONT, font_size)
    c.setFillColor(text_color)
    line_h = font_size + 3
    total_h = line_h * len(lines)
    y0 = cy + total_h / 2 - font_size
    for i, line in enumerate(lines):
        c.drawCentredString(cx, y0 - i * line_h, line)


def draw_diamond(c, cx, cy, w, h, text, fill=None, text_color="#0f1117", font_size=11):
    fill = fill or ORANGE
    c.setFillColor(fill)
    c.setStrokeColor(fill)
    p = c.beginPath()
    p.moveTo(cx, cy + h / 2)
    p.lineTo(cx + w / 2, cy)
    p.lineTo(cx, cy - h / 2)
    p.lineTo(cx - w / 2, cy)
    p.close()
    c.drawPath(p, fill=1, stroke=1)
    lines = wrap_text(text, FONT_B, font_size, w - 50)
    c.setFont(FONT_B, font_size)
    c.setFillColor(HexColor(text_color))
    line_h = font_size + 2
    total_h = line_h * len(lines)
    y0 = cy + total_h / 2 - font_size
    for i, line in enumerate(lines):
        c.drawCentredString(cx, y0 - i * line_h, line)


def arrow_head(c, x, y, direction="down", color=MUTED, size=6):
    c.setFillColor(color)
    p = c.beginPath()
    if direction == "down":
        p.moveTo(x - size, y + size)
        p.lineTo(x + size, y + size)
        p.lineTo(x, y)
    elif direction == "right":
        p.moveTo(x - size, y - size)
        p.lineTo(x - size, y + size)
        p.lineTo(x, y)
    elif direction == "left":
        p.moveTo(x + size, y - size)
        p.lineTo(x + size, y + size)
        p.lineTo(x, y)
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def v_arrow(c, x, y_top, y_bot, color=MUTED, width=1.4):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x, y_top, x, y_bot + 8)
    arrow_head(c, x, y_bot, "down", color)


def elbow_arrow(c, x0, y0, x1, y1, color=MUTED, width=1.2, label=None, label_color=None):
    """From (x0,y0) go horizontal to x1, then vertical to y1, arrow pointing at end."""
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x0, y0, x1, y0)
    if y1 != y0:
        c.line(x1, y0, x1, y1 + 8)
        arrow_head(c, x1, y1, "down", color)
    if label:
        c.setFont(FONT_B, 8.5)
        c.setFillColor(label_color or color)
        c.drawString(min(x0, x1) + 6, y0 + 4, label)


def main():
    c = canvas.Canvas(OUT_PDF, pagesize=(W, H))
    c.setFillColor(BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    cx = W / 2 - 60
    side_x = W - 130

    # Title
    c.setFillColor(TEXT)
    c.setFont(FONT_B, 22)
    c.drawCentredString(W / 2, H - 50, "MetaFVG + Regression Gate")
    c.setFont(FONT, 12.5)
    c.setFillColor(MUTED)
    c.drawCentredString(W / 2, H - 72, "Signal generation & risk management pipeline  --  HTF=4h / LTF=M15  --  reference setup")

    y = H - 120
    box_w, box_h = 520, 46
    gap = 74

    def next_y():
        nonlocal y
        y -= gap
        return y

    # 1. Data fetch
    y1 = y
    draw_box(c, cx, y1, box_w, box_h,
              "Fetch HTF (4h) + LTF (M15) OHLC from MT5 (chunked, paged past the 99,999-bar cap)",
              PANEL, TEXT, 10.5, stroke=BLUE)

    y2 = next_y()
    v_arrow(c, cx, y1 - box_h / 2, y2 + box_h / 2)
    draw_box(c, cx, y2, box_w, box_h,
              "Resample HTF: label='right', closed='left'  (causal -- fixes the lookahead bug: a bucket's close no longer leaks in before the window has elapsed)",
              PANEL, GREEN, 10, stroke=GREEN)

    y3 = next_y()
    v_arrow(c, cx, y2 - box_h / 2, y3 + box_h / 2)
    draw_box(c, cx, y3, box_w, box_h,
              "Detect HTF Fair Value Gap zones (3-candle imbalance pattern)",
              PANEL, TEXT, 10.5, stroke=BLUE)

    y4 = next_y()
    v_arrow(c, cx, y3 - box_h / 2, y4 + box_h / 2)
    draw_box(c, cx, y4, box_w, box_h,
              "Filter zones: fill-% threshold + max crossing count",
              PANEL, TEXT, 10.5, stroke=BLUE)

    # Decision: price in zone?
    y5 = next_y() - 10
    v_arrow(c, cx, y4 - box_h / 2, y5 + 60)
    draw_diamond(c, cx, y5, 420, 100, "Current price inside a filtered HTF zone?  ->  candidate direction (long/short)")
    elbow_arrow(c, cx + 210, y5, side_x, y5, RED, label="no", label_color=RED)
    draw_box(c, side_x, y5, 200, 40, "Wait for next HTF bar", PANEL, MUTED, 9.5, bold=False, stroke=GRID)

    # Decision: regression gate
    y6 = y5 - 130
    v_arrow(c, cx, y5 - 60, y6 + 65)
    draw_diamond(c, cx, y6, 460, 110,
                 "Regression Gate: rolling OLS (window=20) on HTF close -- slope sign matches direction AND R² >= 0.5?",
                 fill=ORANGE)
    elbow_arrow(c, cx + 230, y6, side_x, y6, RED, label="no", label_color=RED)
    draw_box(c, side_x, y6, 200, 40, "Reject signal", PANEL, MUTED, 9.5, bold=False, stroke=GRID)

    # Decision: LTF momentum
    y7 = y6 - 130
    v_arrow(c, cx, y6 - 65, y7 + 65)
    draw_diamond(c, cx, y7, 460, 110,
                 "LTF momentum FVG confirms same direction? (last 3 M15 bars, detect_fvg_momentum_tres_strong)",
                 fill=ORANGE)
    elbow_arrow(c, cx + 230, y7, side_x, y7, RED, label="no", label_color=RED)
    draw_box(c, side_x, y7, 200, 40, "Reject signal", PANEL, MUTED, 9.5, bold=False, stroke=GRID)

    # Trade params
    y8 = y7 - 120
    v_arrow(c, cx, y7 - 65, y8 + box_h / 2)
    draw_box(c, cx, y8, box_w, 62,
              "Set trade parameters:  Entry = momentum-FVG gap boundary   |   SL = last confirmed LTF swing pivot (structural)   |   TP = Entry +/- ATR(14) x sensitivity x risk_reward",
              PANEL, CYAN, 9.8, stroke=CYAN)

    y9 = y8 - 90
    v_arrow(c, cx, y8 - 31, y9 + box_h / 2)
    draw_box(c, cx, y9, box_w, box_h,
              "Place pending limit order at Entry",
              PANEL, TEXT, 10.5, stroke=BLUE)

    # Decision: fill before EOD
    y10 = y9 - 120
    v_arrow(c, cx, y9 - box_h / 2, y10 + 55)
    draw_diamond(c, cx, y10, 380, 100, "Price reaches Entry before end-of-day?")
    elbow_arrow(c, cx + 190, y10, side_x, y10, RED, label="no", label_color=RED)
    draw_box(c, side_x, y10, 200, 40, "Order expires", PANEL, MUTED, 9.5, bold=False, stroke=GRID)

    # Position sizing
    y11 = y10 - 120
    v_arrow(c, cx, y10 - 55, y11 + 34)
    draw_box(c, cx, y11, box_w, 62,
              "Position open -- sized via fixed 2% notional OR risk-normalized sizing (risk_fraction / stop_distance_pct, capped)",
              PANEL, PURPLE, 10, stroke=PURPLE)

    # Final decision: SL or TP
    y12 = y11 - 130
    v_arrow(c, cx, y11 - 31, y12 + 65)
    draw_diamond(c, cx, y12, 380, 110, "Price hits SL or TP first? (checked every bar)", fill=ORANGE)

    y13 = y12 - 110
    loss_x = cx - 170
    win_x = cx + 170
    elbow_arrow(c, cx - 190, y12, loss_x, y13, RED)
    elbow_arrow(c, cx + 190, y12, win_x, y13, GREEN)
    draw_box(c, loss_x, y13, 230, 50, "Closed: Loss (SL hit) -- structural pivot stop", RED, "#0f1117", 10)
    draw_box(c, win_x, y13, 230, 50, "Closed: Win (TP hit) -- ATR-scaled target", GREEN, "#0f1117", 10)

    # Footer
    c.setFont(FONT, 8.5)
    c.setFillColor(MUTED)
    c.drawCentredString(W / 2, 30,
                         "Post lookahead-bias-fix reference setup: avg individual Sharpe +0.010, blended 7-pair FX Majors portfolio Sharpe +0.143")

    c.showPage()
    c.save()


if __name__ == "__main__":
    main()
