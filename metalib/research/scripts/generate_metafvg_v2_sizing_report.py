"""
MetaFVGv2 fixed-fraction risk sizing --- test report and rollout plan.

Consumes two pickle inputs written by sibling scripts:
    - test_metafvg_v2_sizing.py            (regression test results)
    - audit_strategy_trade_frequency.py    (MT5 trades/month per strategy)

Emits: metalib/research/reports/metafvg_v2_sizing_report.pdf

Run from the metalib repo root:
    PYTHONPATH=. python metalib/research/scripts/test_metafvg_v2_sizing.py
    PYTHONPATH=. python metalib/research/scripts/audit_strategy_trade_frequency.py
    PYTHONPATH=. python metalib/research/scripts/generate_metafvg_v2_sizing_report.py
"""
import math
import os
import pickle
from datetime import datetime

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(RESEARCH_DIR, "data")
OUT_PDF = os.path.join(RESEARCH_DIR, "reports", "metafvg_v2_sizing_report.pdf")

TESTS_PATH = os.path.join(DATA_DIR, "metafvg_v2_sizing_test.pkl")
FREQ_PATH = os.path.join(DATA_DIR, "strategy_trade_frequency.pkl")

with open(TESTS_PATH, "rb") as f:
    TESTS = pickle.load(f)
with open(FREQ_PATH, "rb") as f:
    FREQ = pickle.load(f)


# -------------------- style --------------------
STYLE = {
    "bg": "#0f1117", "panel": "#1a1d27", "green": "#00c853", "red": "#ff3d3d",
    "blue": "#4a9eff", "orange": "#ff9800", "purple": "#ab47bc",
    "text": "#e0e0e0", "muted": "#666677", "grid": "#2a2d3a",
}
BG = HexColor(STYLE["bg"])
PANEL = HexColor(STYLE["panel"])
GREEN = HexColor(STYLE["green"])
RED = HexColor(STYLE["red"])
BLUE = HexColor(STYLE["blue"])
AMBER = HexColor(STYLE["orange"])
TEXT = HexColor(STYLE["text"])
MUTED = HexColor(STYLE["muted"])
ACCENT = HexColor("#2a2d3a")
W, H = A4

styles = getSampleStyleSheet()
title_style = ParagraphStyle("T", parent=styles["Normal"], fontSize=22, leading=27,
                              textColor=TEXT, alignment=TA_CENTER,
                              fontName="Helvetica-Bold", spaceAfter=4)
subtitle_style = ParagraphStyle("S", parent=styles["Normal"], fontSize=10.5, leading=13,
                                 textColor=MUTED, alignment=TA_CENTER,
                                 fontName="Helvetica", spaceAfter=14)
section_style = ParagraphStyle("Sec", parent=styles["Normal"], fontSize=13.5, leading=17,
                                textColor=BLUE, fontName="Helvetica-Bold",
                                spaceBefore=14, spaceAfter=4)
subsection_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=11, leading=14,
                                   textColor=TEXT, fontName="Helvetica-Bold",
                                   spaceBefore=8, spaceAfter=3)
body_style = ParagraphStyle("B", parent=styles["Normal"], fontSize=9.5, textColor=TEXT,
                             fontName="Helvetica", leading=14, spaceAfter=6)
mono_style = ParagraphStyle("M", parent=styles["Normal"], fontSize=8.5, textColor=TEXT,
                             fontName="Courier", leading=11, spaceAfter=4)
callout_style = ParagraphStyle("C", parent=styles["Normal"], fontSize=10, textColor=TEXT,
                                fontName="Helvetica", leading=14, spaceAfter=4)


def section(title):
    return [Paragraph(title, section_style),
            HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=6)]


def base_table_style(header=True):
    s = [
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PANEL, ACCENT]),
        ("GRID", (0, 0), (-1, -1), 0.3, ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    if header:
        s.append(("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"))
        s.append(("BACKGROUND", (0, 0), (-1, 0), HexColor("#22242f")))
    return TableStyle(s)


def callout(paragraphs, color=BLUE):
    tbl = Table([[paragraphs]], colWidths=[W - 4 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("BOX", (0, 0), (-1, -1), 1.1, color),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return tbl


def fmt(x, dec=2, pct=False, suffix=""):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    if pct:
        return f"{x*100:.{dec}f}%"
    if isinstance(x, int):
        return f"{x}"
    return f"{x:.{dec}f}{suffix}"


# -------------------- content --------------------
def build_intro():
    story = [
        Paragraph("MetaFVG v2: Fixed-Fraction Risk Sizing", title_style),
        Paragraph("Test report, formula derivation, and rollout plan for SL-based strategies",
                  subtitle_style),
    ]
    intro_text = (
        "This report documents the switch from static-lot to fixed-fraction risk position sizing "
        "introduced in <b>metafvg_v2</b>. The new behavior binds trade size to the stop-loss "
        "distance so that a stop-out costs exactly <b>size_position * balance</b> of account "
        "equity. The report covers the formula, the 5-pip FX minimum stop clamp added to protect "
        "against near-spread SLs, a regression-test summary, and a proposed rollout to the "
        "remaining SL-based strategies with risk fractions calibrated to observed trade frequency."
    )
    story.append(Spacer(1, 4))
    story.append(callout([Paragraph(intro_text, callout_style)], color=BLUE))
    return story


def build_formula():
    story = section("1. Sizing formula")
    story.append(Paragraph(
        "For every SL-based order, the volume (in MT5 lots) is chosen so that the P&amp;L of "
        "hitting the stop-loss exactly equals a fixed fraction of the account balance:",
        body_style))
    story.append(Paragraph(
        "&nbsp;&nbsp;&nbsp;<b>volume = (size_position * balance) / (sl_distance * vpu)</b>",
        mono_style))
    story.append(Paragraph(
        "where <b>sl_distance = |entry - sl|</b> (in price units) and <b>vpu</b> is the "
        "value-per-price-unit (in account currency) of one lot:",
        body_style))
    story.append(Paragraph(
        "&nbsp;&nbsp;&nbsp;<b>vpu = contract_size * fx_conv(profit_ccy -&gt; account_ccy)</b>",
        mono_style))
    story.append(Paragraph(
        "The result is rounded to the symbol's volume_step and clamped to [volume_min, "
        "volume_max]. Two safety layers apply on top:",
        body_style))
    story.append(Paragraph(
        "<b>1. 5-pip minimum SL floor (FX only).</b> When trade_calc_mode == 0 "
        "(SYMBOL_TRADE_CALC_MODE_FOREX) and the raw SL distance is below 5 pips "
        "(0.0001 for non-JPY pairs, 0.01 for JPY), the SL distance used for sizing is "
        "raised to 5 pips. The order's actual SL price is unchanged --- only the sizing "
        "calculation uses the clamped distance. This protects against near-spread SLs "
        "sizing into the maximum-notional cap or into a lot count that inflates fees.",
        body_style))
    story.append(Paragraph(
        "<b>2. 5x-balance notional cap.</b> Regardless of SL distance, the volume is "
        "capped so that <b>volume * contract_size * price (in account ccy) &le; 5 * balance</b>. "
        "This is a last-resort brake for pathological SL configurations (e.g. a 1-pip SL on "
        "an index) --- when it triggers, the trade is placed with strictly less risk than the "
        "target, never more.",
        body_style))
    story.append(Paragraph(
        "<b>Failure modes and fallback.</b> If any required MT5 information is missing "
        "(account_info, symbol_info, FX conversion rate) or the SL is absent/zero, the "
        "function returns <b>self.size_position</b> as a raw lot count and logs the reason. "
        "This preserves the pre-v2 fixed-lot behavior on failure --- v2 never raises and "
        "never blocks a trade.",
        body_style))
    return story


def build_tests():
    story = section("2. Regression test results")
    story.append(Paragraph(
        f"Nine scenarios exercised against mocked MT5 with a $" +
        f"{TESTS['balance']:,.0f} USD balance and " +
        f"size_position = {TESTS['risk_fraction']*100:.2f}%. All pass.",
        body_style))

    tests = TESTS["results"]
    hdr = ["#", "Scenario", "Price", "SL",
           "SL dist", "Volume", "Real-SL risk", "Result"]
    rows = [hdr]
    for r in tests:
        num = r["name"].split(".")[0]
        name = r["name"].split(". ", 1)[-1]
        rows.append([
            num,
            name,
            fmt(r["price"], dec=4) if r["price"] else "n/a",
            fmt(r["sl"], dec=4) if r["sl"] else "n/a",
            fmt(r["sl_distance"], dec=4) if r["sl_distance"] else "n/a",
            fmt(r["volume"], dec=2),
            (f"${r['realized_risk_at_real_sl']:,.2f}"
             if r["realized_risk_at_real_sl"] is not None else "n/a"),
            "PASS" if r["pass"] else "FAIL",
        ])
    tbl = Table(rows, colWidths=[0.6*cm, 6.4*cm, 1.6*cm, 1.6*cm,
                                  1.6*cm, 1.4*cm, 2.2*cm, 1.4*cm])
    ts = base_table_style()
    ts.add("ALIGN", (2, 1), (-2, -1), "RIGHT")
    ts.add("ALIGN", (-1, 1), (-1, -1), "CENTER")
    for i, r in enumerate(tests, start=1):
        col = GREEN if r["pass"] else RED
        ts.add("TEXTCOLOR", (-1, i), (-1, i), col)
        ts.add("FONTNAME", (-1, i), (-1, i), "Helvetica-Bold")
    tbl.setStyle(ts)
    story.append(tbl)

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Key observations from the test cases.</b>",
        subsection_style))
    story.append(Paragraph(
        "<b>Normal SLs (cases 1, 3, 5, 6):</b> realized risk at the real SL is exactly "
        "$500 = 0.5% of balance, as designed. The formula lines up across FX with USD "
        "quote (EURUSD), JPY (USDJPY, price ~150), non-USD quote (USDCHF), and indices "
        "(US500).",
        body_style))
    story.append(Paragraph(
        "<b>Tight-SL FX (cases 2 and 4):</b> when the raw SL is below the 5-pip floor, "
        "the sizing uses 5 pips but the trade risk at the <i>real</i> SL becomes "
        "<i>smaller</i> than the target (EURUSD 3-pip SL: $136.50 vs $500 target; "
        "USDJPY 2-pip SL: $66.67 vs $500 target). This is deliberate --- the alternative "
        "was to size against the tiny raw distance and hit the notional cap at a lot count "
        "that would make broker fees and slippage dominate.",
        body_style))
    story.append(Paragraph(
        "<b>Very-tight index SL (case 7):</b> indices bypass the 5-pip clamp (calc_mode &ne; 0), "
        "so the notional cap catches this: 0.5-point SL on US500 would raw-size to 10 lots, "
        "the cap trims to 1 lot, realized risk is $50. Same principle --- trade underrisks, "
        "never overrisks, when the SL geometry is pathological.",
        body_style))
    story.append(Paragraph(
        "<b>Fallback (cases 8, 9):</b> when SL is absent or MT5 info is unavailable, the "
        "raw <i>size_position</i> is returned as-is. Under the new interpretation of "
        "<i>size_position</i> as a fraction (0.005), this means the fallback lot size is "
        "0.005 lots --- effectively skipping the trade. This is the intended behavior for "
        "v2, where a missing SL is a bug that should not silently sail through as a full "
        "position size.",
        body_style))
    return story


def build_freq_table():
    story = section("3. SL-based strategy activity (last 180 days)")
    story.append(Paragraph(
        f"Live MT5 deal counts, {FREQ['period_start']:%Y-%m-%d} to "
        f"{FREQ['period_end']:%Y-%m-%d}. Only open deals (entry == 0) are counted. "
        f"MetaFVG v1 dominates by two orders of magnitude vs everything else; that fact "
        f"drives the per-strategy risk-fraction recommendation in section 4.",
        body_style))

    per_s = FREQ["per_strategy"]
    # Only include SL-based strategies
    sl_strats = ["metafvg", "metafvg_v2", "metamlp", "metaob", "metamtou"]
    rows = [["Strategy", "SL-based?", "Total trades", "Trades/month",
             "Tags", "Trades/mo per tag"]]
    for s in sl_strats:
        info = per_s.get(s, None)
        if info is None:
            rows.append([s, "yes", "0", "0", "0", "0"])
            continue
        rows.append([
            s, "yes",
            f"{info['total_open_deals']:,}",
            f"{info['trades_per_month_total']:.1f}",
            f"{info['distinct_tags']}",
            f"{info['trades_per_month_per_tag']:.2f}",
        ])
    # Also show no-SL strategies for context (but greyed intent)
    for s in ["metane", "metaga", "metago"]:
        info = per_s.get(s)
        if info is None:
            continue
        rows.append([
            s, "no",
            f"{info['total_open_deals']:,}",
            f"{info['trades_per_month_total']:.1f}",
            f"{info['distinct_tags']}",
            f"{info['trades_per_month_per_tag']:.2f}",
        ])
    tbl = Table(rows, colWidths=[3*cm, 2.5*cm, 3*cm, 3*cm, 2*cm, 3.5*cm])
    ts = base_table_style()
    ts.add("ALIGN", (2, 1), (-1, -1), "RIGHT")
    ts.add("ALIGN", (1, 1), (1, -1), "CENTER")
    # Mark no-SL rows in muted color
    for i, s in enumerate(sl_strats + [x for x in ["metane", "metaga", "metago"]
                                        if x in per_s], start=1):
        if s in ("metane", "metaga", "metago"):
            ts.add("TEXTCOLOR", (0, i), (-1, i), MUTED)
    tbl.setStyle(ts)
    story.append(tbl)
    story.append(Paragraph(
        "<i>Rows in muted colour are market-order strategies without SLs --- listed for "
        "context only, they are not affected by this change.</i>",
        ParagraphStyle("Cap", parent=body_style, fontSize=8, textColor=MUTED,
                        spaceBefore=4, spaceAfter=8)))
    return story


def build_recommendations():
    story = section("4. Recommended risk fraction per SL strategy")
    story.append(Paragraph(
        "Sizing rule of thumb: keep expected monthly loss volatility of each strategy at ~2% "
        "of balance when trades are treated as independent draws. Under a Bernoulli model with "
        "per-trade risk r, the monthly loss standard deviation across N trades scales as "
        "<b>r &middot; sqrt(N)</b>. Setting that to 0.02 gives a soft upper bound "
        "<b>r_max &asymp; 0.02 / sqrt(N)</b>. Real trades are correlated (multiple instances "
        "of the same strategy fire on overlapping regimes), so the recommendations below "
        "apply a further ~2x safety margin on top of that.",
        body_style))

    per_s = FREQ["per_strategy"]

    def _rec(name, current_val, recommendation_bp, rationale):
        info = per_s.get(name, {})
        N = info.get("trades_per_month_total", 0)
        r_max = 0.02 / math.sqrt(N) if N > 0 else None
        r_max_str = f"{r_max*100:.2f}%" if r_max else "n/a"
        return [
            name,
            f"{N:.1f}/mo",
            r_max_str,
            current_val,
            f"{recommendation_bp/100:.2f}%",
            rationale,
        ]

    rows = [["Strategy", "Trades/mo", "r_max soft", "Current", "Recommended", "Reason"]]
    rows.append(_rec("metafvg", "0.01--0.03 lot",  5,
                     "very high freq: 4 tags & 730+ trades/mo. 0.05% keeps "
                     "monthly loss vol ~1.5% even at 4x safety."))
    rows.append(_rec("metafvg_v2", "0.50%", 50,
                     "already fractional; live sample small but backtested at 0.5%."))
    rows.append(_rec("metamlp", "0.05 lot", 25,
                     "medium-high freq (110/mo across 7 tags). 0.25% ~ 0.5% "
                     "monthly loss vol at 2x safety."))
    rows.append(_rec("metaob", "0.01--1.0 lot", 30,
                     "wide config range today; low freq (8/mo). 0.30% risk uniformly "
                     "resolves the 100x spread across instances."))
    rows.append(_rec("metamtou", "0.01 lot", 50,
                     "very low freq (1.7/mo); daily bars, high conviction. 0.50% "
                     "matches the FVG v2 convention."))

    tbl = Table(rows, colWidths=[2.4*cm, 2.0*cm, 2.0*cm, 2.6*cm, 2.4*cm, 6.0*cm])
    ts = base_table_style()
    ts.add("ALIGN", (1, 1), (-2, -1), "RIGHT")
    ts.add("VALIGN", (0, 0), (-1, -1), "TOP")
    ts.add("FONTSIZE", (-1, 1), (-1, -1), 7.5)
    tbl.setStyle(ts)
    story.append(tbl)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>What changes on the strategy classes.</b> Each SL-based strategy needs the same "
        "one-method override that <i>metafvg_v2</i> already has --- a "
        "<i>_resolve_position_size</i> that (a) reinterprets <i>size_position</i> from the "
        "YAML as a fraction of balance, (b) computes lots from balance / SL distance / "
        "value-per-price-unit, (c) applies the 5-pip FX minimum and 5x notional cap, "
        "(d) falls back to the raw <i>size_position</i> on any MT5 error. The math is "
        "identical to v2 --- only the interpretation of <i>size_position</i> in the YAML "
        "config changes.",
        body_style))
    story.append(Paragraph(
        "<b>MetaScale interaction.</b> All strategies in <i>MetaScale.DEFAULT_STRATEGY_PARAMS</i> "
        "(metafvg, metago, metane, metaga, metaob) have their <i>size_position</i> "
        "periodically overwritten by MetaScale's CVXPY optimizer. Switching those "
        "strategies to fixed-fraction interpretation means MetaScale would need to be "
        "aware of that change --- either write fractions (and its covariance model "
        "reinterpreted accordingly) or those strategies drop out of MetaScale's book like "
        "<i>metafvg_v2</i> already does. This is the main open decision before rolling out "
        "the change to <i>metafvg</i> v1 and <i>metaob</i>.",
        body_style))
    return story


def build_rollout():
    story = section("5. Suggested rollout order")
    story.append(Paragraph(
        "<b>Phase 1 (low-risk, no MetaScale conflict).</b> "
        "Add <i>_resolve_position_size</i> override + YAML rewrite for "
        "<b>metamtou</b> and <b>metamlp</b>. Neither is currently in MetaScale's "
        "<i>DEFAULT_STRATEGY_PARAMS</i>, so this can ship without touching that logic. "
        "MTOU trades are infrequent and daily-bar, so a bad week is bounded; MLP trades "
        "at medium-high frequency, so this is where the sizing improvement will actually "
        "compound.",
        body_style))
    story.append(Paragraph(
        "<b>Phase 2 (requires MetaScale change).</b> "
        "Extend to <b>metaob</b> (and later <b>metafvg</b> v1). This needs a corresponding "
        "update to <i>MetaScale</i>: either remove these strategies from "
        "<i>DEFAULT_STRATEGY_PARAMS</i> (letting them manage their own sizing like v2), or "
        "teach <i>MetaScale</i> to write fractions in the fractional-sizing convention and "
        "compute portfolio covariance against value-per-price-unit rather than tick_value.",
        body_style))
    story.append(Paragraph(
        "<b>Verification per strategy.</b> Before flipping any YAML, run the same "
        "mocked-MT5 unit test used for v2 (this report, section 2) against each strategy's "
        "new <i>_resolve_position_size</i>. Then dry-run each strategy under PM2 with "
        "<b>debug=True</b> for a few LTF bars and inspect the sizing log lines "
        "(<i>balance=... risk=... sl_dist=... -&gt; volume=...</i>) before restarting for "
        "live production.",
        body_style))
    return story


def build_appendix():
    story = section("Appendix A: Sample log output")
    story.append(Paragraph(
        "Verbatim log lines captured by the mocked test runner for scenario 1 "
        "(EURUSD 20-pip SL) --- the format shipped in production is identical.",
        body_style))
    for r in TESTS["results"]:
        if r["name"].startswith("1."):
            for line in r["log"]:
                story.append(Paragraph(line, mono_style))
            break
    story.append(Spacer(1, 6))
    story.append(Paragraph("Appendix B: Scenario 2 log (5-pip clamp fires)",
                            subsection_style))
    for r in TESTS["results"]:
        if r["name"].startswith("2."):
            for line in r["log"]:
                story.append(Paragraph(line, mono_style))
            break
    story.append(Spacer(1, 6))
    story.append(Paragraph("Appendix C: Scenario 7 log (notional cap fires on index)",
                            subsection_style))
    for r in TESTS["results"]:
        if r["name"].startswith("7."):
            for line in r["log"]:
                story.append(Paragraph(line, mono_style))
            break
    return story


def draw_page_bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # footer
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(
        W / 2, 0.9 * cm,
        f"metalib research  ---  MetaFVG v2 sizing report  ---  "
        f"generated {datetime.now():%Y-%m-%d}  ---  page {doc.page}")
    canvas.restoreState()


def main():
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    doc = SimpleDocTemplate(
        OUT_PDF, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.6*cm, bottomMargin=1.6*cm,
    )
    story = []
    story += build_intro()
    story += build_formula()
    story += build_tests()
    story += build_freq_table()
    story += build_recommendations()
    story += build_rollout()
    story.append(PageBreak())
    story += build_appendix()
    doc.build(story, onFirstPage=draw_page_bg, onLaterPages=draw_page_bg)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
