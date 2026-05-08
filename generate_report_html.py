"""
Generate a self-contained HTML trading report with interactive Plotly charts.
White-cream palette.  Drop-in replacement for generate_report.py.
"""

import sys
import os
sys.path.insert(0, r"C:\Users\Hermes\PycharmProjects\metalib\metalib\metadash")

from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from utils.mt5_utils import initialize_mt5, get_historical_data, process_deals_data, get_account_info
from utils.metrics import (
    strategy_metrics,
    calculate_daily_performance,
    calculate_strategy_type_metrics,
)

# ── Output ────────────────────────────────────────────────────────────────────
REPORT_DIR  = r"C:\Users\Hermes\PycharmProjects\metalib"
REPORT_PATH = os.path.join(REPORT_DIR, "trading_report.html")

# ── Cream palette ─────────────────────────────────────────────────────────────
C = {
    "bg":       "#FAF8F2",      # warm parchment
    "panel":    "#F2EDE3",      # slightly deeper cream
    "border":   "#DDD5C3",
    "text":     "#2C2416",      # dark brown
    "muted":    "#7A6E5F",
    "green":    "#2E7D32",
    "green_lt": "#C8E6C9",
    "red":      "#C62828",
    "red_lt":   "#FFCDD2",
    "blue":     "#1565C0",
    "orange":   "#E65100",
    "purple":   "#6A1B9A",
    "teal":     "#00695C",
    "grid":     "#E5DDD0",
}

AXIS_STYLE = dict(
    gridcolor=C["grid"],
    zerolinecolor=C["border"],
    tickfont=dict(color=C["muted"]),
    linecolor=C["border"],
)

BASE_LAYOUT = dict(
    paper_bgcolor=C["panel"],
    plot_bgcolor=C["panel"],
    font=dict(family="Georgia, serif", color=C["text"], size=12),
    legend=dict(bgcolor=C["bg"], bordercolor=C["border"], borderwidth=1,
                font=dict(color=C["text"])),
    margin=dict(l=50, r=30, t=50, b=50),
)

def _style_axes(fig):
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig

def _div(fig, full_js=False):
    """Return a Plotly chart as an HTML <div> string."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn" if full_js else False,
        config={"displayModeBar": False},
    )

def _color(v):
    return C["green"] if v >= 0 else C["red"]

def _fmt(v, decimals=2, sign=True):
    s = "+" if (sign and v > 0) else ""
    return f"{s}${v:,.{decimals}f}"

# ── Connect & fetch ───────────────────────────────────────────────────────────
ok, msg = initialize_mt5()
if not ok:
    raise SystemExit(msg)

to_date   = datetime.utcnow()
from_date = to_date - timedelta(days=30)

acct        = get_account_info()
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

# ── KPI values ────────────────────────────────────────────────────────────────
total_pnl = overall["Total Profit"]
win_rate  = overall["Win Rate (%)"]
pf        = overall["Profit Factor"]
sharpe    = overall["Sharpe Ratio"]
rrr       = overall["RRR"]
mdd       = overall["Max Drawdown (%)"]
n_trades  = int(overall["Number of Trades"])

# ── 1. Equity curve ───────────────────────────────────────────────────────────
eq_positive = daily["cumulative"].iloc[-1] >= 0
eq_color    = C["green"] if eq_positive else C["red"]
eq_fill     = "rgba(46,125,50,0.15)" if eq_positive else "rgba(198,40,40,0.15)"
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=daily["date"], y=daily["cumulative"],
    mode="lines",
    line=dict(color=eq_color, width=2.5),
    fill="tozeroy",
    fillcolor=eq_fill,
    name="Cumulative P&L",
    hovertemplate="%{x|%b %d}<br><b>%{y:+$,.2f}</b><extra></extra>",
))
fig_eq.add_hline(y=0, line_color=C["muted"], line_dash="dot", line_width=1)
fig_eq.update_layout(
    **BASE_LAYOUT,
    title="Cumulative P&L",
    yaxis_title="USD",
    showlegend=False,
    height=320,
)
_style_axes(fig_eq)

# ── 2. Daily P&L bar ──────────────────────────────────────────────────────────
bar_colors = [_color(v) for v in daily["daily_pnl"]]
fig_daily = go.Figure()
fig_daily.add_trace(go.Bar(
    x=daily["date"], y=daily["daily_pnl"],
    marker_color=bar_colors,
    text=[f"{v:+.1f}" for v in daily["daily_pnl"]],
    textposition="outside",
    textfont=dict(size=10, color=C["text"]),
    hovertemplate="%{x|%b %d}<br><b>%{y:+$,.2f}</b><extra></extra>",
    name="Daily P&L",
))
fig_daily.add_hline(y=0, line_color=C["muted"], line_dash="dot", line_width=1)
fig_daily.update_layout(
    **BASE_LAYOUT,
    title="Daily P&L",
    yaxis_title="USD",
    showlegend=False,
    height=320,
)
_style_axes(fig_daily)

# ── 3. Strategy trade-volume pie ──────────────────────────────────────────────
pie_palette = [C["blue"], C["orange"], C["purple"], C["teal"], C["green"]]
st_trades = st_stats["Number of Trades"].astype(float)
fig_pie = go.Figure()
fig_pie.add_trace(go.Pie(
    labels=st_trades.index.tolist(),
    values=st_trades.values.tolist(),
    marker=dict(colors=pie_palette[:len(st_trades)],
                line=dict(color=C["bg"], width=2)),
    textinfo="label+percent",
    textfont=dict(size=12),
    hole=0.35,
    hovertemplate="<b>%{label}</b><br>%{value:.0f} trades (%{percent})<extra></extra>",
))
fig_pie.update_layout(
    **BASE_LAYOUT,
    title="Trade Volume by Strategy",
    showlegend=False,
    height=340,
)

# ── 4. Strategy P&L bar ───────────────────────────────────────────────────────
pnl_vals = st_stats["Total Profit"].astype(float)
fig_spnl = go.Figure()
fig_spnl.add_trace(go.Bar(
    x=pnl_vals.index.tolist(),
    y=pnl_vals.values.tolist(),
    marker_color=[_color(v) for v in pnl_vals],
    text=[f"{v:+.2f}" for v in pnl_vals],
    textposition="outside",
    textfont=dict(size=11, color=C["text"]),
    hovertemplate="<b>%{x}</b><br>%{y:+$,.2f}<extra></extra>",
))
fig_spnl.add_hline(y=0, line_color=C["muted"], line_dash="dot", line_width=1)
fig_spnl.update_layout(
    **BASE_LAYOUT,
    title="Total P&L by Strategy",
    yaxis_title="USD",
    showlegend=False,
    height=340,
)
_style_axes(fig_spnl)

# ── 5. Win rate horizontal bar ────────────────────────────────────────────────
wr = st_stats["Win Rate (%)"].astype(float)
fig_wr = go.Figure()
fig_wr.add_trace(go.Bar(
    y=wr.index.tolist(),
    x=wr.values.tolist(),
    orientation="h",
    marker_color=[C["green"] if v >= 50 else C["red"] for v in wr],
    text=[f"{v:.1f}%" for v in wr],
    textposition="outside",
    textfont=dict(size=11, color=C["text"]),
    hovertemplate="<b>%{y}</b><br>Win rate: %{x:.1f}%<extra></extra>",
))
fig_wr.add_vline(x=50, line_color=C["muted"], line_dash="dash", line_width=1.2)
fig_wr.update_layout(
    **BASE_LAYOUT,
    title="Win Rate by Strategy (%)",
    showlegend=False,
    height=max(260, len(wr) * 70),
)
fig_wr.update_xaxes(**AXIS_STYLE, title="Win Rate (%)", range=[0, 120])
fig_wr.update_yaxes(**AXIS_STYLE)

# ── 6. Symbol P&L horizontal bar ─────────────────────────────────────────────
sym = (
    merged.groupby("symbol_open")["total_profit"]
    .sum()
    .sort_values()
)
fig_sym = go.Figure()
fig_sym.add_trace(go.Bar(
    y=sym.index.tolist(),
    x=sym.values.tolist(),
    orientation="h",
    marker_color=[_color(v) for v in sym],
    text=[f"{v:+.2f}" for v in sym],
    textposition="outside",
    textfont=dict(size=10, color=C["text"]),
    hovertemplate="<b>%{y}</b><br>%{x:+$,.2f}<extra></extra>",
))
fig_sym.add_vline(x=0, line_color=C["muted"], line_dash="dot", line_width=1)
fig_sym.update_layout(
    **BASE_LAYOUT,
    title="P&L by Symbol",
    xaxis_title="USD",
    showlegend=False,
    height=max(280, len(sym) * 55),
)
_style_axes(fig_sym)

# ── Render chart divs ─────────────────────────────────────────────────────────
div_eq    = _div(fig_eq,    full_js=True)   # first chart loads Plotly CDN
div_daily = _div(fig_daily)
div_pie   = _div(fig_pie)
div_spnl  = _div(fig_spnl)
div_wr    = _div(fig_wr)
div_sym   = _div(fig_sym)

# ── Daily table rows ──────────────────────────────────────────────────────────
daily_rows = ""
for _, row in daily.iterrows():
    cls_d = "pos" if row["daily_pnl"] >= 0 else "neg"
    cls_c = "pos" if row["cumulative"] >= 0 else "neg"
    daily_rows += (
        f'<tr><td>{row["date"].strftime("%b %d")}</td>'
        f'<td class="{cls_d}">{_fmt(row["daily_pnl"])}</td>'
        f'<td class="{cls_c}">{_fmt(row["cumulative"])}</td></tr>\n'
    )

# ── Strategy table rows ───────────────────────────────────────────────────────
cols_show = ["Number of Trades", "Total Profit", "Win Rate (%)", "Profit Factor", "RRR", "Max Drawdown (%)"]
available = [c for c in cols_show if c in st_stats.columns]

st_header = '<tr><th>Strategy</th>' + "".join(f"<th>{c}</th>" for c in available) + "</tr>"
st_rows = ""
for strat, row in st_stats[available].iterrows():
    cells = []
    for c, v in row.items():
        if c == "Number of Trades":
            cells.append(f'<td>{int(v)}</td>')
        elif c == "Total Profit":
            cls = "pos" if float(v) >= 0 else "neg"
            cells.append(f'<td class="{cls}">{_fmt(float(v))}</td>')
        elif "%" in c:
            cells.append(f'<td>{float(v):.1f}%</td>')
        elif float(v) == float("inf"):
            cells.append("<td>∞</td>")
        else:
            cells.append(f'<td>{float(v):.2f}</td>')
    st_rows += f'<tr><td><strong>{strat}</strong></td>{"".join(cells)}</tr>\n'

# ── Symbol table rows ─────────────────────────────────────────────────────────
sym_df = (
    merged.groupby("symbol_open")
    .agg(
        trades=("total_profit", "count"),
        total_pnl=("total_profit", "sum"),
        win_rate=("total_profit", lambda x: 100 * (x > 0).mean()),
        avg_pnl=("total_profit", "mean"),
    )
    .sort_values("total_pnl", ascending=False)
)
sym_rows = ""
for sym_name, row in sym_df.iterrows():
    cls = "pos" if row["total_pnl"] >= 0 else "neg"
    sym_rows += (
        f'<tr><td><strong>{sym_name}</strong></td>'
        f'<td>{int(row["trades"])}</td>'
        f'<td class="{cls}">{_fmt(row["total_pnl"])}</td>'
        f'<td>{row["win_rate"]:.1f}%</td>'
        f'<td class="{"pos" if row["avg_pnl"]>=0 else "neg"}">{_fmt(row["avg_pnl"])}</td></tr>\n'
    )

# ── KPI card helper ───────────────────────────────────────────────────────────
def kpi(label, value, cls=""):
    return f'<div class="kpi-card {cls}"><div class="kpi-val">{value}</div><div class="kpi-label">{label}</div></div>'

pnl_cls   = "kpi-pos" if total_pnl >= 0 else "kpi-neg"
sharp_cls = "kpi-pos" if sharpe >= 0 else "kpi-neg"
wr_cls    = "kpi-pos" if win_rate >= 50 else "kpi-neg"

# ── HTML assembly ─────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Report — {from_date.strftime('%b %d')} to {to_date.strftime('%b %d, %Y')}</title>
<style>
  :root {{
    --bg:      {C["bg"]};
    --panel:   {C["panel"]};
    --border:  {C["border"]};
    --text:    {C["text"]};
    --muted:   {C["muted"]};
    --green:   {C["green"]};
    --red:     {C["red"]};
    --green-lt:{C["green_lt"]};
    --red-lt:  {C["red_lt"]};
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 14px;
    line-height: 1.6;
  }}

  /* ── Header ── */
  .report-header {{
    background: var(--panel);
    border-bottom: 2px solid var(--border);
    padding: 36px 48px 28px;
  }}
  .report-header h1 {{
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
  }}
  .report-header .meta {{
    font-size: 13px;
    color: var(--muted);
    font-family: 'Courier New', monospace;
  }}
  .report-header .meta span {{ margin-right: 24px; }}

  /* ── Layout ── */
  .container {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 24px 64px;
  }}

  section {{ margin-bottom: 40px; }}

  h2 {{
    font-size: 17px;
    font-weight: 700;
    color: var(--text);
    border-left: 4px solid var(--border);
    padding-left: 12px;
    margin-bottom: 20px;
    letter-spacing: 0.3px;
  }}

  /* ── KPI cards ── */
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px;
    margin-bottom: 36px;
  }}
  .kpi-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px 16px;
    text-align: center;
  }}
  .kpi-val {{
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 4px;
  }}
  .kpi-label {{
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-family: 'Courier New', monospace;
  }}
  .kpi-pos .kpi-val {{ color: var(--green); }}
  .kpi-neg .kpi-val {{ color: var(--red); }}

  /* ── Account snapshot ── */
  .acct-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
  }}
  .acct-item {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
  }}
  .acct-item .label {{
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.7px;
    font-family: 'Courier New', monospace;
    margin-bottom: 4px;
  }}
  .acct-item .value {{ font-size: 18px; font-weight: 700; }}

  /* ── Chart card ── */
  .chart-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 4px;
    margin-bottom: 16px;
  }}

  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }}

  /* ── Tables ── */
  .tbl-wrap {{ overflow-x: auto; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  th {{
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 9px 12px;
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--muted);
    font-family: 'Courier New', monospace;
  }}
  td {{
    border: 1px solid var(--border);
    padding: 9px 12px;
  }}
  tr:nth-child(even) td {{ background: var(--panel); }}
  tr:hover td {{ background: var(--border); }}
  .pos {{ color: var(--green); font-weight: 600; }}
  .neg {{ color: var(--red);   font-weight: 600; }}

  /* ── Divider ── */
  hr {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 36px 0;
  }}

  /* ── Footer ── */
  .footer {{
    text-align: center;
    font-size: 12px;
    color: var(--muted);
    padding: 24px;
    font-family: 'Courier New', monospace;
  }}

  @media (max-width: 680px) {{
    .two-col {{ grid-template-columns: 1fr; }}
    .report-header {{ padding: 24px 20px 18px; }}
  }}
</style>
</head>
<body>

<!-- ── Header ─────────────────────────────────────────────────────────────── -->
<div class="report-header">
  <h1>Trading Report</h1>
  <div class="meta">
    <span>{from_date.strftime('%b %d')} &ndash; {to_date.strftime('%b %d, %Y')}</span>
    <span>Account: <strong>{acct.get('name', '')}</strong></span>
    <span>Server: {acct.get('server', '')}</span>
    <span>Currency: {acct.get('currency', 'USD')}</span>
  </div>
</div>

<div class="container">

<!-- ── KPI strip ──────────────────────────────────────────────────────────── -->
<div class="kpi-row">
  {kpi("Balance",       f"${acct.get('balance',0):,.0f}")}
  {kpi("Equity",        f"${acct.get('equity',0):,.0f}")}
  {kpi("Open P&amp;L",  f"{'+' if acct.get('profit',0)>=0 else ''}${acct.get('profit',0):,.2f}",
       "kpi-pos" if acct.get("profit",0)>=0 else "kpi-neg")}
  {kpi("Total P&amp;L", _fmt(total_pnl), pnl_cls)}
  {kpi("Win Rate",      f"{win_rate:.1f}%", wr_cls)}
  {kpi("Trades",        str(n_trades))}
  {kpi("Profit Factor", f"{pf:.2f}" if pf != float('inf') else "∞",
       "kpi-pos" if pf >= 1 else "kpi-neg")}
  {kpi("Sharpe Ratio",  f"{sharpe:.2f}", sharp_cls)}
  {kpi("RRR",           f"{rrr:.2f}")}
  {kpi("Max Drawdown",  f"{mdd:.2f}%")}
</div>

<hr>

<!-- ── Equity curve ────────────────────────────────────────────────────────── -->
<section>
  <h2>Equity Curve</h2>
  <div class="chart-card">{div_eq}</div>
</section>

<!-- ── Daily P&L ──────────────────────────────────────────────────────────── -->
<section>
  <h2>Daily P&amp;L</h2>
  <div class="chart-card">{div_daily}</div>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>Date</th><th>Daily P&amp;L</th><th>Cumulative</th></tr></thead>
      <tbody>{daily_rows}</tbody>
    </table>
  </div>
</section>

<hr>

<!-- ── Strategy breakdown ─────────────────────────────────────────────────── -->
<section>
  <h2>Strategy Breakdown</h2>
  <div class="two-col">
    <div class="chart-card">{div_pie}</div>
    <div class="chart-card">{div_spnl}</div>
  </div>
  <div class="chart-card">{div_wr}</div>
  <div class="tbl-wrap" style="margin-top:16px;">
    <table>
      <thead>{st_header}</thead>
      <tbody>{st_rows}</tbody>
    </table>
  </div>
</section>

<hr>

<!-- ── Symbol P&L ──────────────────────────────────────────────────────────── -->
<section>
  <h2>Symbol P&amp;L</h2>
  <div class="chart-card">{div_sym}</div>
  <div class="tbl-wrap" style="margin-top:16px;">
    <table>
      <thead><tr><th>Symbol</th><th>Trades</th><th>Total P&amp;L</th><th>Win Rate</th><th>Avg P&amp;L</th></tr></thead>
      <tbody>{sym_rows}</tbody>
    </table>
  </div>
</section>

</div><!-- /container -->

<div class="footer">Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC &nbsp;&mdash;&nbsp; metalib trading framework</div>

</body>
</html>
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"HTML report written to: {REPORT_PATH}")
