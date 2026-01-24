# tab_daily_calendar.py
"""
Daily PnL Calendar Tab
- Shows a calendar-style table of daily PnL for a given (strategy, symbol) instance.
- Works with your existing merged_deals schema:
    time_open, profit_open, profit_close, comment_open (strategy), symbol_open
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import calendar
import numpy as np
import pandas as pd

from dash import html, dcc, Input, Output, State, callback, register_page

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PALETTE = {
    "green": "#00712D",  # primary
    "tint": "#D5ED9F",
    "paper": "#FFFBE6",
    "accent": "#FF9100",
    "ink": "#0f172a",
    "muted": "#6b7280",
    "grid": "rgba(15,23,42,0.08)",
    "red": "#b91c1c",
    "gray": "#e5e7eb",
}


@dataclass
class MonthCtx:
    year: int
    month: int

    @property
    def first(self) -> date:
        return date(self.year, self.month, 1)

    @property
    def last(self) -> date:
        last_day = calendar.monthrange(self.year, self.month)[1]
        return date(self.year, self.month, last_day)

    def step(self, delta_months: int) -> "MonthCtx":
        y, m = self.year, self.month + delta_months
        while m < 1:
            y -= 1
            m += 12
        while m > 12:
            y += 1
            m -= 12
        return MonthCtx(y, m)


def _prep_instance_daily(df: pd.DataFrame, strategy: str, symbol: str) -> pd.DataFrame:
    """Aggregate to daily PnL for the selected (strategy, symbol)."""
    if df.empty:
        return pd.DataFrame(columns=["date", "pnl", "n_trades"])

    work = df.copy()
    work["total_profit"] = work["profit_open"].fillna(0) + work["profit_close"].fillna(
        0
    )
    work["date"] = pd.to_datetime(work["time_open"]).dt.date

    mask = (work["comment_open"] == strategy) & (work["symbol_open"] == symbol)
    work = work.loc[mask]

    if work.empty:
        return pd.DataFrame(columns=["date", "pnl", "n_trades"])

    grouped = work.groupby("date").agg(
        pnl=("total_profit", "sum"),
        n_trades=("total_profit", "size"),
    )
    grouped = grouped.reset_index().sort_values("date")
    return grouped


def _month_series(ctx: MonthCtx) -> list[date]:
    """All dates in the calendar grid (6 weeks x 7 days)."""
    cal = calendar.Calendar(firstweekday=calendar.SUNDAY)
    weeks = cal.monthdatescalendar(ctx.year, ctx.month)  # 4–6 weeks
    # Ensure 6 rows for a consistent grid height
    while len(weeks) < 6:
        last_week = weeks[-1]
        next_week = [last_week[-1] + timedelta(days=i) for i in range(1, 8)]
        weeks.append(next_week)
    flat = [d for wk in weeks for d in wk]
    return flat


def _color_for_value(v: float) -> dict:
    """Return bg and text color for the chip depending on PnL."""
    if v > 0:
        return {
            "bg": "rgba(0, 113, 45, 0.10)",
            "border": PALETTE["green"],
            "text": PALETTE["green"],
        }
    if v < 0:
        return {
            "bg": "rgba(185, 28, 28, 0.10)",
            "border": PALETTE["red"],
            "text": PALETTE["red"],
        }
    return {
        "bg": "rgba(15,23,42,0.04)",
        "border": "rgba(15,23,42,0.20)",
        "text": PALETTE["muted"],
    }


def _format_money(x: float) -> str:
    try:
        return f"${x:,.0f}" if abs(x) >= 1000 else f"${x:,.2f}"
    except Exception:
        return "$0.00"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def render_daily_calendar_tab(merged_deals, default_strategy, default_symbol):
    strategies = sorted(merged_deals["comment_open"].dropna().unique().tolist())
    symbols = sorted(merged_deals["symbol_open"].dropna().unique().tolist())
    from datetime import date

    today = date.today()
    this_month = f"{today.year:04d}-{today.month:02d}-01"

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Daily PnL Calendar", className="calendar-title"),
                            html.Div(
                                [
                                    html.Button(
                                        "◀", id="cal_prev_month", className="flat-btn"
                                    ),
                                    html.Div(
                                        id="cal_month_label",
                                        style={
                                            "minWidth": "120px",
                                            "textAlign": "center",
                                            "fontWeight": "800",
                                        },
                                    ),
                                    html.Button(
                                        "▶", id="cal_next_month", className="flat-btn"
                                    ),
                                ],
                                className="calendar-controls",
                            ),
                        ],
                        className="calendar-controls",
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="cal_strategy",
                                options=[{"label": s, "value": s} for s in strategies],
                                value=(
                                    default_strategy
                                    if default_strategy in strategies
                                    else (strategies[0] if strategies else None)
                                ),
                                clearable=False,
                                className="flat-select",
                            ),
                            dcc.DatePickerSingle(
                                id="cal_anchor_date",
                                date=this_month,
                                display_format="MMMM YYYY",
                                className="flat-date",
                            ),
                            html.Div(
                                [
                                    html.Span("Legend: "),
                                    html.Span(
                                        className="legend-dot",
                                        style={"background": "rgba(0,113,45,.35)"},
                                    ),
                                    html.Span("Profit", style={"marginRight": "12px"}),
                                    html.Span(
                                        className="legend-dot",
                                        style={"background": "rgba(185,28,28,.35)"},
                                    ),
                                    html.Span("Loss"),
                                ],
                                style={"fontSize": "12px", "color": "#6b7280"},
                            ),
                        ],
                        className="calendar-controls",
                    ),
                ],
                className="calendar-toolbar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(w, className="calendar-weekday")
                            for w in ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
                        ],
                        className="calendar-grid",
                        style={"gridTemplateRows": "auto"},
                    ),
                    html.Div(id="cal_grid", className="calendar-grid"),
                ],
                className="mb-4",
            ),
            html.Div(
                [
                    html.H4(
                        "Monthly stats",
                        style={"margin": "0 0 10px 0", "color": "#0f172a"},
                    ),
                    html.Div(id="cal_stats", className="stats-panel"),
                ]
            ),
        ],
        className="content-container",
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register_daily_calendar_callbacks(app, merged_deals: pd.DataFrame):
    """Register all callbacks for the calendar tab."""

    @app.callback(
        Output("cal_anchor_date", "date"),
        Output("cal_month_label", "children"),
        Input("cal_prev_month", "n_clicks"),
        Input("cal_next_month", "n_clicks"),
        State("cal_anchor_date", "date"),
        prevent_initial_call=True,
    )
    def change_month(prev_clicks, next_clicks, anchor_date):
        ctx = dash_ctx_triggered_prop()
        dt = (
            pd.to_datetime(anchor_date).date()
            if anchor_date
            else date.today().replace(day=1)
        )
        mctx = MonthCtx(dt.year, dt.month)

        if ctx == "cal_prev_month":
            mctx = mctx.step(-1)
        elif ctx == "cal_next_month":
            mctx = mctx.step(+1)

        new_date = f"{mctx.year:04d}-{mctx.month:02d}-01"
        label = f"{calendar.month_name[mctx.month]} {mctx.year}"
        return new_date, label

    @app.callback(
        Output("cal_grid", "children"),
        Output("cal_month_label", "children"),
        Output("cal_stats", "children"),
        Input("cal_anchor_date", "date"),
        Input("cal_strategy", "value"),
        Input("cal_symbol", "value"),
    )
    def render_grid(anchor_date, strategy, symbol):
        # Month context
        dt = (
            pd.to_datetime(anchor_date).date()
            if anchor_date
            else date.today().replace(day=1)
        )
        mctx = MonthCtx(dt.year, dt.month)
        label = f"{calendar.month_name[mctx.month]} {mctx.year}"

        # Daily df for instance
        daily = _prep_instance_daily(merged_deals, strategy, symbol)
        daily = (
            daily.set_index("date")
            if not daily.empty
            else pd.DataFrame(columns=["pnl", "n_trades"])
        )

        # Build grid
        cells = []
        days = _month_series(mctx)
        for d in days:
            in_month = d.month == mctx.month
            pnl = (
                float(daily.loc[d, "pnl"])
                if (not daily.empty and d in daily.index)
                else 0.0
            )
            ntr = (
                int(daily.loc[d, "n_trades"])
                if (not daily.empty and d in daily.index)
                else 0
            )

            c = _color_for_value(pnl)
            chip_style = {
                "background": c["bg"],
                "borderColor": c["border"],
                "color": c["text"],
            }

            cells.append(
                html.Div(
                    [
                        html.Div(str(d.day), className="calendar-day"),
                        html.Div(
                            [
                                html.Span(
                                    _format_money(pnl),
                                    className="pnl-chip",
                                    style=chip_style,
                                ),
                                html.Span(
                                    f"{ntr} trades",
                                    className="trades-badge",
                                    title="Number of trades that day",
                                ),
                            ],
                            style={
                                "display": "flex",
                                "gap": "8px",
                                "alignItems": "center",
                            },
                        ),
                    ],
                    className="calendar-cell" + ("" if in_month else " is-out"),
                    title=f"{d:%b %d, %Y} — PnL: {_format_money(pnl)} • Trades: {ntr}",
                )
            )

        # Monthly stats
        month_mask = (
            (daily.index >= mctx.first) & (daily.index <= mctx.last)
            if not daily.empty
            else []
        )
        month_df = (
            daily.loc[month_mask]
            if not daily.empty
            else pd.DataFrame(columns=["pnl", "n_trades"])
        )
        if month_df.empty:
            stats = html.Div(
                "No trades for this month.", style={"color": PALETTE["muted"]}
            )
        else:
            total = month_df["pnl"].sum()
            win_days = (month_df["pnl"] > 0).sum()
            loss_days = (month_df["pnl"] < 0).sum()
            zero_days = (month_df["pnl"] == 0).sum()
            best = month_df["pnl"].max()
            worst = month_df["pnl"].min()
            avg = month_df["pnl"].mean()
            wr = 100 * win_days / max(1, len(month_df))

            stats = html.Div(
                [
                    _stat_row("Total PnL", _format_money(total)),
                    _stat_row("Avg / day", _format_money(avg)),
                    _stat_row("Win rate (days)", f"{wr:.1f}%"),
                    _stat_row("Best day", _format_money(best)),
                    _stat_row("Worst day", _format_money(worst)),
                    _stat_row(
                        "Days: + / 0 / -", f"{win_days} / {zero_days} / {loss_days}"
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(3,minmax(160px,1fr))",
                    "gap": "10px",
                },
            )

        return cells, label, stats


# ---------------------------------------------------------------------------
# Tiny utilities
# ---------------------------------------------------------------------------


def _stat_row(label: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "12px", "color": PALETTE["muted"]}),
            html.Div(value, style={"fontWeight": "800", "color": PALETTE["ink"]}),
        ],
        style={
            "border": "1px solid " + PALETTE["grid"],
            "padding": "10px",
            "background": "#fff",
        },
    )


def dash_ctx_triggered_prop() -> str | None:
    """Return the prop_id (component_id.property) that triggered the callback."""
    import dash  # local import to avoid hard dependency in module scope

    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    return ctx.triggered[0]["prop_id"].split(".")[0]
