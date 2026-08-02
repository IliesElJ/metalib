"""
Renders the full baseline trade list (metalib/research/data/baseline_trades.pkl,
~4,837 closed trades across all 14 instruments at the report's baseline config)
as a LaTeX longtable fragment, \\input{}'d from the main research report rather
than inlined -- keeps metafvg_research_report.tex itself editable without
carrying ~4,800 generated rows in the same file.

No lot-size column: this backtest engine has no concept of real MT5 lot/
contract sizing anywhere (build_vbt_portfolio uses vectorbt's abstract
size_type='percent', 2% of equity, explicitly not tied to real position
sizing -- see the comment at metalib/metafvg_backtest.py:557). A note to that
effect is written into the appendix intro.

Usage:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/generate_trades_appendix_tex.py
"""
import os

import pandas as pd

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADES_PKL = os.path.join(RESEARCH_DIR, "data", "baseline_trades.pkl")
OUT_TEX = os.path.join(RESEARCH_DIR, "reports", "trades_appendix_table.tex")


def fmt_price(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.5g}"


def fmt_pnl(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:+.5g}"


def main():
    df = pd.read_pickle(TRADES_PKL)
    df = df.sort_values("entry_time").reset_index(drop=True)

    lines = []
    lines.append(r"\begin{longtable}{@{}l r r r r r@{}}")
    lines.append(r"\caption{Full baseline trade list (window $w=20$, $\tau_\rho=0.5$, "
                 r"ATR sensitivity $=4.0$, fixed 2\% notional sizing) --- all "
                 f"{len(df):,} closed trades across the 14-instrument universe, "
                 r"chronological by entry time.} \label{tab:trades} \\")
    lines.append(r"\toprule")
    lines.append(r"Asset & Entry & SL & TP & Exit & PnL \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{6}{l}{\small\itshape (Table~\ref{tab:trades} continued)} \\")
    lines.append(r"\toprule")
    lines.append(r"Asset & Entry & SL & TP & Exit & PnL \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{r}{\small\itshape continued on next page} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for _, row in df.iterrows():
        pnl_str = fmt_pnl(row["pnl"])
        color = r"\good" if row["pnl"] is not None and row["pnl"] >= 0 else r"\bad"
        lines.append(
            f"{row['symbol']} & {fmt_price(row['entry'])} & {fmt_price(row['sl'])} & "
            f"{fmt_price(row['tp'])} & {fmt_price(row['exit_price'])} & "
            f"{color}{{${pnl_str}$}} \\\\"
        )

    lines.append(r"\end{longtable}")

    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"written: {OUT_TEX} ({len(df)} rows)")


if __name__ == "__main__":
    main()
