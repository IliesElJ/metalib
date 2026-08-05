"""
Unit test / regression harness for the shared fixed-fraction sizing helper
on MetaStrategy, exercised via the MetaFVGv2, MetaMTOU and MetaMLP
subclasses that delegate to it.

Mocks mt5.account_info / mt5.symbol_info so we can drive the sizing logic
across FX (normal, tight SL, JPY, non-USD quote), indices, and fallback
paths without needing a live MT5 connection. Emits a pickle at
metalib/research/data/metafvg_v2_sizing_test.pkl consumed by
generate_metafvg_v2_sizing_report.py.

Run from the metalib repo root:
    PYTHONPATH=. python metalib/research/scripts/test_metafvg_v2_sizing.py
"""
import os
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import patch

import MetaTrader5 as mt5

from metalib.metafvg_v2 import MetaFVGv2
from metalib.metamtou import MetaMTOU
from metalib.metamlp import MetaMLP
from metalib.metastrategy import MetaStrategy


BALANCE = 100_000.0
RISK = 0.005  # 0.5% at SL (FVGv2 default; MTOU/MLP tested separately below)
MAX_NOTIONAL_MULT = 5.0


def _acct(balance=BALANCE, currency="USD"):
    return SimpleNamespace(balance=balance, currency=currency)


def _sym(
    contract_size=100_000,
    profit="USD",
    calc_mode=0,
    vmin=0.01,
    vmax=500.0,
    vstep=0.01,
    digits=5,
):
    return SimpleNamespace(
        trade_contract_size=contract_size,
        currency_profit=profit,
        trade_calc_mode=calc_mode,
        volume_min=vmin,
        volume_max=vmax,
        volume_step=vstep,
        digits=digits,
    )


CONV_RATES = {
    ("JPY", "USD"): 1.0 / 150.0,
    ("CHF", "USD"): 1.0 / 0.90,
    ("EUR", "USD"): 1.10,
}


def _fake_conv(from_ccy, to_ccy):
    if from_ccy == to_ccy:
        return 1.0
    return CONV_RATES.get((from_ccy, to_ccy))


def _make_fvg_v2(symbol, risk=RISK):
    return MetaFVGv2(
        symbols=[symbol], timeframe=mt5.TIMEFRAME_M15,
        size_position=risk, tag=f"test_{symbol.lower()}",
        limit_number_position=1,
    )


def _make_mtou(symbol, risk=0.01):
    return MetaMTOU(
        symbols=[symbol], timeframe=mt5.TIMEFRAME_D1,
        tag=f"test_mtou_{symbol.lower()}",
        size_position=risk,
    )


def _make_mlp(symbol, risk=0.0025):
    return MetaMLP(
        symbols=[symbol], timeframe=mt5.TIMEFRAME_H1,
        tag=f"test_mlp_{symbol.lower()}",
        size_position=risk,
    )


def _run_case(name, symbol, price, sl, acct, sym,
              expected_volume=None, expected_risk_usd=None,
              tolerance=0.02, factory=_make_fvg_v2, risk_override=None):
    """Run one sizing scenario and capture inputs / outputs."""
    log_lines = []
    risk = risk_override if risk_override is not None else RISK
    with patch.object(mt5, "account_info", return_value=acct), \
         patch.object(mt5, "symbol_info", return_value=sym), \
         patch.object(MetaStrategy, "_get_conversion_rate", staticmethod(_fake_conv)):
        strat = factory(symbol) if risk_override is None else factory(symbol, risk)
        strat._log = log_lines.append
        volume = strat._resolve_position_size(symbol, price=price, sl=sl)

    # Recompute the metric we care about (actual risk at REAL sl distance,
    # in account currency) to check whether the cap or clamp changed it.
    if sl and sl > 0 and price and volume:
        real_sl_dist = abs(price - sl)
        vpu = sym.trade_contract_size
        if sym.currency_profit != acct.currency:
            vpu *= _fake_conv(sym.currency_profit, acct.currency)
        realized_risk = volume * real_sl_dist * vpu
    else:
        realized_risk = None

    result = {
        "name": name,
        "symbol": symbol,
        "price": price,
        "sl": sl,
        "sl_distance": abs(price - sl) if sl else None,
        "balance": acct.balance,
        "currency": acct.currency,
        "contract_size": sym.trade_contract_size,
        "profit_ccy": sym.currency_profit,
        "calc_mode": sym.trade_calc_mode,
        "risk_fraction": risk,
        "risk_amount_target": risk * acct.balance,
        "volume": volume,
        "realized_risk_at_real_sl": realized_risk,
        "expected_volume": expected_volume,
        "expected_risk": expected_risk_usd,
        "log": log_lines[:],
        "pass": True,
        "note": "",
    }

    if expected_volume is not None:
        rel = abs(volume - expected_volume) / max(1e-9, expected_volume)
        if rel > tolerance:
            result["pass"] = False
            result["note"] = (f"volume {volume} vs expected {expected_volume} "
                              f"(rel diff {rel:.3%})")
    return result


def run_all():
    results = []

    # -------------------------------------------------------------------
    # FX pair, normal SL: EURUSD 20-pip SL @ 1.10
    # Expected: 500 / (0.002 * 100000) = 2.5 lots
    # -------------------------------------------------------------------
    results.append(_run_case(
        "1. EURUSD - normal 20-pip SL",
        "EURUSD", price=1.10, sl=1.098,
        acct=_acct(), sym=_sym(profit="USD"),
        expected_volume=2.50, expected_risk_usd=500.0,
    ))

    # -------------------------------------------------------------------
    # FX pair, tight SL below 5-pip floor: EURUSD 3-pip SL
    # sl_distance clamped 0.0003 -> 0.0005
    # raw = 500 / (0.0005 * 100000) = 10 lots
    # notional = 10 * 100000 * 1.10 = 1.1M > 500k cap
    # capped_vol = 500000 / 110000 = 4.545 -> 4.55 (0.01 step)
    # Realized risk at real 3-pip SL: 4.55 * 0.0003 * 100000 = 136.50 USD
    # (well below the 500 target -- cap + clamp both bit; user's tight SL
    #  gets a "risk less than intended" outcome, not "risk more")
    # -------------------------------------------------------------------
    results.append(_run_case(
        "2. EURUSD - tight 3-pip SL (below floor)",
        "EURUSD", price=1.10, sl=1.0997,
        acct=_acct(), sym=_sym(profit="USD"),
        expected_volume=4.55, expected_risk_usd=None,  # cap-driven, not risk-target
    ))

    # -------------------------------------------------------------------
    # JPY pair, normal SL: USDJPY 30-JPY-pip SL @ 150
    # vpu = 100000 * (1/150) = 666.67
    # raw = 500 / (0.30 * 666.67) = 2.5 lots
    # -------------------------------------------------------------------
    results.append(_run_case(
        "3. USDJPY - normal 30-pip SL",
        "USDJPY", price=150.0, sl=150.30,
        acct=_acct(), sym=_sym(profit="JPY"),
        expected_volume=2.50, expected_risk_usd=500.0,
    ))

    # -------------------------------------------------------------------
    # JPY pair, tight SL: USDJPY 2-JPY-pip SL @ 150
    # 5-pip minimum for JPY = 0.05 (not 0.0005)
    # clamped 0.02 -> 0.05
    # raw = 500 / (0.05 * 666.67) = 15 lots
    # notional/lot = 100000 USD, notional = 1.5M > 500k cap
    # cap_vol = 500000 / 100000 = 5.0 lots
    # Realized risk at real 2-pip SL: 5 * 0.02 * 666.67 = 66.67 USD
    # -------------------------------------------------------------------
    results.append(_run_case(
        "4. USDJPY - tight 2-pip SL (below JPY floor)",
        "USDJPY", price=150.0, sl=150.02,
        acct=_acct(), sym=_sym(profit="JPY"),
        expected_volume=5.00, expected_risk_usd=None,
    ))

    # -------------------------------------------------------------------
    # FX pair, non-USD quote: USDCHF 20-pip SL @ 0.90
    # vpu = 100000 / 0.90 = 111111.11
    # raw = 500 / (0.002 * 111111.11) = 2.25 lots
    # -------------------------------------------------------------------
    results.append(_run_case(
        "5. USDCHF - normal 20-pip SL (non-USD quote)",
        "USDCHF", price=0.90, sl=0.898,
        acct=_acct(), sym=_sym(profit="CHF"),
        expected_volume=2.25, expected_risk_usd=500.0,
    ))

    # -------------------------------------------------------------------
    # Index, tight SL (indices don't use pip floor)
    # US500 @ 5000, SL 4995 (5 pt)
    # vpu = 100 USD/pt, contract 100
    # raw = 500 / (5 * 100) = 1.0 lot
    # notional = 1 * 100 * 5000 = 500k = cap (exactly at edge)
    # -------------------------------------------------------------------
    results.append(_run_case(
        "6. US500 - 5-point SL",
        "US500", price=5000.0, sl=4995.0,
        acct=_acct(), sym=_sym(contract_size=100, profit="USD",
                               calc_mode=32, vstep=0.1),
        expected_volume=1.00, expected_risk_usd=500.0,
    ))

    # -------------------------------------------------------------------
    # Index, very tight SL (0.5 pt) - cap bites, no pip clamp
    # raw = 500 / (0.5 * 100) = 10 lots
    # notional = 10 * 100 * 5000 = 5M > 500k cap -> vol_cap = 1 lot
    # Realized risk = 1 * 0.5 * 100 = 50 USD
    # -------------------------------------------------------------------
    results.append(_run_case(
        "7. US500 - 0.5-point SL (cap only, no pip floor)",
        "US500", price=5000.0, sl=4999.5,
        acct=_acct(), sym=_sym(contract_size=100, profit="USD",
                               calc_mode=32, vstep=0.1),
        expected_volume=1.00, expected_risk_usd=None,
    ))

    # -------------------------------------------------------------------
    # MTOU delegates to the shared helper. 1% risk, EURUSD, 40-pip SL
    # raw = 1000 / (0.004 * 100000) = 2.5 lots  (2x FVGv2 case 1 as risk 2x)
    # -------------------------------------------------------------------
    results.append(_run_case(
        "8. MTOU - EURUSD 40-pip SL @ 1% risk",
        "EURUSD", price=1.10, sl=1.096,
        acct=_acct(), sym=_sym(profit="USD"),
        expected_volume=2.50, expected_risk_usd=1000.0,
        factory=_make_mtou, risk_override=0.01,
    ))

    # -------------------------------------------------------------------
    # MLP delegates to the shared helper. 0.25% risk, GBPCHF, 30-pip SL @ 1.10
    # vpu = 100000 * (1/0.9) = 111111.11
    # raw = 250 / (0.003 * 111111.11) = 250/333.33 = 0.75 lots
    # -------------------------------------------------------------------
    results.append(_run_case(
        "9. MLP - GBPCHF 30-pip SL @ 0.25% risk",
        "GBPCHF", price=1.10, sl=1.097,
        acct=_acct(), sym=_sym(profit="CHF"),
        expected_volume=0.75, expected_risk_usd=250.0,
        factory=_make_mlp, risk_override=0.0025,
    ))

    # -------------------------------------------------------------------
    # Fallback: no SL provided -> returns self.size_position raw
    # -------------------------------------------------------------------
    results.append(_run_case(
        "10. Fallback - no SL provided (FVGv2)",
        "EURUSD", price=1.10, sl=None,
        acct=_acct(), sym=_sym(),
        expected_volume=RISK,  # falls through to raw size_position
    ))

    # -------------------------------------------------------------------
    # Fallback: MT5 account_info unavailable
    # -------------------------------------------------------------------
    with patch.object(mt5, "account_info", return_value=None), \
         patch.object(mt5, "symbol_info", return_value=_sym()):
        strat = _make_fvg_v2("EURUSD")
        strat._log = lambda *a, **k: None
        v = strat._resolve_position_size("EURUSD", price=1.10, sl=1.098)
    results.append({
        "name": "11. Fallback - MT5 account_info unavailable",
        "symbol": "EURUSD", "price": 1.10, "sl": 1.098,
        "sl_distance": 0.002, "balance": None, "currency": None,
        "contract_size": None, "profit_ccy": None, "calc_mode": None,
        "risk_fraction": RISK, "risk_amount_target": None,
        "volume": v, "realized_risk_at_real_sl": None,
        "expected_volume": RISK, "expected_risk": None, "log": [],
        "pass": abs(v - RISK) < 1e-9,
        "note": "" if abs(v - RISK) < 1e-9 else f"got {v}, expected {RISK}",
    })

    return results


def main():
    results = run_all()
    print(f"\n{'='*80}")
    print(f"MetaFVGv2._resolve_position_size regression tests")
    print(f"{'='*80}\n")

    fail = 0
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[{status}] {r['name']}")
        print(f"        price={r['price']} sl={r['sl']} "
              f"sl_dist={r['sl_distance']} -> volume={r['volume']}")
        if r["expected_volume"] is not None:
            print(f"        expected volume={r['expected_volume']}  "
                  f"realized_risk={r['realized_risk_at_real_sl']}")
        if not r["pass"]:
            print(f"        NOTE: {r['note']}")
            fail += 1
        print()

    print(f"{fail} failure(s) out of {len(results)}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_pkl = os.path.join(out_dir, "metafvg_v2_sizing_test.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "results": results,
            "balance": BALANCE,
            "risk_fraction": RISK,
            "max_notional_mult": MAX_NOTIONAL_MULT,
        }, f)
    print(f"\nWrote results to {out_pkl}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
