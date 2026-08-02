"""
Asset universe for the MetaFVG A/B trend-filter sweep, bucketed by class for
the summary report. Shared by metafvg_ab_sweep.py and the report builder.
"""

UNIVERSE = {
    "FX Majors": ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"],
    "FX Minors": ["AUDJPY", "CADCHF", "EURCHF", "EURGBP", "EURJPY", "GBPCHF", "GBPJPY", "NZDCHF", "USDSGD"],
    "FX Exotics": ["EURNOK", "USDCNH", "USDZAR"],
    "Metals": ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD", "XCUUSD"],
    "Energy": ["WTI", "XBRUSD", "XNGUSD"],
    "Softs": ["COFFEE", "WHEAT", "CORN", "COCOA", "COTTON", "SUGAR"],
    "Indices": ["US30", "GER30", "HK50", "JP225", "US500", "UK100", "US100", "GER40"],
    "Crypto": ["BTCUSD", "ETHUSD", "LTCUSD", "ADAUSD", "SOLUSD", "DOGUSD"],
}


def symbol_to_bucket() -> dict:
    return {sym: bucket for bucket, syms in UNIVERSE.items() for sym in syms}


def all_symbols() -> list:
    return [sym for syms in UNIVERSE.values() for sym in syms]


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    import MetaTrader5 as mt5

    mt5.initialize()
    ok, bad = [], []
    for bucket, syms in UNIVERSE.items():
        for sym in syms:
            selected = mt5.symbol_select(sym, True)
            info = mt5.symbol_info(sym)
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H4, 0, 50)
            n = len(rates) if rates is not None else 0
            status = "OK" if (selected and info is not None and n > 10) else "FAIL"
            (ok if status == "OK" else bad).append((bucket, sym))
            print(f"{status:5s} {bucket:12s} {sym:10s} selected={selected} info={info is not None} bars={n}")

    print(f"\n{len(ok)} ok, {len(bad)} failed")
    if bad:
        print("failed:", bad)
