from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


DEFAULT_FRED_API_KEY = "d32f52d45fc621a2848f31540442b8cd"
DEFAULT_START = "2005-01-01"
DEFAULT_EURUSD_SYMBOL = "EURUSD"

FRED_SERIES: dict[str, str] = {
    "us_10y_nominal": "DGS10",
    "us_2y_nominal": "DGS2",
    "us_10y_real": "DFII10",
    "vix": "VIXCLS",
    "fed_funds": "DFF",
    "us_curve_10y2y": "T10Y2Y",
}

ECB_SERIES: dict[str, str] = {
    "eu_10y_nominal": "YC.B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y",
    "eu_2y_nominal": "YC.B.U2.EUR.4F.G_N_C.SV_C_YM.SR_2Y",
    "ecb_deposit_rate": "FM.D.U2.EUR.4F.KR.DFR.LEV",
    "ecb_main_refi_rate": "FM.D.U2.EUR.4F.KR.MRR_FR.LEV",
    "eu_ciss": "CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX",
    "eu_hicp_yoy": "ICP.M.U2.N.000000.4.ANR",
}

RESEARCH_FEATURES: tuple[str, ...] = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "vol_63d",
    "sma_10",
    "ema_10",
    "dist_sma_10",
    "sma_20",
    "ema_20",
    "dist_sma_20",
    "sma_50",
    "ema_50",
    "dist_sma_50",
    "sma_100",
    "ema_100",
    "dist_sma_100",
    "sma_200",
    "ema_200",
    "dist_sma_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_z_20",
    "atr_14",
    "atr_pct_14",
    "breakout_20",
    "drawdown_20",
    "range_pct",
    "us_10y_nominal",
    "us_2y_nominal",
    "us_10y_real",
    "vix",
    "fed_funds",
    "eu_10y_nominal",
    "eu_2y_nominal",
    "eu_10y_real",
    "ecb_deposit_rate",
    "ecb_main_refi_rate",
    "eu_curve_10y2y",
    "eu_ciss",
    "eu_ciss_low_regime",
    "eu_ciss_mid_regime",
    "eu_ciss_high_regime",
    "eu_hicp_yoy",
    "cot_euro_net_noncommercial",
    "cot_euro_net_4w_change",
    "cot_euro_net_z_156w",
    "us_curve_10y2y",
    "vix_low_regime",
    "vix_mid_regime",
    "vix_high_regime",
    "cot_euro_net_4w_ma",
    "kitchin_cycle_sin",
    "kitchin_cycle_cos",
    "juglar_cycle_sin",
    "juglar_cycle_cos",
)


def build_eurusd_mtofx_daily_features(
    *,
    symbol: str = DEFAULT_EURUSD_SYMBOL,
    start: str | pd.Timestamp = DEFAULT_START,
    end: str | pd.Timestamp | None = None,
    fred_api_key: str | None = None,
    output_path: str | Path | None = None,
    include_fred: bool = True,
    include_ecb: bool = True,
    include_cot: bool = True,
    cot_report_type: str = "legacy_fut",
    cot_release_lag_days: int = 3,
    mt5_shutdown: bool = True,
) -> pd.DataFrame:
    """Pull EURUSD daily data and build the full notebook feature table.

    The returned frame is indexed by daily dates and includes OHLC columns,
    all features listed in ``eurusd_mtofx_daily_framework.ipynb``, and
    ``target_ret_1d_fwd``.
    """
    end_ts = _normalize_end(end)
    start_ts = _normalize_date(start)

    prices = pull_mt5_daily_ohlc(
        symbol=symbol,
        start=start_ts,
        end=end_ts,
        shutdown=mt5_shutdown,
    )
    data = add_price_action_features(prices)

    if include_fred:
        fred = pull_fred_features(
            start=start_ts,
            end=end_ts,
            api_key=fred_api_key,
        )
        data = data.join(_align_to_index(fred, data.index), how="left")

    data = add_vix_regimes(data)

    if include_ecb:
        ecb = pull_ecb_features(
            start=start_ts,
            end=end_ts,
        )
        data = data.join(_align_to_index(ecb, data.index), how="left")

    data = add_eu_macro_derived_features(data)

    if include_cot:
        cot = pull_cot_euro_positioning_features(
            start=start_ts,
            end=end_ts,
            report_type=cot_report_type,
            release_lag_days=cot_release_lag_days,
        )
        data = data.join(_align_to_index(cot, data.index), how="left")

    data = add_cycle_features(data)
    data["target_ret_1d_fwd"] = data["ret_1d"].shift(-1)

    ordered_cols = [
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "real_volume",
        "spread",
        *[col for col in RESEARCH_FEATURES if col in data.columns],
        "target_ret_1d_fwd",
    ]
    ordered_cols = [col for col in ordered_cols if col in data.columns]
    data = data[ordered_cols]

    if output_path is not None:
        write_feature_table(data, output_path)

    return data


def pull_mt5_daily_ohlc(
    *,
    symbol: str = DEFAULT_EURUSD_SYMBOL,
    start: str | pd.Timestamp = DEFAULT_START,
    end: str | pd.Timestamp | None = None,
    shutdown: bool = True,
    symbol_suffixes: Sequence[str] = ("", ".a", ".b", ".pro", "_SB"),
) -> pd.DataFrame:
    """Pull daily OHLCV bars from MetaTrader5."""
    try:
        import MetaTrader5 as mt5
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MetaTrader5 is required for the EURUSD daily pull. "
            "Install it with `pip install MetaTrader5` in the notebook environment."
        ) from exc

    start_dt = _as_utc_datetime(start)
    end_dt = _as_utc_datetime(_normalize_end(end) + pd.Timedelta(days=1))

    initialized_here = False
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    initialized_here = True

    try:
        last_error: Any = None
        for suffix in symbol_suffixes:
            candidate = symbol if suffix == "" else f"{symbol}{suffix}"
            mt5.symbol_select(candidate, True)
            rates = mt5.copy_rates_range(candidate, mt5.TIMEFRAME_D1, start_dt, end_dt)
            if rates is not None and len(rates) > 0:
                out = pd.DataFrame(rates)
                out["date"] = pd.to_datetime(out["time"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
                out = out.drop(columns=["time"]).set_index("date").sort_index()
                out["symbol"] = candidate
                numeric_cols = [col for col in out.columns if col != "symbol"]
                out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
                return out[["symbol", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
            last_error = mt5.last_error()

        raise RuntimeError(f"No MT5 daily rates returned for {symbol}. Last MT5 error: {last_error}")
    finally:
        if shutdown and initialized_here:
            mt5.shutdown()


def pull_fred_features(
    *,
    start: str | pd.Timestamp = DEFAULT_START,
    end: str | pd.Timestamp | None = None,
    api_key: str | None = None,
    series: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Pull configured FRED series and return one numeric column per feature."""
    api_key = api_key or os.getenv("FRED_API_KEY") or DEFAULT_FRED_API_KEY
    if not api_key:
        raise ValueError("A FRED API key is required.")

    start_date = _normalize_date(start).strftime("%Y-%m-%d")
    end_date = _normalize_end(end).strftime("%Y-%m-%d")
    features = dict(series or FRED_SERIES)
    frames: list[pd.Series] = []

    for feature_name, series_id in features.items():
        observations = _fred_observations(series_id, api_key, start_date, end_date)
        values = pd.DataFrame(observations)
        if values.empty:
            raise RuntimeError(f"FRED returned no observations for {series_id}.")
        values["date"] = pd.to_datetime(values["date"]).dt.normalize()
        values[feature_name] = pd.to_numeric(values["value"].replace(".", np.nan), errors="coerce")
        frames.append(values.set_index("date")[feature_name].sort_index())

    out = pd.concat(frames, axis=1).sort_index()
    return out


def pull_ecb_features(
    *,
    start: str | pd.Timestamp = DEFAULT_START,
    end: str | pd.Timestamp | None = None,
    series: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Pull configured ECB Data Portal series via ecbdata."""
    try:
        from ecbdata import ecbdata
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ecbdata is required for ECB/EU macro features. "
            "Install it with `pip install ecbdata` in the notebook environment."
        ) from exc

    features = dict(series or ECB_SERIES)
    frames: list[pd.Series] = []

    for feature_name, series_key in features.items():
        start_date, end_date = _ecb_date_bounds(series_key, start, end)
        values = ecbdata.get_series(series_key, start=start_date, end=end_date)
        if values.empty:
            raise RuntimeError(f"ECB returned no observations for {series_key}.")
        date_col = _first_existing_col(values, ["TIME_PERIOD", "time_period", "date"])
        value_col = _first_existing_col(values, ["OBS_VALUE", "obs_value", "value"])
        frame = values[[date_col, value_col]].copy()
        frame["date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize()
        frame[feature_name] = pd.to_numeric(frame[value_col], errors="coerce")
        frames.append(frame.dropna(subset=["date"]).set_index("date")[feature_name].sort_index())

    return pd.concat(frames, axis=1).sort_index()


def pull_cot_euro_positioning_features(
    *,
    start: str | pd.Timestamp = DEFAULT_START,
    end: str | pd.Timestamp | None = None,
    report_type: str = "legacy_fut",
    release_lag_days: int = 3,
) -> pd.DataFrame:
    """Pull EUR futures COT non-commercial positioning with cot_reports."""
    try:
        import cot_reports as cot
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "cot_reports is required for COT positioning features. "
            "Install it with `pip install cot_reports` in the notebook environment."
        ) from exc

    start_ts = _normalize_date(start)
    end_ts = _normalize_end(end)
    years = range(start_ts.year, end_ts.year + 1)

    yearly_frames = []
    for year in years:
        try:
            yearly_frames.append(cot.cot_year(year=year, cot_report_type=report_type))
        except TypeError:
            yearly_frames.append(cot.cot_year(year, report_type))

    raw = pd.concat(yearly_frames, ignore_index=True)
    raw = _with_normalized_columns(raw)

    market_col = _first_existing_col(
        raw,
        [
            "market_and_exchange_names",
            "market_and_exchange_name",
            "contract_market_name",
            "market",
        ],
    )
    date_col = _first_existing_col(
        raw,
        [
            "as_of_date_in_form_yyyy_mm_dd",
            "report_date_as_yyyy_mm_dd",
            "as_of_date",
            "date",
        ],
    )
    long_col = _first_existing_col(
        raw,
        [
            "noncommercial_positions_long_all",
            "noncommercial_positions_long",
            "noncommercial_long_all",
            "noncommercial_long",
        ],
    )
    short_col = _first_existing_col(
        raw,
        [
            "noncommercial_positions_short_all",
            "noncommercial_positions_short",
            "noncommercial_short_all",
            "noncommercial_short",
        ],
    )

    market = raw[market_col].astype(str).str.upper()
    euro = raw[market.str.contains("EURO FX", na=False) & ~market.str.contains("EURODOLLAR", na=False)].copy()
    if euro.empty:
        examples = raw[market_col].dropna().astype(str).head(10).tolist()
        raise RuntimeError(f"No EURO FX COT rows found. Example markets: {examples}")

    euro["date"] = pd.to_datetime(euro[date_col], errors="coerce").dt.normalize()
    euro["net"] = _to_numeric_clean(euro[long_col]) - _to_numeric_clean(euro[short_col])
    euro = euro.dropna(subset=["date", "net"]).sort_values("date")
    weekly = euro.groupby("date", as_index=True)["net"].last().to_frame("cot_euro_net_noncommercial")
    weekly = weekly.loc[(weekly.index >= start_ts) & (weekly.index <= end_ts)]
    if weekly.empty:
        raise RuntimeError("No EURO FX COT observations in the requested date range.")

    weekly["cot_euro_net_4w_change"] = weekly["cot_euro_net_noncommercial"].diff(4)
    rolling_mean = weekly["cot_euro_net_noncommercial"].rolling(156, min_periods=52).mean()
    rolling_std = weekly["cot_euro_net_noncommercial"].rolling(156, min_periods=52).std()
    weekly["cot_euro_net_z_156w"] = (weekly["cot_euro_net_noncommercial"] - rolling_mean) / rolling_std
    weekly["cot_euro_net_4w_ma"] = weekly["cot_euro_net_noncommercial"].rolling(4, min_periods=1).mean()

    weekly.index = weekly.index + pd.Timedelta(days=release_lag_days)
    weekly.index = weekly.index.normalize()
    return weekly.sort_index()


def add_price_action_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add all notebook price-action, trend, momentum, and volatility features."""
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f"Missing OHLC columns: {missing}")

    out = data.copy().sort_index()
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")

    out["ret_1d"] = np.log(close / close.shift(1))
    out["ret_5d"] = np.log(close / close.shift(5))
    out["ret_20d"] = np.log(close / close.shift(20))
    out["vol_20d"] = out["ret_1d"].rolling(20).std() * np.sqrt(252)
    out["vol_63d"] = out["ret_1d"].rolling(63).std() * np.sqrt(252)

    for window in (10, 20, 50, 100, 200):
        sma_col = f"sma_{window}"
        ema_col = f"ema_{window}"
        out[sma_col] = close.rolling(window, min_periods=window).mean()
        out[ema_col] = close.ewm(span=window, adjust=False, min_periods=window).mean()
        out[f"dist_sma_{window}"] = close / out[sma_col] - 1.0

    out["rsi_14"] = _rsi(close, 14)
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["macd"] = ema_12 - ema_26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    bb_mean = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    out["bb_z_20"] = (close - bb_mean) / bb_std

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    out["atr_pct_14"] = out["atr_14"] / close

    prior_high_20 = high.shift(1).rolling(20, min_periods=20).max()
    prior_low_20 = low.shift(1).rolling(20, min_periods=20).min()
    out["breakout_20"] = np.select([close > prior_high_20, close < prior_low_20], [1.0, -1.0], default=0.0)
    out.loc[prior_high_20.isna() | prior_low_20.isna(), "breakout_20"] = np.nan
    out["drawdown_20"] = close / close.rolling(20, min_periods=20).max() - 1.0
    out["range_pct"] = (high - low) / close

    return out


def add_vix_regimes(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    if "vix" not in out.columns:
        out["vix_low_regime"] = np.nan
        out["vix_mid_regime"] = np.nan
        out["vix_high_regime"] = np.nan
        return out

    vix = pd.to_numeric(out["vix"], errors="coerce")
    out["vix_low_regime"] = np.where(vix.notna(), (vix < 15.0).astype(float), np.nan)
    out["vix_mid_regime"] = np.where(vix.notna(), ((vix >= 15.0) & (vix < 25.0)).astype(float), np.nan)
    out["vix_high_regime"] = np.where(vix.notna(), (vix >= 25.0).astype(float), np.nan)
    return out


def add_eu_macro_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()

    if {"eu_10y_nominal", "eu_2y_nominal"}.issubset(out.columns):
        out["eu_curve_10y2y"] = pd.to_numeric(out["eu_10y_nominal"], errors="coerce") - pd.to_numeric(
            out["eu_2y_nominal"], errors="coerce"
        )

    if {"eu_10y_nominal", "eu_hicp_yoy"}.issubset(out.columns):
        out["eu_10y_real"] = pd.to_numeric(out["eu_10y_nominal"], errors="coerce") - pd.to_numeric(
            out["eu_hicp_yoy"], errors="coerce"
        )

    if "eu_ciss" not in out.columns:
        out["eu_ciss_low_regime"] = np.nan
        out["eu_ciss_mid_regime"] = np.nan
        out["eu_ciss_high_regime"] = np.nan
        return out

    ciss = pd.to_numeric(out["eu_ciss"], errors="coerce")
    out["eu_ciss_low_regime"] = np.where(ciss.notna(), (ciss < 0.10).astype(float), np.nan)
    out["eu_ciss_mid_regime"] = np.where(ciss.notna(), ((ciss >= 0.10) & (ciss < 0.30)).astype(float), np.nan)
    out["eu_ciss_high_regime"] = np.where(ciss.notna(), (ciss >= 0.30).astype(float), np.nan)
    return out


def add_cycle_features(
    data: pd.DataFrame,
    *,
    anchor_date: str | pd.Timestamp = "2000-01-01",
    kitchin_months: float = 40.0,
    juglar_years: float = 8.0,
) -> pd.DataFrame:
    out = data.copy()
    anchor = pd.Timestamp(anchor_date).normalize()
    elapsed_days = (pd.DatetimeIndex(out.index).normalize() - anchor).days.to_numpy(dtype=float)
    kitchin_days = kitchin_months * 365.25 / 12.0
    juglar_days = juglar_years * 365.25

    out["kitchin_cycle_sin"] = np.sin(2.0 * np.pi * elapsed_days / kitchin_days)
    out["kitchin_cycle_cos"] = np.cos(2.0 * np.pi * elapsed_days / kitchin_days)
    out["juglar_cycle_sin"] = np.sin(2.0 * np.pi * elapsed_days / juglar_days)
    out["juglar_cycle_cos"] = np.cos(2.0 * np.pi * elapsed_days / juglar_days)
    return out


def write_feature_table(data: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    out = data.copy()
    out.index.name = "date"
    if suffix == ".csv":
        out.to_csv(path)
    elif suffix in {".parquet", ".pq"}:
        out.to_parquet(path)
    elif suffix in {".pkl", ".pickle"}:
        out.to_pickle(path)
    elif suffix == ".feather":
        out.reset_index().to_feather(path)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def _fred_observations(series_id: str, api_key: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    response = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if "error_message" in payload:
        raise RuntimeError(f"FRED error for {series_id}: {payload['error_message']}")
    return payload.get("observations", [])


def _align_to_index(data: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    source = data.copy().sort_index()
    source.index = pd.DatetimeIndex(source.index).normalize()
    target_index = pd.DatetimeIndex(index).normalize()
    return source.reindex(source.index.union(target_index)).ffill().reindex(target_index)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _normalize_end(end: str | pd.Timestamp | None) -> pd.Timestamp:
    if end is None:
        return _normalize_date(pd.Timestamp.utcnow())
    return _normalize_date(end)


def _normalize_date(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def _as_utc_datetime(value: str | pd.Timestamp) -> Any:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _ecb_date_bounds(
    series_key: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
) -> tuple[str, str]:
    frequency = series_key.split(".", 2)[1] if "." in series_key else ""
    start_ts = _normalize_date(start)
    end_ts = _normalize_end(end)
    if frequency == "M":
        return start_ts.strftime("%Y-%m"), end_ts.strftime("%Y-%m")
    if frequency == "Q":
        return f"{start_ts.year}-Q{start_ts.quarter}", f"{end_ts.year}-Q{end_ts.quarter}"
    if frequency == "A":
        return str(start_ts.year), str(end_ts.year)
    return start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")


def _with_normalized_columns(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out.columns = [_normalize_name(col) for col in out.columns]
    return out


def _normalize_name(value: Any) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _first_existing_col(data: pd.DataFrame, candidates: Sequence[str]) -> str:
    columns = set(data.columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"None of these columns were found: {list(candidates)}")


def _to_numeric_clean(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype(str).str.replace(",", "", regex=False), errors="coerce")
