# EURUSD MTOFX Daily Feature Reference

This file documents the daily feature table produced by `metalib.mtofx_daily_data.build_eurusd_mtofx_daily_features`.

## Source Documentation

| Source | Used for | Documentation |
|---|---|---|
| MetaTrader5 Python API | EURUSD daily OHLCV bars via `copy_rates_range(..., TIMEFRAME_D1, ...)` | https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py |
| FRED API | US rates, real yield, VIX, Fed funds, US curve spread | https://fred.stlouisfed.org/docs/api/fred/series_observations.html |
| FRED series pages | Individual US macro/risk series metadata | https://fred.stlouisfed.org/series/DGS10 |
| ECB Data Portal API | Euro-area rates, ECB policy rates, HICP, CISS | https://data.ecb.europa.eu/help/api/data |
| `ecbdata` Python package | Python wrapper around the ECB Data Portal | https://pypi.org/project/ecbdata/ |
| CFTC COT reports | EURO FX futures positioning source data | https://www.cftc.gov/MarketReports/CommitmentsofTraders/AbouttheCOTReports/index.htm |
| CFTC COT legacy field names | Non-commercial long/short field definitions | https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalViewable/deanexplanatory.html |
| `cot_reports` Python package | Python wrapper used to download annual COT files | https://pypi.org/project/cot-reports/ |

## Raw Price Data

| Column | Source | Description | Documentation |
|---|---|---|---|
| `symbol` | MetaTrader5 | Broker symbol that returned data, e.g. `EURUSD` or a suffixed broker variant. | https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py |
| `open` | MetaTrader5 | Daily bar open. | https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py |
| `high` | MetaTrader5 | Daily bar high. | https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py |
| `low` | MetaTrader5 | Daily bar low. | https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py |
| `close` | MetaTrader5 | Daily bar close. All price-action features are derived from this and daily OHLC. | https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py |
| `tick_volume` | MetaTrader5 | Tick volume from the MT5 daily bar. | https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py |
| `real_volume` | MetaTrader5 | Real volume if supplied by the broker; often zero for spot FX. | https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py |
| `spread` | MetaTrader5 | Daily bar spread field supplied by MT5. | https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py |

## Price Action And Technical Features

These features are calculated locally from MT5 daily OHLC data.

| Feature | Group | Construction | Input |
|---|---|---|---|
| `ret_1d` | `price_action` | `log(close / close.shift(1))`. | `close` |
| `ret_5d` | `momentum_trend` | `log(close / close.shift(5))`. | `close` |
| `ret_20d` | `momentum_trend` | `log(close / close.shift(20))`. | `close` |
| `vol_20d` | `realized_volatility` | 20-day rolling standard deviation of `ret_1d`, annualized by `sqrt(252)`. | `ret_1d` |
| `vol_63d` | `realized_volatility` | 63-day rolling standard deviation of `ret_1d`, annualized by `sqrt(252)`. | `ret_1d` |
| `sma_10` | `momentum_trend` | 10-day simple moving average. | `close` |
| `ema_10` | `momentum_trend` | 10-day exponential moving average. | `close` |
| `dist_sma_10` | `momentum_trend` | `close / sma_10 - 1`. | `close`, `sma_10` |
| `sma_20` | `momentum_trend` | 20-day simple moving average. | `close` |
| `ema_20` | `momentum_trend` | 20-day exponential moving average. | `close` |
| `dist_sma_20` | `momentum_trend` | `close / sma_20 - 1`. | `close`, `sma_20` |
| `sma_50` | `momentum_trend` | 50-day simple moving average. | `close` |
| `ema_50` | `momentum_trend` | 50-day exponential moving average. | `close` |
| `dist_sma_50` | `momentum_trend` | `close / sma_50 - 1`. | `close`, `sma_50` |
| `sma_100` | `momentum_trend` | 100-day simple moving average. | `close` |
| `ema_100` | `momentum_trend` | 100-day exponential moving average. | `close` |
| `dist_sma_100` | `momentum_trend` | `close / sma_100 - 1`. | `close`, `sma_100` |
| `sma_200` | `momentum_trend` | 200-day simple moving average. | `close` |
| `ema_200` | `momentum_trend` | 200-day exponential moving average. | `close` |
| `dist_sma_200` | `momentum_trend` | `close / sma_200 - 1`. | `close`, `sma_200` |
| `rsi_14` | `momentum_trend` | 14-period RSI using exponentially smoothed average gains/losses. | `close` |
| `macd` | `momentum_trend` | 12-day EMA minus 26-day EMA. | `close` |
| `macd_signal` | `momentum_trend` | 9-day EMA of `macd`. | `macd` |
| `macd_hist` | `momentum_trend` | `macd - macd_signal`. | `macd`, `macd_signal` |
| `bb_z_20` | `price_action` | `(close - rolling_mean_20) / rolling_std_20`. | `close` |
| `atr_14` | `realized_volatility` | 14-day rolling mean of true range. | `high`, `low`, `close.shift(1)` |
| `atr_pct_14` | `realized_volatility` | `atr_14 / close`. | `atr_14`, `close` |
| `breakout_20` | `price_action` | `1` if close breaks prior 20-day high, `-1` if it breaks prior 20-day low, else `0`. | `high`, `low`, `close` |
| `drawdown_20` | `price_action` | `close / rolling_max_20(close) - 1`. | `close` |
| `range_pct` | `price_action` | `(high - low) / close`. | `high`, `low`, `close` |

## FRED US Macro And Risk Features

All FRED series are pulled with the FRED observations API and forward-filled onto the MT5 trading-date index.

| Feature | FRED series | Group | Description | Documentation |
|---|---|---|---|---|
| `us_10y_nominal` | `DGS10` | `macro_rates` | US 10-year Treasury constant maturity yield, percent, daily. | https://fred.stlouisfed.org/series/DGS10 |
| `us_2y_nominal` | `DGS2` | `macro_rates` | US 2-year Treasury constant maturity yield, percent, daily. | https://fred.stlouisfed.org/series/DGS2 |
| `us_10y_real` | `DFII10` | `macro_rates` | US 10-year inflation-indexed Treasury constant maturity yield, percent, daily. | https://fred.stlouisfed.org/series/DFII10 |
| `vix` | `VIXCLS` | `risk_confidence` | CBOE VIX close, index level, daily. | https://fred.stlouisfed.org/series/VIXCLS |
| `fed_funds` | `DFF` | `macro_rates` | Effective federal funds rate, percent, daily. | https://fred.stlouisfed.org/series/DFF |
| `us_curve_10y2y` | `T10Y2Y` | `macro_rates` | US 10-year minus 2-year Treasury spread, percent, daily. | https://fred.stlouisfed.org/series/T10Y2Y |
| `vix_low_regime` | derived from `VIXCLS` | `risk_confidence` | `1` when `vix < 15`, else `0`; missing when VIX is missing. | https://fred.stlouisfed.org/series/VIXCLS |
| `vix_mid_regime` | derived from `VIXCLS` | `risk_confidence` | `1` when `15 <= vix < 25`, else `0`; missing when VIX is missing. | https://fred.stlouisfed.org/series/VIXCLS |
| `vix_high_regime` | derived from `VIXCLS` | `risk_confidence` | `1` when `vix >= 25`, else `0`; missing when VIX is missing. | https://fred.stlouisfed.org/series/VIXCLS |

## ECB / EU Macro And Risk Features

ECB series are pulled through `ecbdata.get_series(...)` and forward-filled onto the MT5 trading-date index.

| Feature | ECB series key | Group | Description | Documentation |
|---|---|---|---|---|
| `eu_10y_nominal` | `YC.B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y` | `macro_rates` | All euro-area yield curve 10-year spot rate, nominal government bonds, all issuers/all ratings. | https://data.ecb.europa.eu/data/datasets/YC/YC.B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y |
| `eu_2y_nominal` | `YC.B.U2.EUR.4F.G_N_C.SV_C_YM.SR_2Y` | `macro_rates` | All euro-area yield curve 2-year spot rate, nominal government bonds, all issuers/all ratings. | https://data.ecb.europa.eu/data/datasets/YC/YC.B.U2.EUR.4F.G_N_C.SV_C_YM.SR_2Y |
| `ecb_deposit_rate` | `FM.D.U2.EUR.4F.KR.DFR.LEV` | `macro_rates` | ECB deposit facility rate, percent. | https://data.ecb.europa.eu/data/datasets/FM/FM.D.U2.EUR.4F.KR.DFR.LEV |
| `ecb_main_refi_rate` | `FM.D.U2.EUR.4F.KR.MRR_FR.LEV` | `macro_rates` | ECB main refinancing operations fixed rate tender rate, percent. | https://data.ecb.europa.eu/data/datasets/FM/FM.D.U2.EUR.4F.KR.MRR_FR.LEV |
| `eu_ciss` | `CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX` | `risk_confidence` | ECB new Composite Indicator of Systemic Stress for the euro area. | https://data.ecb.europa.eu/data/datasets/CISS/CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX |
| `eu_hicp_yoy` | `ICP.M.U2.N.000000.4.ANR` | `macro_rates` | Euro-area HICP overall index, annual rate of change. | https://data.ecb.europa.eu/data/datasets/ICP/ICP.M.U2.N.000000.4.ANR |
| `eu_curve_10y2y` | derived | `macro_rates` | `eu_10y_nominal - eu_2y_nominal`. | https://data.ecb.europa.eu/data/datasets/YC |
| `eu_10y_real` | derived | `macro_rates` | Proxy real yield: `eu_10y_nominal - eu_hicp_yoy`. | https://data.ecb.europa.eu/data/datasets/YC |
| `eu_ciss_low_regime` | derived from `eu_ciss` | `risk_confidence` | `1` when `eu_ciss < 0.10`, else `0`; missing when CISS is missing. | https://data.ecb.europa.eu/data/datasets/CISS/CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX |
| `eu_ciss_mid_regime` | derived from `eu_ciss` | `risk_confidence` | `1` when `0.10 <= eu_ciss < 0.30`, else `0`; missing when CISS is missing. | https://data.ecb.europa.eu/data/datasets/CISS/CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX |
| `eu_ciss_high_regime` | derived from `eu_ciss` | `risk_confidence` | `1` when `eu_ciss >= 0.30`, else `0`; missing when CISS is missing. | https://data.ecb.europa.eu/data/datasets/CISS/CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX |

## COT Positioning Features

The COT pull uses `cot_reports.cot_year(..., cot_report_type="legacy_fut")`, filters for `EURO FX`, and computes non-commercial net positioning from the CFTC legacy report fields.

| Feature | Source fields | Group | Construction | Documentation |
|---|---|---|---|---|
| `cot_euro_net_noncommercial` | `Noncommercial Positions-Long (All)`, `Noncommercial Positions-Short (All)` | `positioning_flows` | Long minus short for EURO FX futures. | https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalViewable/deanexplanatory.html |
| `cot_euro_net_4w_change` | `cot_euro_net_noncommercial` | `positioning_flows` | 4-report difference in non-commercial net position. | https://www.cftc.gov/MarketReports/CommitmentsofTraders/AbouttheCOTReports/index.htm |
| `cot_euro_net_z_156w` | `cot_euro_net_noncommercial` | `positioning_flows` | Z-score versus 156-week rolling mean and standard deviation, with at least 52 observations. | https://www.cftc.gov/MarketReports/CommitmentsofTraders/AbouttheCOTReports/index.htm |
| `cot_euro_net_4w_ma` | `cot_euro_net_noncommercial` | `positioning_flows` | 4-report rolling mean of non-commercial net position. | https://www.cftc.gov/MarketReports/CommitmentsofTraders/AbouttheCOTReports/index.htm |

## Cycle Features

Cycle features are deterministic calendar encodings. They are not pulled from an external data provider.

| Feature | Group | Construction | Notes |
|---|---|---|---|
| `kitchin_cycle_sin` | `cycles_regime` | `sin(2*pi*elapsed_days/kitchin_days)`, with default 40-month cycle length. | Anchor date: `2000-01-01`. |
| `kitchin_cycle_cos` | `cycles_regime` | `cos(2*pi*elapsed_days/kitchin_days)`, with default 40-month cycle length. | Anchor date: `2000-01-01`. |
| `juglar_cycle_sin` | `cycles_regime` | `sin(2*pi*elapsed_days/juglar_days)`, with default 8-year cycle length. | Anchor date: `2000-01-01`. |
| `juglar_cycle_cos` | `cycles_regime` | `cos(2*pi*elapsed_days/juglar_days)`, with default 8-year cycle length. | Anchor date: `2000-01-01`. |

## Target

| Feature | Construction | Notes |
|---|---|---|
| `target_ret_1d_fwd` | `ret_1d.shift(-1)` | One-day forward EURUSD log return used as the default regression target. |
