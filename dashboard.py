"""
dashboard.py — OU rolling grid builder for MetaMTOU.

Provides two functions consumed by MetaMTOU.fit():
  get_all_second_monday_opens(ohlc)           -> dict[int, pd.Series]
  compute_rolling_grid(ohlc, X_log, sm, ...)  -> pd.DataFrame

OU model: ΔX_t = α + κ(μ_i − X_{t−1}) + ε_t
  κ > 0  → price reverts to level μ_i  (regime_active = True when κ > 0 and p-val < 0.10)
  z_score > 0  → price above attractor  (wz > 0 → SHORT signal in MetaMTOU)
  level_price  → attractor in raw price space
  resid_vol_daily → residual σ of Δlog(price), used for stop sizing
  half_life_days  → log(2) / κ
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as _t_dist

# Make sure metalib is importable when called from any working directory
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from metalib.indicators import get_second_monday_open_ffill

_LAGS = [0, 1, 2]
_TRADING_DAYS_PER_MONTH = 21
_PVAL_THRESH = 0.10          # κ must be significant at this level for regime_active=True


def get_all_second_monday_opens(ohlc: pd.DataFrame) -> dict:
    """
    Return {lag: pd.Series} of second-Monday 'open' prices forward-filled to ohlc.index.
    lag=0 → current month anchor; lag=1 → previous month; lag=2 → two months ago.
    """
    return {
        lag: get_second_monday_open_ffill(ohlc, ohlc.index, i=lag)
        for lag in _LAGS
    }


def _fit_ou(x_window: np.ndarray, level_log: float) -> dict | None:
    """
    OLS: ΔX_t = α + κ(μ − X_{t−1}) + ε_t
    Uses manual sum-based OLS to avoid BLAS/LAPACK calls (which are unstable
    in certain conda envs on Windows). p-value via scipy.stats.t.sf.

    Returns dict with kappa, pval_kappa, resid_vol_daily, half_life_days, z_score.
    Returns None when the window is too short or degenerate.
    """
    n = len(x_window)
    if n < 12:
        return None

    dx = np.diff(x_window)                  # log-return series, length n-1
    spread = level_log - x_window[:-1]      # μ − X_{t−1}
    m = len(dx)  # == n - 1

    ss = spread.sum()
    ss2 = (spread * spread).sum()
    D = m * ss2 - ss * ss
    if abs(D) < 1e-20:
        return None

    sd = (spread * dx).sum()
    yd = dx.sum()

    kappa = float((m * sd - ss * yd) / D)          # slope (= κ)
    alpha = float((yd - kappa * ss) / m)            # intercept

    resid = dx - alpha - kappa * spread
    dof = max(m - 2, 1)
    mse = float((resid * resid).sum() / dof)
    se_kappa = float(np.sqrt(mse * m / D)) if mse > 0 else 1e-12

    t_stat = kappa / se_kappa if se_kappa > 1e-15 else 0.0
    pval = float(2.0 * _t_dist.sf(abs(t_stat), df=dof))

    resid_std = float(np.sqrt(mse))
    half_life = float(np.log(2) / kappa) if kappa > 1e-9 else float("nan")

    # z_score > 0 when current price is ABOVE the attractor
    z = (x_window[-1] - level_log) / resid_std if resid_std > 1e-12 else float("nan")

    return {
        "kappa":           kappa,
        "pval_kappa":      pval,
        "resid_vol_daily": resid_std,
        "half_life_days":  half_life,
        "z_score":         z,
    }


def compute_rolling_grid(
    ohlc: pd.DataFrame,
    X_log: pd.Series,
    sm: dict,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build the full rolling OU grid across all eval_dates, horizons, and lags.

    Parameters
    ----------
    ohlc     : daily OHLC DataFrame with DatetimeIndex
    X_log    : log-close Series, same index as ohlc
    sm       : dict {lag: pd.Series} from get_all_second_monday_opens()
    horizons : horizon lengths in months (default [3, 6])

    Returns
    -------
    DataFrame columns:
        eval_date, horizon_months, lag, regime_active,
        z_score, pval_kappa, level_price, resid_vol_daily, half_life_days
    """
    if horizons is None:
        horizons = [3, 6]

    _COLS = [
        "eval_date", "horizon_months", "lag", "regime_active",
        "z_score", "pval_kappa", "level_price", "resid_vol_daily", "half_life_days",
    ]

    idx = X_log.index
    vals = X_log.values
    rows: list[dict] = []

    for h in horizons:
        window = int(h * _TRADING_DAYS_PER_MONTH)

        for i in range(window, len(idx)):
            eval_date = idx[i]
            x_win = vals[i - window: i + 1]

            for lag in _LAGS:
                lvl_px = sm[lag].iloc[i]
                if pd.isna(lvl_px) or lvl_px <= 0:
                    continue

                lvl_log = np.log(float(lvl_px))
                fit = _fit_ou(x_win, lvl_log)
                if fit is None:
                    continue

                active = fit["kappa"] > 0 and fit["pval_kappa"] < _PVAL_THRESH

                rows.append({
                    "eval_date":       eval_date,
                    "horizon_months":  h,
                    "lag":             lag,
                    "regime_active":   active,
                    "z_score":         fit["z_score"],
                    "pval_kappa":      fit["pval_kappa"],
                    "level_price":     float(lvl_px),
                    "resid_vol_daily": fit["resid_vol_daily"],
                    "half_life_days":  fit["half_life_days"],
                })

    if not rows:
        return pd.DataFrame(columns=_COLS)

    df = pd.DataFrame(rows)
    df["eval_date"] = pd.to_datetime(df["eval_date"])
    return df
