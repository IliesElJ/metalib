"""
Walk-forward, regularized (Lasso) trend-continuation gate -- an upgrade to
metafvg_variants.py's "Regression Gate" (hard-threshold rolling-OLS slope/R^2)
that adds two-window realized-volatility features and a small curated set of
trend x volatility interaction terms, then fits a numba coordinate-descent
Lasso -- BLAS-free, since this env's BLAS crashes on any matrix-matrix
multiply (GEMM); see metalib/notebooks/metahar_wf_backtest.py, whose
_lasso_path_cd/fit_lasso_cv pattern this duplicates rather than imports
(notebooks/ is research code, not a stable import target for backtest
scripts) -- walk-forward over the HTF series to predict the forward-K-bar log
return.

No-lookahead discipline:
  - Features (slope, r2, realized vol) at bar t are trailing-window rolling
    stats using only bars <= t -- always valid to use "live" at t.
  - The target at bar t (forward-K-bar log return) is only *realized* once
    bar t+K has occurred. A walk-forward fit "as of" cutoff bar `cut` uses
    training rows only where t+K <= cut, i.e. strictly-in-the-past, fully
    -observed outcomes -- never a training example whose target reaches past
    the fit's own cutoff.
  - Refits happen periodically (every test_bars bars); predictions for the
    bars between refits are genuinely out-of-sample (produced by a model
    fit strictly before those bars).
  - The gating magnitude threshold is a *causal* rolling quantile of the
    prediction series' own absolute value, shifted by one bar so the
    threshold at t never uses a prediction generated at or after t.

This whole pass runs once per symbol, ahead of the bar-by-bar backtest
simulation (metafvg_variants.py's simulate_fvg_trades_variant) -- refitting a
regularized model at every single simulated bar would be a needless
quadratic blowup; the simulation loop instead just looks up this
precomputed, already-leak-free prediction/threshold at each signal check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from metafvg_variants import _rolling_slope_r2
from metalib.fastfinance import ema as _ema_fn

# =========================================================================
# BLAS-free Lasso (numba coordinate descent), duplicated from
# metahar_wf_backtest.py::_lasso_path_cd / fit_lasso_cv.
# =========================================================================

@njit(cache=True)
def _lasso_path_cd(X, y, alphas, max_iter, tol):
    """Coordinate-descent Lasso path, sklearn objective
    1/(2n)*||y-Xw||^2 + alpha*||w||_1, warm-started along descending alphas.
    Explicit loops only -- numba's np.dot lowers to the broken BLAS."""
    n, p = X.shape
    coefs = np.zeros((alphas.shape[0], p))
    w = np.zeros(p)
    r = y.copy()
    col_sq = np.zeros(p)
    for j in range(p):
        s = 0.0
        for i in range(n):
            s += X[i, j] * X[i, j]
        col_sq[j] = s
    for a in range(alphas.shape[0]):
        lam = alphas[a] * n
        for _ in range(max_iter):
            max_delta = 0.0
            for j in range(p):
                if col_sq[j] <= 0.0:
                    continue
                rho = 0.0
                for i in range(n):
                    rho += X[i, j] * r[i]
                rho += col_sq[j] * w[j]
                if rho > lam:
                    w_new = (rho - lam) / col_sq[j]
                elif rho < -lam:
                    w_new = (rho + lam) / col_sq[j]
                else:
                    w_new = 0.0
                d = w_new - w[j]
                if d != 0.0:
                    for i in range(n):
                        r[i] -= X[i, j] * d
                    if abs(d) > max_delta:
                        max_delta = abs(d)
                w[j] = w_new
            if max_delta < tol:
                break
        coefs[a] = w
    return coefs


def fit_lasso_cv(X_df: pd.DataFrame, y_ser: pd.Series, n_alphas=60, cv=5, max_iter=300, tol=1e-6):
    """BLAS-free Lasso: standardize, pick alpha by contiguous K-fold CV MSE,
    refit on the full training set. Returns a predict(X_df) -> np.ndarray callable."""
    X = X_df.to_numpy(dtype=np.float64)
    y = y_ser.to_numpy(dtype=np.float64)
    n = len(y)

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0  # constant cols (e.g. 'const') -> coef forced to 0
    Xs = (X - mu) / sd
    ym = y.mean()
    yc = y - ym

    alpha_max = max(np.abs((Xs * yc[:, None]).sum(axis=0)).max() / n, 1e-12)
    alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_max * 1e-4), n_alphas)

    fold_edges = np.linspace(0, n, cv + 1).astype(int)
    mse = np.zeros(len(alphas))
    for k in range(cv):
        lo, hi = fold_edges[k], fold_edges[k + 1]
        tr = np.ones(n, dtype=bool)
        tr[lo:hi] = False
        coefs = _lasso_path_cd(Xs[tr], yc[tr], alphas, max_iter, tol)
        for a in range(len(alphas)):
            resid = yc[lo:hi] - (Xs[lo:hi] * coefs[a]).sum(axis=1)
            mse[a] += (resid ** 2).mean()
    best = int(np.argmin(mse))

    coefs = _lasso_path_cd(Xs, yc, alphas[: best + 1], max_iter, tol)
    w = coefs[-1]

    def predict(X_new_df: pd.DataFrame) -> np.ndarray:
        Xn = (X_new_df.to_numpy(dtype=np.float64) - mu) / sd
        return ym + (Xn * w).sum(axis=1)

    return predict, alphas[best], w


# =========================================================================
# Features
# =========================================================================

def build_features(
    close: pd.Series,
    regression_window: int = 20,
    rv_short_window: int = 10,
    rv_long_window: int = 60,
    ema_fast_period: int = 9,
    ema_slow_period: int = 21,
) -> pd.DataFrame:
    """
    slope, r2      : rolling-OLS trend (same as the existing Regression Gate)
    rv_short/long  : realized vol of log returns over two lookback windows
    rv_ratio       : rv_short / rv_long -- vol-regime signal (expanding vs
                     contracting)
    ema_fast_vadj,
    ema_slow_vadj  : EMA(fast=9, slow=21) of *vol-adjusted* log returns
                     (log_ret / rv_short), not of price -- each bar's return
                     is first normalized by trailing realized vol so the
                     smoothed series is comparable across vol regimes, then
                     EMA'd. The EMA value itself is the signal here (a
                     smoothed vol-adjusted-momentum estimate), unlike a
                     price-level EMA there's no natural "distance" to take.
    ema_cross_vadj : ema_fast_vadj - ema_slow_vadj -- momentum-acceleration
                     signal, positive when recent vol-adjusted momentum is
                     stronger than the longer trailing average.
                     Uses the ema() already fixed this session for leading-
                     and mid-series-NaN handling (fastfinance.py).
    interactions   : a small curated set (not full polynomial expansion) --
                     Lasso shrinks whichever of these don't matter, so this
                     stays interpretable by construction rather than by
                     hand-picking.

    `close` must already be gap-free (no NaN rows) -- see
    compute_lasso_gate_series for why: the raw HTF series resampled from an
    LTF with real market-closed gaps (e.g. weekends on a sub-daily HTF) has
    ~30% NaN rows, and a strict rolling window (pandas' default
    min_periods=window) essentially never sees a clean window over data that
    gappy -- a 60-bar window had 0/8819 non-NaN outputs measured on real
    EURUSD 4h data. Compute on the densified (gap-dropped) series instead,
    and reindex the *final* walk-forward output back onto the full HTF index
    with ffill (compute_lasso_gate_series does this) -- the same
    hold-steady-across-gaps principle as the ema() fix in fastfinance.py,
    applied at the orchestration layer instead of inside the indicator.
    """
    slope, r2 = _rolling_slope_r2(close.values, regression_window)
    log_ret = np.log(close).diff()
    rv_short = log_ret.rolling(rv_short_window).std(ddof=0)
    rv_long = log_ret.rolling(rv_long_window).std(ddof=0)
    rv_ratio = rv_short / rv_long

    vol_adj_ret = (log_ret / rv_short.replace(0.0, np.nan)).to_numpy()
    ema_fast_vadj = _ema_fn(vol_adj_ret, ema_fast_period)
    ema_slow_vadj = _ema_fn(vol_adj_ret, ema_slow_period)
    ema_cross_vadj = ema_fast_vadj - ema_slow_vadj

    feats = pd.DataFrame(
        {
            "slope": slope,
            "r2": r2,
            "rv_short": rv_short.to_numpy(),
            "rv_long": rv_long.to_numpy(),
            "rv_ratio": rv_ratio.to_numpy(),
            "ema_fast_vadj": ema_fast_vadj,
            "ema_slow_vadj": ema_slow_vadj,
            "ema_cross_vadj": ema_cross_vadj,
        },
        index=close.index,
    )
    feats["slope_x_rvratio"] = feats["slope"] * feats["rv_ratio"]
    feats["r2_x_rvratio"] = feats["r2"] * feats["rv_ratio"]
    feats["slope_x_rvshort"] = feats["slope"] * feats["rv_short"]
    feats["ema_cross_vadj_x_rvratio"] = feats["ema_cross_vadj"] * feats["rv_ratio"]
    feats["const"] = 1.0
    return feats.replace([np.inf, -np.inf], np.nan)


# =========================================================================
# Walk-forward fit / predict
# =========================================================================

def walk_forward_predictions(
    close: pd.Series,
    regression_window: int = 20,
    rv_short_window: int = 10,
    rv_long_window: int = 60,
    ema_fast_period: int = 9,
    ema_slow_period: int = 21,
    forward_k: int = 5,
    train_bars: int = 450,
    test_bars: int = 190,
    min_train_bars: int = 150,
    n_alphas: int = 60,
    cv: int = 5,
) -> pd.Series:
    """
    Out-of-sample predicted forward-`forward_k`-bar log return, one value per
    bar of `close` (must be gap-free -- see build_features), refit every
    `test_bars` bars on the trailing `train_bars` bars. Bars before the first
    fit (warmup) are NaN.
    """
    feats = build_features(close, regression_window, rv_short_window, rv_long_window, ema_fast_period, ema_slow_period)
    target = np.log(close.shift(-forward_k)) - np.log(close)

    valid = feats.notna().all(axis=1).to_numpy()
    target_valid = target.notna().to_numpy()
    n = len(feats)

    preds = pd.Series(np.nan, index=feats.index)

    cut = train_bars
    while cut < n:
        tr_start = max(0, cut - train_bars)
        tr_idx = np.arange(tr_start, cut)
        tr_idx = tr_idx[tr_idx + forward_k <= cut]  # outcome must be fully realized by `cut`
        tr_idx = tr_idx[valid[tr_idx] & target_valid[tr_idx]]

        te_end = min(n, cut + test_bars)
        te_idx = np.arange(cut, te_end)
        te_idx = te_idx[valid[te_idx]]

        if len(tr_idx) < min_train_bars or len(te_idx) == 0:
            cut += test_bars
            continue

        X_tr, y_tr = feats.iloc[tr_idx], target.iloc[tr_idx]
        predict_fn, _, _ = fit_lasso_cv(X_tr, y_tr, n_alphas=n_alphas, cv=cv)

        pred_te = predict_fn(feats.iloc[te_idx])
        # clamp to the training target range: extreme out-of-sample feature
        # levels otherwise extrapolate linearly to absurd magnitudes (same
        # discipline as metahar_wf_backtest.py's walk_forward()).
        pred_te = np.clip(pred_te, y_tr.min(), y_tr.max())
        preds.iloc[te_idx] = pred_te

        cut += test_bars

    return preds


def gating_threshold(pred_series: pd.Series, quantile: float = 0.7, min_periods: int = 50) -> pd.Series:
    """Causal (past-only) rolling quantile of |prediction| -- 'is this
    signal's predicted magnitude large relative to recent history', without
    using any prediction at or after the bar being gated to set the bar."""
    thr = pred_series.abs().expanding(min_periods=min_periods).quantile(quantile)
    return thr.shift(1)


# =========================================================================
# Orchestration: the entry point metafvg_ab_sweep.py actually calls
# =========================================================================

def compute_lasso_gate_series(
    htf_df: pd.DataFrame,
    regression_window: int = 20,
    rv_short_window: int = 10,
    rv_long_window: int = 60,
    ema_fast_period: int = 9,
    ema_slow_period: int = 21,
    forward_k: int = 5,
    train_bars: int = 450,
    test_bars: int = 190,
    min_train_bars: int = 150,
    n_alphas: int = 60,
    cv: int = 5,
    threshold_quantile: float = 0.7,
) -> tuple[pd.Series, pd.Series]:
    """
    Full pipeline for one symbol: densify htf_df's close (drop the real
    market-closed gap rows so rolling windows compute on genuinely
    consecutive bars), run the walk-forward Lasso fit/predict pass and the
    causal gating-threshold pass in that dense space, then reindex both
    result series back onto the *original* htf_df.index with a forward-fill
    -- gap bars simply carry forward the last known real prediction/threshold,
    the same hold-steady-across-gaps convention as the ema() fix in
    fastfinance.py. This is what metafvg_variants.py's trend_filter=="lasso"
    branch looks up via `avail_weekly.index[-1]`, which can itself land on a
    gap row.

    Returns (lasso_pred, lasso_threshold), both indexed like htf_df.
    """
    real_close = htf_df["close"].dropna()

    pred_dense = walk_forward_predictions(
        real_close,
        regression_window=regression_window, rv_short_window=rv_short_window, rv_long_window=rv_long_window,
        ema_fast_period=ema_fast_period, ema_slow_period=ema_slow_period,
        forward_k=forward_k, train_bars=train_bars, test_bars=test_bars,
        min_train_bars=min_train_bars, n_alphas=n_alphas, cv=cv,
    )
    thr_dense = gating_threshold(pred_dense, quantile=threshold_quantile)

    pred_full = pred_dense.reindex(htf_df.index, method="ffill")
    thr_full = thr_dense.reindex(htf_df.index, method="ffill")
    return pred_full, thr_full
