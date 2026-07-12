"""
Walk-forward validation of the MetaHAR vol-prediction concept.

Setup (per discussion 2026-07-12):
- M15 bars from MT5 (FPTrading demo), prod MIX basket with US500 instead of US500.cash.
- Vol horizon upscaled: predict the next-4h-bar change in log realized variance,
  where RV is the variance of consecutive 15-min log returns over a 16-bar (4h)
  window. This fixes the log_var definition in indicators.py (which measures
  log-price dispersion, not return variance).
- Features mirror MetaHAR.retrieve_indicators bar-for-bar with factors expressed
  in bars: short=16 (4h), long=256 (~2.7d) — same 1:16 ratio as prod (15/240 min).
- EWMs/rollings are computed once over full history (past-only, no lookahead),
  instead of per training window as prod does; only difference is EWM warmup tails.
- Walk-forward: 66-day train window (prod TRAINING_PERIOD_DAYS), refit every 28
  days, predict strictly out-of-sample until the next refit.
- Model: Lasso with 5-fold CV alpha selection on standardized features, like
  prod build_lasso_cv — but implemented as numba coordinate descent with
  explicit loops. The BLAS shipped in this env crashes the process on ANY
  matrix-matrix multiply (np.dot GEMM), which kills sklearn LassoCV; same
  root cause as the known manual-OLS workaround. Deviation from prod: the
  alpha grid is data-driven (alpha_max down 4 decades, 100 points) instead of
  logspace(-10,10,1000); CV folds are contiguous KFold like sklearn's default.

Baselines the model must beat for "actual value":
- zero forecast (random-walk log-vol): OOS R2 benchmark
- AR(1) on the vol diff, coefficient fit manually on train (manual OLS: BLAS in
  this env crashes on linalg paths — do not replace with statsmodels/scipy)
- naive mean-reversion sign: predict sign(y[t]) = -sign(last observed diff)

Usage (adonys env, PYTHONPATH=repo root):
  python metahar_wf_backtest.py pull   # fetch + cache M15 data from MT5
  python metahar_wf_backtest.py run    # walk-forward + metrics
"""

import os
import sys

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSD", "EURUSD", "XAUUSD", "US500", "GBPUSD"]
SHORT = 16    # bars of M15 -> 4h
LONG = 256    # bars -> ~2.7 days, keeps prod short:long = 1:16
TRAIN_DAYS = 66
TEST_DAYS = 28
MIN_TRAIN_SAMPLES = 150

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO, "metalib", "data", "mix_m15")
OUT_DIR = os.path.join(REPO, "metalib", "notebooks", "metahar_wf")


def pull_data():
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    os.makedirs(DATA_DIR, exist_ok=True)
    for sym in SYMBOLS:
        mt5.symbol_select(sym, True)
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 80000)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"no M15 data for {sym}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        path = os.path.join(DATA_DIR, f"{sym}_m15.csv")
        df.to_csv(path)
        print(f"{sym}: {len(df)} bars {df.index[0]} -> {df.index[-1]} -> {path}")
    mt5.shutdown()


def load_closes():
    closes = {}
    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{sym}_m15.csv")
        df = pd.read_csv(path, index_col="time", parse_dates=True)
        closes[sym] = df["close"]
    out = pd.DataFrame(closes).dropna()
    print(f"aligned closes: {len(out)} bars {out.index[0]} -> {out.index[-1]}")
    return out


def build_features(closes):
    """Mirror of MetaHAR.retrieve_indicators, bar-based, with fixed log_var
    (variance of consecutive log returns instead of log-price dispersion)."""
    ret = np.log(closes).diff().dropna()
    sq = ret**2
    down = ret.where(ret < 0, 0.0) ** 2
    up = ret.where(ret > 0, 0.0) ** 2

    feats = pd.concat(
        [
            np.log(ret.rolling(LONG).var(ddof=0)).add_prefix("long_scale_std_"),
            np.log(ret.rolling(SHORT).var(ddof=0)).add_prefix("short_scale_std_"),
            ret.ewm(halflife=SHORT).mean().add_prefix("trend_ewm_short_factor_"),
            ret.ewm(halflife=LONG).mean().add_prefix("trend_ewm_long_factor_"),
            sq.ewm(halflife=SHORT).mean().add_prefix("sq_ret_ewm_short_factor_"),
            sq.ewm(halflife=LONG).mean().add_prefix("sq_ret_ewm_long_factor_"),
            down.ewm(halflife=SHORT).mean().add_prefix("downside_ewm_short_factor_"),
            down.ewm(halflife=LONG).mean().add_prefix("downside_ewm_long_factor_"),
            up.ewm(halflife=SHORT).mean().add_prefix("upside_ewm_short_factor_"),
            up.ewm(halflife=LONG).mean().add_prefix("upside_ewm_long_factor_"),
        ],
        axis=1,
    )
    feats["const"] = 1.0

    resampled = (
        feats.resample("4h", closed="right", label="right")
        .last()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    print(f"4h feature samples: {len(resampled)} {resampled.index[0]} -> {resampled.index[-1]}")
    return resampled


from numba import njit


@njit(cache=True)
def _lasso_path_cd(X, y, alphas, max_iter, tol):
    """Coordinate-descent Lasso path, sklearn objective
    1/(2n)*||y-Xw||^2 + alpha*||w||_1, warm-started along descending alphas.
    Explicit loops only — numba's np.dot lowers to the broken BLAS."""
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


def fit_lasso_cv(X_df, y_ser, n_alphas=100, cv=5, max_iter=500, tol=1e-7):
    """BLAS-free stand-in for build_lasso_cv: standardize, pick alpha by
    contiguous K-fold CV MSE, refit on full training set.
    Returns a predict(X_df) callable."""
    X = X_df.to_numpy(dtype=np.float64)
    y = y_ser.to_numpy(dtype=np.float64)
    n = len(y)

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0  # constant cols (e.g. 'const') become all-zero, coef 0
    Xs = (X - mu) / sd
    ym = y.mean()
    yc = y - ym

    # data-driven alpha grid, descending for warm starts
    alpha_max = max(np.abs((Xs * yc[:, None]).sum(axis=0)).max() / n, 1e-12)
    alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_max * 1e-4), n_alphas)

    fold_edges = np.linspace(0, n, cv + 1).astype(int)
    mse = np.zeros(len(alphas))
    for k in range(cv):
        lo, hi = fold_edges[k], fold_edges[k + 1]
        tr = np.ones(n, dtype=bool)
        tr[lo:hi] = False
        coefs = _lasso_path_cd(Xs[tr], yc[tr], alphas, max_iter, tol)
        # (X * w).sum(1) is elementwise + reduce: no GEMM
        for a in range(len(alphas)):
            resid = yc[lo:hi] - (Xs[lo:hi] * coefs[a]).sum(axis=1)
            mse[a] += (resid**2).mean()
    best = int(np.argmin(mse))

    coefs = _lasso_path_cd(Xs, yc, alphas[: best + 1], max_iter, tol)
    w = coefs[-1]

    def predict(X_new_df):
        Xn = (X_new_df.to_numpy(dtype=np.float64) - mu) / sd
        return ym + (Xn * w).sum(axis=1)

    return predict, alphas[best], w


def build_seasonal_dummies(index):
    """Deterministic calendar dummies describing the NEXT 4h bar (the one whose
    vol change is the target). All computable from the clock at time t — no
    economic-calendar feed, no leakage. Last row is NaN (no next bar) and gets
    dropped with the target anyway."""
    nxt = index.to_series().shift(-1)
    hour = nxt.dt.hour
    dow = nxt.dt.dayofweek
    day = nxt.dt.day
    month = nxt.dt.month

    d = pd.DataFrame(index=index)
    # bar-of-day (right-edge labels 00/04/08/12/16/20; drop 00 as base)
    for h in [4, 8, 12, 16, 20]:
        d[f"dum_next_h{h:02d}"] = (hour == h).astype(float)
    # day-of-week (drop Wednesday as base)
    for dw, name in [(0, "mon"), (1, "tue"), (3, "thu"), (4, "fri")]:
        d[f"dum_next_{name}"] = (dow == dw).astype(float)
    # NFP: first Friday of month, bar covering 12:00-16:00 UTC (8:30 ET print)
    d["dum_next_nfp"] = ((dow == 4) & (day <= 7) & (hour == 16)).astype(float)
    # US-morning bar on any day (contains 8:30 ET macro slot year-round)
    d["dum_next_us_morning"] = (hour == 16).astype(float)
    # OpEx: third Friday
    d["dum_next_opex"] = ((dow == 4) & (day >= 15) & (day <= 21)).astype(float)
    # month-end (last 2 calendar days) and quarter-end
    month_end = (nxt.dt.days_in_month - day) <= 1
    d["dum_next_month_end"] = month_end.astype(float)
    d["dum_next_quarter_end"] = (month_end & month.isin([3, 6, 9, 12])).astype(float)
    # holiday season
    d["dum_next_holidays"] = (
        ((month == 12) & (day >= 24)) | ((month == 1) & (day <= 2))
    ).astype(float)
    # first bar after a market gap (weekend): next label > 4h away
    d["dum_next_gap_open"] = ((nxt - index.to_series()) > pd.Timedelta(hours=4)).astype(float)
    return d


def welch_t(a, b):
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    return (a.mean() - b.mean()) / np.sqrt(va / na + vb / nb)


def event_study():
    """Per-dummy, per-symbol: mean target on event vs non-event bars, Welch t."""
    closes = load_closes()
    feats = build_features(closes)
    dums = build_seasonal_dummies(feats.index)
    rows = []
    for sym in SYMBOLS:
        x = feats[f"short_scale_std_{sym}"]
        y = (x.shift(-1) - x).dropna()
        for col in dums.columns:
            dcol = dums[col].reindex(y.index)
            on, off = y[dcol == 1], y[dcol == 0]
            rows.append(
                {
                    "dummy": col,
                    "symbol": sym,
                    "n_event": len(on),
                    "mean_on": on.mean(),
                    "mean_off": off.mean(),
                    "welch_t": welch_t(on, off),
                }
            )
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "event_study.csv"), index=False)
    pivot = df.pivot(index="dummy", columns="symbol", values="welch_t")
    print("\nWelch t-stats of next-4h vol change, event vs non-event bars:")
    print(pivot.round(1).to_string())
    return df


def manual_ar1(y_tr, x_tr, x_te):
    """Scalar OLS y = a + b*x by hand (BLAS-safe)."""
    xm, ym = x_tr.mean(), y_tr.mean()
    dx = x_tr - xm
    denom = (dx**2).sum()
    b = ((dx * (y_tr - ym)).sum() / denom) if denom > 0 else 0.0
    a = ym - b * xm
    return a + b * x_te


def spearman_ic(a, b):
    # manual pearson on ranks: np.corrcoef calls GEMM (crashes in this env)
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    da, db = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((da**2).sum() * (db**2).sum())
    return (da * db).sum() / denom if denom > 0 else np.nan


def walk_forward(feats, target_sym):
    x = feats[f"short_scale_std_{target_sym}"]
    y = (x.shift(-1) - x).dropna()          # prod target: shift(-1).diff()
    last_diff = x.diff()                     # known at t, for baselines
    idx = feats.index.intersection(y.index).intersection(last_diff.dropna().index)
    X, y, last_diff, x = feats.loc[idx], y.loc[idx], last_diff.loc[idx], x.loc[idx]

    rows = []
    fit_date = X.index[0] + pd.Timedelta(days=TRAIN_DAYS)
    fold = 0
    while fit_date < X.index[-1]:
        tr = (X.index > fit_date - pd.Timedelta(days=TRAIN_DAYS)) & (X.index <= fit_date)
        te = (X.index > fit_date) & (X.index <= fit_date + pd.Timedelta(days=TEST_DAYS))
        fit_date += pd.Timedelta(days=TEST_DAYS)
        if tr.sum() < MIN_TRAIN_SAMPLES or te.sum() == 0:
            continue
        predict, _, _ = fit_lasso_cv(X[tr], y[tr])
        # clamp to the target range seen in training: extreme feature levels
        # (e.g. XAUUSD Apr-2025 vol shock) otherwise extrapolate linearly to
        # absurd magnitudes (predicted log-vol changes of -11 vs realized +/-2)
        pred = np.clip(predict(X[te]), y[tr].min(), y[tr].max())
        pred_ar1 = manual_ar1(y[tr], last_diff[tr], last_diff[te]).values
        # tougher baseline: next diff regressed on current vol LEVEL
        # (mean reversion of measured RV is the dominant mechanical signal)
        pred_level = manual_ar1(y[tr], x[tr], x[te]).values
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": X.index[te],
                    "fold": fold,
                    "y": y[te].values,
                    "pred": pred,
                    "pred_ar1": pred_ar1,
                    "pred_level": pred_level,
                    "last_diff": last_diff[te].values,
                }
            )
        )
        fold += 1
    return pd.concat(rows, ignore_index=True)


def evaluate(preds, sym):
    y = preds["y"].values
    p = preds["pred"].values
    ar1 = preds["pred_ar1"].values
    lvl = preds["pred_level"].values
    naive_sign = -np.sign(preds["last_diff"].values)

    ss_tot = (y**2).sum()  # zero-forecast benchmark
    r2 = 1 - ((y - p) ** 2).sum() / ss_tot
    r2_ar1 = 1 - ((y - ar1) ** 2).sum() / ss_tot
    r2_level = 1 - ((y - lvl) ** 2).sum() / ss_tot

    hit = (np.sign(p) == np.sign(y)).mean()
    hit_ar1 = (np.sign(ar1) == np.sign(y)).mean()
    hit_level = (np.sign(lvl) == np.sign(y)).mean()
    hit_naive = (naive_sign == np.sign(y)).mean()

    # per-fold model-vs-level-baseline hit diff, paired t
    def fold_hits_lvl(d):
        s = np.sign(d["y"].values)
        return pd.Series(
            {
                "m": (np.sign(d["pred"].values) == s).mean(),
                "l": (np.sign(d["pred_level"].values) == s).mean(),
            }
        )

    fhl = preds.groupby("fold").apply(fold_hits_lvl)
    dl = fhl["m"] - fhl["l"]
    hit_vs_level_t = dl.mean() / (dl.std(ddof=1) / np.sqrt(len(dl)))

    q = np.quantile(np.abs(p), 0.75)
    conv = np.abs(p) >= q
    hit_conv = (np.sign(p[conv]) == np.sign(y[conv])).mean()

    # per-fold IC for a t-stat that respects serial correlation across time
    fold_ics = preds.groupby("fold").apply(
        lambda d: spearman_ic(d["pred"], d["y"]) if len(d) > 10 else np.nan
    )
    fold_ics = fold_ics.dropna()
    ic_mean = fold_ics.mean()
    ic_t = ic_mean / (fold_ics.std(ddof=1) / np.sqrt(len(fold_ics)))

    # per-fold model-vs-naive hit-rate diff, paired t
    def fold_hits(d):
        s = np.sign(d["y"].values)
        return pd.Series(
            {
                "m": (np.sign(d["pred"].values) == s).mean(),
                "n": (-np.sign(d["last_diff"].values) == s).mean(),
            }
        )

    fh = preds.groupby("fold").apply(fold_hits)
    diff = fh["m"] - fh["n"]
    hit_diff_t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))

    return {
        "symbol": sym,
        "n_preds": len(y),
        "n_folds": preds["fold"].nunique(),
        "oos_r2": r2,
        "oos_r2_ar1": r2_ar1,
        "oos_r2_level": r2_level,
        "hit": hit,
        "hit_naive_mr": hit_naive,
        "hit_ar1": hit_ar1,
        "hit_level": hit_level,
        "hit_vs_level_tstat": hit_vs_level_t,
        "hit_top25pct_conviction": hit_conv,
        "ic_spearman_mean": ic_mean,
        "ic_tstat": ic_t,
        "hit_minus_naive_tstat": hit_diff_t,
    }


def run(with_dummies=False, own_only=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    closes = load_closes()
    feats = build_features(closes)
    suffix = ""
    if with_dummies:
        feats = pd.concat([feats, build_seasonal_dummies(feats.index)], axis=1).dropna()
        suffix = "_dummies"
        print(f"with seasonal dummies: {feats.shape[1]} features")
    if own_only:
        suffix += "_own"

    summary = []
    for sym in SYMBOLS:
        print(f"===== walk-forward: {sym} =====")
        f = feats
        if own_only:
            # ablation: drop all cross-asset columns, keep target symbol's
            # features + calendar dummies + const
            keep = [
                c for c in feats.columns
                if c.endswith(f"_{sym}") or c.startswith("dum_") or c == "const"
            ]
            f = feats[keep]
            print(f"{sym}: own-only feature set = {len(keep)} columns")
        preds = walk_forward(f, sym)
        preds.to_csv(os.path.join(OUT_DIR, f"predictions_{sym}{suffix}.csv"), index=False)
        m = evaluate(preds, sym)
        summary.append(m)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})

    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(OUT_DIR, f"summary{suffix}.csv"), index=False)
    print("\n===== SUMMARY =====")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "pull":
        pull_data()
    elif cmd == "run":
        run()
    elif cmd == "run-dummies":
        run(with_dummies=True)
    elif cmd == "run-own":
        run(with_dummies=True, own_only=True)
    elif cmd == "events":
        event_study()
    else:
        raise SystemExit(f"unknown command: {cmd}")
