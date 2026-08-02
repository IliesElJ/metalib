"""
Walk-forward, shallow decision-tree trend-continuation gate -- a third
interpretable alternative to metafvg_variants.py's Regression Gate and
metafvg_lasso_gate.py's Lasso Trend Gate, using the *same* feature set as the
Lasso gate (build_features is reused directly, not reimplemented) so any
performance difference between the two isolates model choice rather than
conflating it with feature choice.

Unlike Lasso, CART-style tree building is pure sorting + impurity-reduction
arithmetic -- no matrix algebra -- so sklearn's DecisionTreeClassifier is
safe in this env despite the documented BLAS/LAPACK crash on any GEMM
(confirmed empirically before writing this module, not assumed).

Framed as classification, not regression: predict whether the forward-K-bar
return is positive or negative, using predict_proba as the confidence score
that plays the same role as the Lasso gate's |predicted return| magnitude --
gate on (predicted class matches trade direction) AND (predicted probability
clears a causal rolling-quantile threshold of its own past). "Short" tree =
max_depth is small (default 3) so the fitted rule set stays literally
readable as a short if/else chain, not just "regularized" in the abstract.

Same no-lookahead discipline as the Lasso gate:
  - Features at bar t are trailing-window rolling stats (t and earlier only).
  - A training row at bar t is only used once bar t+K has actually occurred
    by the fit's own cutoff bar.
  - The gating threshold is a causal (past-only) rolling quantile of the
    prediction confidence series, shifted so bar t's threshold never uses a
    prediction generated at or after t.
  - Densify-then-reindex-ffill for the ~30%-NaN-gappy 4h HTF series, same as
    the Lasso gate and the earlier ema() fix in fastfinance.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from metafvg_lasso_gate import build_features, gating_threshold


def walk_forward_predictions(
    close: pd.Series,
    regression_window: int = 20,
    rv_short_window: int = 10,
    rv_long_window: int = 60,
    forward_k: int = 5,
    train_bars: int = 450,
    test_bars: int = 190,
    min_train_bars: int = 150,
    max_depth: int = 3,
    min_samples_leaf: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """
    Out-of-sample (predicted_sign, predicted_confidence) for each bar of
    `close` (must be gap-free -- see build_features), refit every
    `test_bars` bars on the trailing `train_bars` bars, same walk-forward
    cutoff discipline as metafvg_lasso_gate.walk_forward_predictions.

    predicted_sign: +1.0 / -1.0 (direction of predicted forward-K-bar return)
    predicted_confidence: predict_proba of the predicted class, in [0.5, 1.0]
    """
    feats = build_features(close, regression_window, rv_short_window, rv_long_window)
    fwd_ret = np.log(close.shift(-forward_k)) - np.log(close)
    target = (fwd_ret > 0).astype(float)  # 1.0 = up, 0.0 = down

    feat_cols = [c for c in feats.columns if c != "const"]  # a constant column is useless to a tree split
    valid = feats[feat_cols].notna().all(axis=1).to_numpy()
    target_valid = target.notna().to_numpy()
    n = len(feats)

    sign = pd.Series(np.nan, index=feats.index)
    conf = pd.Series(np.nan, index=feats.index)

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

        y_tr = target.iloc[tr_idx]
        if y_tr.nunique() < 2:
            # a training window that's all-up or all-down can't fit a
            # meaningful split -- skip this fold rather than fit a
            # degenerate always-one-class tree.
            cut += test_bars
            continue

        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0)
        clf.fit(feats.iloc[tr_idx][feat_cols], y_tr)

        proba = clf.predict_proba(feats.iloc[te_idx][feat_cols])
        up_col = list(clf.classes_).index(1.0)
        p_up = proba[:, up_col]

        sign.iloc[te_idx] = np.where(p_up >= 0.5, 1.0, -1.0)
        conf.iloc[te_idx] = np.where(p_up >= 0.5, p_up, 1.0 - p_up)

        cut += test_bars

    return sign, conf


def compute_tree_gate_series(
    htf_df: pd.DataFrame,
    regression_window: int = 20,
    rv_short_window: int = 10,
    rv_long_window: int = 60,
    forward_k: int = 5,
    train_bars: int = 450,
    test_bars: int = 190,
    min_train_bars: int = 150,
    max_depth: int = 3,
    min_samples_leaf: int = 20,
    threshold_quantile: float = 0.7,
) -> tuple[pd.Series, pd.Series]:
    """
    Full pipeline for one symbol, mirroring
    metafvg_lasso_gate.compute_lasso_gate_series: densify htf_df's close,
    walk-forward-fit/predict in that dense space, causal-quantile-threshold
    the confidence series, then reindex both back onto htf_df's original
    (gap-including) index with ffill.

    Returns (tree_pred, tree_threshold), both indexed like htf_df.
    tree_pred holds the SIGNED prediction (+/-1 * confidence) so
    metafvg_variants.py's gate check can reuse the identical
    "(pred > 0) == is_bullish and abs(pred) >= thr" logic as the Lasso gate.
    """
    real_close = htf_df["close"].dropna()

    sign_dense, conf_dense = walk_forward_predictions(
        real_close, regression_window, rv_short_window, rv_long_window,
        forward_k, train_bars, test_bars, min_train_bars, max_depth, min_samples_leaf,
    )
    pred_dense = sign_dense * conf_dense
    thr_dense = gating_threshold(pred_dense, quantile=threshold_quantile)

    pred_full = pred_dense.reindex(htf_df.index, method="ffill")
    thr_full = thr_dense.reindex(htf_df.index, method="ffill")
    return pred_full, thr_full
