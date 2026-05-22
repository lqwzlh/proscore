"""Model evaluation: KS, AUC, PSI, and score-distribution table."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from proscore.utils._psi import psi_from_distributions


def evaluate(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    *,
    X_oot: pd.DataFrame | None = None,
    y_oot: pd.Series | None = None,
    features: list[str] | None = None,
    n_bins: int = 10,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Comprehensive model evaluation for a binary classifier.

    The target must be binary (two classes) and each dataset must contain
    examples of both classes.

    Parameters
    ----------
    model : statsmodels Logit result, sklearn classifier, or any object with
        ``predict_proba`` or ``predict`` returning probabilities.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target (1 = bad).
    X_test : pd.DataFrame, optional
        In-time test features.
    y_test : pd.Series, optional
        In-time test target.
    X_oot : pd.DataFrame, optional
        Out-of-time features.
    y_oot : pd.Series, optional
        Out-of-time target.
    features : list of str, optional
        Columns to use.  Defaults to all columns in *X_train*.
    n_bins : int
        Number of equal-frequency bins for the score-distribution table.
    threshold : float
        Probability threshold for accuracy and classification.

        .. warning::

           ``trn_acc`` / ``test_acc`` / ``oot_acc`` are computed at
           *threshold* (default 0.5) and may be misleading with imbalanced
           credit data.

    Returns
    -------
    dict
        Keys: ``trn_ks``, ``trn_auc``, ``trn_acc``, ``model_vars``, and
        conditionally ``test_ks``, ``test_auc``, ``test_acc``, ``psi``,
        ``ks_reduce``, ``ks_rel_gap``, ``oot_ks``, ``oot_auc``, ``oot_acc``,
        ``psi_oot``, ``score_table``.
    """
    feats = features or list(X_train.columns)

    prob_trn = _predict_proba(model, X_train[feats])
    yt = y_train.values

    trn_ks = _ks(prob_trn, yt)
    trn_auc = float(roc_auc_score(yt, prob_trn))
    pred_trn = (prob_trn >= threshold).astype(int)
    trn_acc = float(accuracy_score(yt, pred_trn))

    result: dict[str, Any] = {
        "trn_ks": round(trn_ks, 6),
        "trn_auc": round(trn_auc, 6),
        "trn_acc": round(trn_acc, 6),
        "model_vars": feats,
    }

    # ── test (in-time) ───────────────────────────────────────────────────────
    if X_test is not None and y_test is not None:
        prob_tst = _predict_proba(model, X_test[feats])
        ys = y_test.values

        result["test_ks"] = round(_ks(prob_tst, ys), 6)
        result["test_auc"] = round(float(roc_auc_score(ys, prob_tst)), 6)
        pred_tst = (prob_tst >= threshold).astype(int)
        result["test_acc"] = round(float(accuracy_score(ys, pred_tst)), 6)

        ks_reduce = trn_ks - result["test_ks"]
        result["ks_reduce"] = round(ks_reduce, 6)
        result["ks_rel_gap"] = round(abs(ks_reduce) / trn_ks if trn_ks > 0 else 0.0, 6)
        result["psi"] = round(_calc_psi(prob_trn, prob_tst, n_bins), 6)

        result["score_table"] = _score_distribution(
            prob_trn, prob_tst, yt, ys, n_bins,
        )
    else:
        result["score_table"] = _score_distribution_single(prob_trn, yt, n_bins)

    # ── oot (out-of-time) ────────────────────────────────────────────────────
    if X_oot is not None and y_oot is not None:
        prob_oot = _predict_proba(model, X_oot[feats])
        yo = y_oot.values

        result["oot_ks"] = round(_ks(prob_oot, yo), 6)
        result["oot_auc"] = round(float(roc_auc_score(yo, prob_oot)), 6)
        pred_oot = (prob_oot >= threshold).astype(int)
        result["oot_acc"] = round(float(accuracy_score(yo, pred_oot)), 6)
        result["psi_oot"] = round(_calc_psi(prob_trn, prob_oot, n_bins), 6)

    return result


def evaluate_by_period(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    periods: dict[str, tuple[pd.DataFrame, pd.Series]] | None = None,
    df: pd.DataFrame | None = None,
    time_col: str | None = None,
    target: str | None = None,
    features: list[str] | None = None,
    n_bins: int = 10,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluate model performance on multiple out-of-time periods (e.g. by year).

    Either pass *periods* explicitly as ``{label: (X, y), ...}``, or pass a
    single *df* with *time_col* / *target* — periods are built by calendar
    year of *time_col*.

    Score PSI for each period is computed vs the **training** score
    distribution (same binning as :func:`evaluate`).

    Parameters
    ----------
    model : fitted classifier with ``predict`` / ``predict_proba``
    X_train, y_train : training features and binary target (1 = bad)
    periods : dict, optional
        Pre-split OOT slices, e.g. ``{"2022": (X_22, y_22), "2023": (X_23, y_23)}``.
    df, time_col, target : optional
        Alternative: one DataFrame; grouped by ``time_col.dt.year``.
    features : list of str, optional
        Model feature columns.
    n_bins : int
        Bins for score PSI (train vs period).
    threshold : float
        Classification threshold for accuracy.

    Returns
    -------
    pd.DataFrame
        Columns: ``period | n | bad_rate | ks | auc | acc | psi_score |
        ks_decay | auc_decay`` (decay vs training KS/AUC).
    """
    feats = features or list(X_train.columns)
    prob_trn = _predict_proba(model, X_train[feats])
    yt = np.asarray(y_train).ravel()
    trn_ks = _ks(prob_trn, yt)
    trn_auc = float(roc_auc_score(yt, prob_trn))

    if periods is None:
        if df is None or time_col is None or target is None:
            raise ValueError(
                "Provide either periods={label: (X, y), ...} or "
                "df + time_col + target."
            )
        periods = _split_df_by_year(df, time_col, target, feats)

    rows: list[dict[str, Any]] = []
    for label in sorted(periods.keys()):
        X_p, y_p = periods[label]
        prob = _predict_proba(model, X_p[feats])
        y_arr = np.asarray(y_p).ravel()
        n = len(y_arr)
        ks_p = _ks(prob, y_arr)
        auc_p = float(roc_auc_score(y_arr, prob))
        acc_p = float(accuracy_score(y_arr, (prob >= threshold).astype(int)))
        psi_p = _calc_psi(prob_trn, prob, n_bins)
        ks_decay = (trn_ks - ks_p) / trn_ks if trn_ks > 0 else np.nan
        auc_decay = (trn_auc - auc_p) / trn_auc if trn_auc > 0.5 else np.nan
        rows.append({
            "period": label,
            "n": int(n),
            "bad_rate": round(float(y_arr.mean()), 6),
            "ks": round(ks_p, 6),
            "auc": round(auc_p, 6),
            "acc": round(acc_p, 6),
            "psi_score": round(psi_p, 6),
            "ks_decay": round(ks_decay, 6) if not np.isnan(ks_decay) else np.nan,
            "auc_decay": round(auc_decay, 6) if not np.isnan(auc_decay) else np.nan,
        })

    return pd.DataFrame(rows)


def _split_df_by_year(
    df: pd.DataFrame,
    time_col: str,
    target: str,
    features: list[str],
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Group *df* into yearly periods for :func:`evaluate_by_period`."""
    if time_col not in df.columns:
        raise KeyError(f"time_col {time_col!r} not in DataFrame")
    if target not in df.columns:
        raise KeyError(f"target {target!r} not in DataFrame")
    out: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    years = sorted(df[time_col].dropna().dt.year.unique())
    for year in years:
        mask = df[time_col].dt.year == year
        sub = df.loc[mask]
        out[str(int(year))] = (sub[features], sub[target])
    return out


# ── internal helpers ───────────────────────────────────────────────────────


def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Extract probability of the positive class (index 1)."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float).ravel()
    # statsmodels Logit / other: needs constant, predict returns probabilities
    X_aug = sm.add_constant(X, has_constant="add")
    return np.asarray(model.predict(X_aug), dtype=float).ravel()


def _ks(prob: np.ndarray, y: np.ndarray) -> float:
    """KS statistic from predicted probabilities."""
    # KS = max(|TPR - FPR|) via sklearn.
    # For manual CDF-based KS, see _stepwise._ks_from_probs.
    fpr, tpr, _ = roc_curve(y, prob)
    return float(np.abs(tpr - fpr).max())


def _quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Return bin edges for equal-frequency binning (first=-inf, last=+inf)."""
    bins = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)
    bins[0] = -np.inf
    bins[-1] = np.inf
    return bins


def _calc_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI between two probability distributions (expected = train, actual = test)."""
    bins = _quantile_bins(expected, n_bins)
    e_bin = pd.cut(expected, bins=bins, labels=False, right=True)
    a_bin = pd.cut(actual, bins=bins, labels=False, right=True)

    e_cnt = pd.Series(e_bin).value_counts(sort=False)
    a_cnt = pd.Series(a_bin).value_counts(sort=False)
    all_idx = e_cnt.index.union(a_cnt.index)
    e_arr = e_cnt.reindex(all_idx, fill_value=0).to_numpy(dtype=float)
    a_arr = a_cnt.reindex(all_idx, fill_value=0).to_numpy(dtype=float)

    return psi_from_distributions(e_arr, a_arr)


def _score_distribution_single(
    prob_trn: np.ndarray,
    y_trn: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build per-bin bad-rate table for train data only (no test set).

    Delegates to :func:`_score_distribution` with ``prob_tst=None,
    y_tst=None``, which falls back to using the train data as its own
    reference.
    """
    return _score_distribution(prob_trn, None, y_trn, None, n_bins)


def _score_distribution(
    prob_trn: np.ndarray,
    prob_tst: np.ndarray | None,
    y_trn: np.ndarray,
    y_tst: np.ndarray | None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build per-bin bad-rate and cumulative distribution table.

    When *prob_tst* / *y_tst* are ``None`` (single-dataset usage) the
    function falls back to *prob_trn* / *y_trn*, identical to the
    behaviour of ``_score_distribution_single``.
    """
    prob = prob_tst if prob_tst is not None else prob_trn
    y_arr = y_tst if y_tst is not None else y_trn

    bins = _quantile_bins(prob_trn, n_bins)
    labels = [f"({bins[i]:.4f}, {bins[i+1]:.4f}]" for i in range(len(bins) - 1)]
    try:
        data_bin = pd.cut(prob, bins=bins, labels=labels, right=True)
    except ValueError:
        data_bin = pd.cut(prob, bins=bins, labels=labels, right=True, ordered=False)

    rows = []
    for label in labels:
        mask = data_bin == label
        count = mask.sum()
        if count == 0:
            continue
        bad = y_arr[mask].sum()
        good = count - bad
        rows.append({
            "bin": label,
            "count": int(count),
            "good": int(good),
            "bad": int(bad),
            "bad_rate": round(bad / count, 4) if count > 0 else 0.0,
            "pct": round(count / len(y_arr) * 100, 2),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["cum_good_pct"] = (df["good"].cumsum() / df["good"].sum() * 100).round(2)
        df["cum_bad_pct"] = (df["bad"].cumsum() / df["bad"].sum() * 100).round(2)
        df["ks"] = (df["cum_bad_pct"] - df["cum_good_pct"]).abs().round(2)
    return df
