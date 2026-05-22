"""Variable quality assessment: IV, single-variable AUC, single-variable KS, PSI."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from proscore.utils import is_categorical, require_unique_column_labels

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

_SUPPORTED_ESTIMATORS = {"dt", "tree", "decisiontree"}
if _XGB_AVAILABLE:
    _SUPPORTED_ESTIMATORS |= {"xgb", "xgboost"}
if _LGB_AVAILABLE:
    _SUPPORTED_ESTIMATORS |= {"lgb", "lightgbm"}

# ── Estimator resolution ───────────────────────────────────────────────────


def _resolve_estimator(estimator: str):
    """Return a fresh classifier instance for the given estimator name."""
    key = estimator.lower().replace("-", "").replace("_", "")
    if key in ("dt", "tree", "decisiontree"):
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
    if key in ("xgb", "xgboost"):
        if not _XGB_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )
        return XGBClassifier(
            n_estimators=50, max_depth=3, min_child_weight=5, random_state=42, verbosity=0
        )
    if key in ("lgb", "lightgbm"):
        if not _LGB_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )
        return LGBMClassifier(
            n_estimators=50, max_depth=3, min_child_samples=5, random_state=42, verbose=-1
        )
    raise ValueError(
        f"Unknown estimator: {estimator}. "
        f"Supported: {sorted(_SUPPORTED_ESTIMATORS)}"
    )


def _validate_estimator(estimator: str) -> None:
    """Raise if *estimator* is unknown or optional dependency is missing."""
    key = estimator.lower().replace("-", "").replace("_", "")
    if key not in _SUPPORTED_ESTIMATORS:
        raise ValueError(
            f"Unknown estimator: {estimator}. "
            f"Supported: {sorted(_SUPPORTED_ESTIMATORS)}"
        )
    if key in ("xgb", "xgboost") and not _XGB_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    if key in ("lgb", "lightgbm") and not _LGB_AVAILABLE:
        raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")


# ── Public API ─────────────────────────────────────────────────────────────


def quality(
    df: pd.DataFrame,
    target: str,
    df_test: pd.DataFrame | None = None,
    skip_columns: list[str] | None = None,
    max_categories: int = 20,
    estimator: str = "dt",
    compute_ks: bool = True,
    errors: Literal["raise", "ignore"] = "raise",
    warn_insample_bias: bool = False,
) -> pd.DataFrame:
    """
    Compute quality metrics for each candidate feature.

    Single-variable AUC and KS are computed on the **same** rows used to fit
    the estimator (in-sample scores). They are useful for coarse screening
    but are **not** unbiased estimates of out-of-sample performance.

    Parameters
    ----------
    df : pd.DataFrame
        Training data (column labels must be unique).
    target : str
        Target column name (binary, 0/1).
    df_test : pd.DataFrame, optional
        Test/OOT data (column labels must be unique when provided).
        When set, ``psi`` compares each feature's marginal bin distribution on
        *df_test* to *df* (reference). Bin definitions match the IV logic
        (quantile bins on train for continuous features).
    skip_columns : list of str, optional
        Columns to skip (e.g. primary key, month).
    max_categories : int
        Threshold for categorical detection.
    estimator : str
        Estimator used for single-variable AUC / KS computation.
        One of ``"dt"`` (default), ``"xgb"``, ``"lgb"``.
    compute_ks : bool
        Whether to compute single-variable KS (adds run time).
    errors : {"raise", "ignore"}
        If ``"raise"`` (default), failures in AUC/KS fitting propagate.
        If ``"ignore"``, those metrics become ``None`` when computation fails.
    warn_insample_bias : bool
        If ``True``, emit a ``UserWarning`` that single-variable AUC/KS are
        in-sample and optimistic. Default ``False`` (see docstring above).

    Returns
    -------
    pd.DataFrame
        Columns: ``variable | dtype | iv | auc | ks | psi | missing_pct | n_unique``
        (``psi`` is ``None`` when *df_test* is omitted).
    """
    require_unique_column_labels(df)
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in DataFrame")
    if df_test is not None:
        require_unique_column_labels(df_test)
    _validate_estimator(estimator)
    if warn_insample_bias:
        warnings.warn(
            "Single-variable AUC and KS are computed in-sample and are biased "
            "(optimistic) estimates. Use only for coarse screening, not for "
            "final feature selection.",
            UserWarning,
            stacklevel=2,
        )

    skip = set(skip_columns or [])
    y = df[target].values

    if df_test is not None:
        needed = [c for c in df.columns if c != target and c not in skip]
        missing_cols = [c for c in needed if c not in df_test.columns]
        if missing_cols:
            raise KeyError(
                "df_test is missing columns present in df (excluding target/skip): "
                f"{missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}"
            )

    rows: list[dict] = []

    for col in df.columns:
        if col == target or col in skip:
            continue

        series = df[col]
        cat = is_categorical(series, max_categories)

        # IV
        iv_val = _calc_iv(series, y, cat)

        # AUC / KS (numeric features only)
        auc_val = None
        ks_val = None
        if not cat:
            prob, y_enc = _fit_predict_single_var(series, y, estimator, errors=errors)
            if prob is not None:
                auc_val = float(roc_auc_score(y_enc, prob))
                if compute_ks:
                    ks_val = _calc_ks_from_probs(prob, y_enc)

        psi_val = None
        if df_test is not None:
            psi_val = _calc_psi(series, df_test[col], cat)

        n_nonnull = series.count()
        missing_pct = round((len(series) - n_nonnull) / len(series) * 100, 2)

        rows.append(
            {
                "variable": col,
                "dtype": "categorical" if cat else "numeric",
                "iv": round(iv_val, 6),
                "auc": round(auc_val, 6) if auc_val is not None else None,
                "ks": round(ks_val, 6) if ks_val is not None else None,
                "psi": round(psi_val, 6) if psi_val is not None else None,
                "missing_pct": missing_pct,
                "n_unique": int(series.nunique()),
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("iv", ascending=False).reset_index(drop=True)
    return result


# ── Target encoding ────────────────────────────────────────────────────────


def _encode_binary_target(y: np.ndarray) -> np.ndarray | None:
    """Map *y* to ``{0, 1}`` if exactly two distinct values; else ``None``."""
    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y).ravel())
    if len(le.classes_) != 2:
        return None
    return y_enc


# ── Shared: fit single-variable model, return predictions ──────────────────


def _fit_predict_single_var(
    series: pd.Series,
    y: np.ndarray,
    estimator: str = "dt",
    *,
    errors: Literal["raise", "ignore"] = "raise",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Fit a single-variable model and return ``(probabilities, encoded_target)``.

    Returns ``(None, None)`` when the feature is unsuitable (too few samples,
    constant, or more than two target classes).
    """
    temp = pd.DataFrame({"var": series, "target": y}).dropna()
    if len(temp) < 10 or temp["var"].nunique() <= 1:
        return None, None

    X = temp["var"].values.reshape(-1, 1).astype(float)
    y_sub = temp["target"].values
    y_enc = _encode_binary_target(y_sub)
    if y_enc is None:
        return None, None

    def _run() -> tuple[np.ndarray, np.ndarray]:
        clf = _resolve_estimator(estimator)
        clf.fit(X, y_enc)
        return clf.predict_proba(X)[:, 1], y_enc

    if errors == "ignore":
        try:
            return _run()
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            return None, None
    return _run()


# ── KS from cached predictions ─────────────────────────────────────────────


def _calc_ks_from_probs(prob: np.ndarray, y_enc: np.ndarray) -> float | None:
    """Kolmogorov-Smirnov statistic from predicted probabilities and binary target."""
    df_ks = pd.DataFrame({"pred": prob, "target": y_enc})
    df_ks = df_ks.sort_values("pred").reset_index(drop=True)

    total_bad = df_ks["target"].sum()
    total_good = len(df_ks) - total_bad

    if total_bad == 0 or total_good == 0:
        return None

    cum_bad = df_ks["target"].cumsum() / total_bad
    cum_good = (1 - df_ks["target"]).cumsum() / total_good

    return float((cum_bad - cum_good).max())


# ── IV ─────────────────────────────────────────────────────────────────────


def _calc_iv(series: pd.Series, y: np.ndarray, is_cat: bool = False) -> float:
    """Calculate Information Value for a single feature."""
    temp = pd.DataFrame({"var": series, "target": y}).dropna()
    if len(temp) == 0:
        return 0.0

    total_bad = temp["target"].sum()
    total_good = len(temp) - total_bad

    if total_bad == 0 or total_good == 0:
        return 0.0

    if is_cat:
        groups = temp.groupby("var", observed=True)
    else:
        n_unique = temp["var"].nunique()
        if n_unique <= 5:
            groups = temp.groupby("var", observed=True)
        else:
            n_bins = min(10, max(2, n_unique))
            try:
                temp["bin"] = pd.qcut(temp["var"], n_bins, duplicates="drop")
            except ValueError:
                return 0.0
            groups = temp.groupby("bin", observed=True)

    iv = 0.0
    for _, g in groups:
        bad = g["target"].sum()
        good = len(g) - bad
        bad_dist = bad / total_bad
        good_dist = good / total_good
        if bad_dist > 0 and good_dist > 0:
            woe = np.log(bad_dist / good_dist)
            iv += (bad_dist - good_dist) * woe
    return iv


# ── PSI ────────────────────────────────────────────────────────────────────


def _psi_from_distributions(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    """Population Stability Index; *expected* = reference (train), *actual* = compare (test)."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = np.clip(e, eps, 1.0)
    a = np.clip(a, eps, 1.0)
    return float(np.sum((a - e) * np.log(a / e)))


def _calc_psi(train: pd.Series, test: pd.Series, is_cat: bool) -> float | None:
    """
    PSI of *test* vs *train* marginal distribution.

    Bins are defined on *train* only (same structure as IV: category levels or
    qcut / raw levels for low-cardinality numerics).
    """
    tr = train.dropna()
    te = test.dropna()
    if len(tr) < 2 or len(te) < 1:
        return None
    if tr.nunique() <= 1:
        return None

    other = "__PSI_UNSEEN__"

    if is_cat:
        train_props = tr.value_counts(normalize=True)
        cats = train_props.index
        te_bins = te.where(te.isin(cats), other)
        test_props = te_bins.value_counts(normalize=True)
    else:
        n_unique = tr.nunique()
        if n_unique <= 5:
            train_props = tr.value_counts(normalize=True)
            cats = train_props.index
            te_bins = te.where(te.isin(cats), other)
            test_props = te_bins.value_counts(normalize=True)
        else:
            n_bins = min(10, max(2, int(n_unique)))
            try:
                _, bin_edges = pd.qcut(tr, n_bins, duplicates="drop", retbins=True)
            except ValueError:
                return None
            train_bin = pd.cut(tr, bins=bin_edges, include_lowest=True)
            test_bin = pd.cut(te, bins=bin_edges, include_lowest=True)
            oob = test_bin.isna()
            train_props = train_bin.value_counts(normalize=True)
            test_props = test_bin.dropna().value_counts(normalize=True)
            if oob.any():
                nan_frac = float(oob.sum()) / float(len(te))
                if nan_frac > 0:
                    test_props = test_props.copy()
                    test_props[other] = float(test_props.get(other, 0.0)) + nan_frac

    idx = train_props.index.union(test_props.index, sort=False)
    e = train_props.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    a = test_props.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    es = e.sum()
    asum = a.sum()
    if es <= 0 or asum <= 0:
        return None
    e = e / es
    a = a / asum
    return _psi_from_distributions(e, a)


# ── Estimator registry ─────────────────────────────────────────────────────


def list_supported_estimators() -> list[str]:
    """Return list of estimator names available on this system."""
    return sorted(_SUPPORTED_ESTIMATORS)
