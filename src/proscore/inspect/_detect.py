"""Data profiling: variable quality overview."""

from __future__ import annotations

import numpy as np
import pandas as pd

from proscore.utils import is_categorical, require_unique_column_labels


def detect(
    df: pd.DataFrame,
    target: str | None = None,
    special_values: list | None = None,
    max_categories: int = 20,
) -> pd.DataFrame:
    """
    Produce a variable quality overview table for every column in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Input data; column labels must be unique.
    target : str, optional
        Target column name. If provided, two association columns are added:
        ``target_pearson`` (numeric features: Pearson with target) and
        ``target_cramers_v`` (categorical: Cramér's V with target). The target
        column itself is excluded from the table.
    special_values : list, optional
        Values to treat as special (e.g. ``[-999, -1, 999999999]``).
        When ``None`` (default), no special-value check is performed and
        ``special_pct`` / ``special_pct_valid`` are set to 0.0 / NaN.
    max_categories : int
        Columns with fewer unique values than this threshold are treated
        as categorical for dtype inference.

    Returns
    -------
    pd.DataFrame
        Base columns: ``variable | dtype | count | n_unique | missing |
        missing_pct | one_value_pct | special_pct | special_pct_valid |
        mean | std | min | p25 | p50 | p75 | max``.
        If *target* is set, also ``target_pearson`` and ``target_cramers_v``
        (exactly one is non-NaN per row when computable).
    """
    require_unique_column_labels(df)
    if target is not None and target not in df.columns:
        raise KeyError(f"target column {target!r} not in DataFrame")

    rows: list[dict] = []
    target_series = df[target] if target is not None else None

    for col in df.columns:
        if col == target:
            continue

        series = df[col]
        n = len(series)
        n_nonnull = series.count()
        n_missing = n - n_nonnull

        # Infer type
        if is_categorical(series, max_categories):
            dtype_label = "categorical"
        elif pd.api.types.is_numeric_dtype(series):
            dtype_label = "numeric"
        else:
            dtype_label = "other"

        # One-value rate
        if n_nonnull > 0:
            top_count = series.dropna().value_counts().iloc[0]
            one_value_pct = top_count / n_nonnull
        else:
            one_value_pct = np.nan

        # Special value rate (two denominators: total and non-null)
        if special_values and n_nonnull > 0:
            special_mask = series.isin(special_values)
            special_count = special_mask.sum()
            special_pct = special_count / n
            special_pct_valid = special_count / n_nonnull
        else:
            special_pct = 0.0
            special_pct_valid = np.nan

        row: dict = {
            "variable": col,
            "dtype": dtype_label,
            "count": n,
            "n_unique": int(series.nunique()),
            "missing": n_missing,
            "missing_pct": round(n_missing / n * 100, 2) if n > 0 else 0.0,
            "one_value_pct": (
                round(one_value_pct * 100, 2) if not np.isnan(one_value_pct) else np.nan
            ),
            "special_pct": round(special_pct * 100, 2),
            "special_pct_valid": (
                round(special_pct_valid * 100, 2) if not np.isnan(special_pct_valid) else np.nan
            ),
        }

        # Numeric stats
        if dtype_label == "numeric":
            desc = series.describe()
            row.update(
                {
                    "mean": round(desc.get("mean", np.nan), 4) if "mean" in desc else np.nan,
                    "std": round(desc.get("std", np.nan), 4) if "std" in desc else np.nan,
                    "min": desc.get("min", np.nan),
                    "p25": desc.get("25%", np.nan),
                    "p50": desc.get("50%", np.nan),
                    "p75": desc.get("75%", np.nan),
                    "max": desc.get("max", np.nan),
                }
            )
        else:
            row.update(
                {
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "p25": np.nan,
                    "p50": np.nan,
                    "p75": np.nan,
                    "max": np.nan,
                }
            )

        if target is not None and target_series is not None:
            pearson_v, cramers_v = _target_association(series, target_series, dtype_label)
            row["target_pearson"] = pearson_v
            row["target_cramers_v"] = cramers_v

        rows.append(row)

    result = pd.DataFrame(rows)
    cols = [
        "variable", "dtype", "count", "n_unique", "missing", "missing_pct",
        "one_value_pct", "special_pct", "special_pct_valid",
        "mean", "std", "min", "p25", "p50", "p75", "max",
    ]
    if target is not None:
        cols.extend(["target_pearson", "target_cramers_v"])
    return result[[c for c in cols if c in result.columns]]


# ── Target association helpers ─────────────────────────────────────────────


def _chi2_independence(table: np.ndarray) -> float:
    """Pearson chi-squared statistic for a contingency table (no scipy)."""
    obs = table.astype(float)
    n = obs.sum()
    if n <= 0:
        return 0.0
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = np.where(expected > 0, (obs - expected) ** 2 / expected, 0.0)
    return float(np.nansum(contrib))


def _cramers_v_with_target(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V between two categorical-like columns (pairwise non-null)."""
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 2:
        return float("nan")
    if pair["y"].nunique() <= 1 or pair["x"].nunique() <= 1:
        return 0.0
    tab = pd.crosstab(pair["x"], pair["y"])
    chi2 = _chi2_independence(tab.values)
    n = tab.values.sum()
    min_dim = min(tab.shape[0] - 1, tab.shape[1] - 1)
    if n <= 0 or min_dim <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def _target_association(
    series: pd.Series,
    target: pd.Series,
    dtype_label: str,
) -> tuple[float, float]:
    """Return (pearson_with_target, cramers_v_with_target); one is NaN by dtype."""
    nan = float("nan")
    if dtype_label == "numeric":
        pair = pd.DataFrame({"x": series, "y": target}).dropna()
        if len(pair) < 2:
            return nan, nan
        y_num = pd.to_numeric(pair["y"], errors="coerce")
        mask = y_num.notna()
        pair = pair.loc[mask].copy()
        pair["y"] = y_num.loc[mask].astype(float)
        if len(pair) < 2 or pair["x"].nunique() <= 1 or pair["y"].nunique() <= 1:
            return nan, nan
        pair["x"] = pd.to_numeric(pair["x"], errors="coerce")
        pair = pair.dropna(subset=["x"])
        if len(pair) < 2 or pair["x"].nunique() <= 1:
            return nan, nan
        r = pair["x"].corr(pair["y"], method="pearson")
        if r is None or (isinstance(r, float) and np.isnan(r)):
            return nan, nan
        return round(float(r), 6), nan

    if dtype_label == "categorical":
        v = _cramers_v_with_target(series, target)
        if np.isnan(v):
            return nan, nan
        return nan, round(float(v), 6)

    return nan, nan
