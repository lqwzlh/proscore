"""Time-series variable stability analysis: bad_rate trend, PSI drift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from proscore.utils import is_categorical
from proscore.utils._psi import psi_from_distributions


def stability(
    df: pd.DataFrame,
    target: str,
    time_col: str,
    features: list[str] | None = None,
    n_bins: int = 5,
    bad_rate_trend_threshold: float = 0.5,
    psi_warn_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Time-series stability analysis for each feature.

    For each time period, computes sample count, bad rate, distribution PSI
    (vs first period and vs previous period), and **two independent flags**:

    - ``psi_flag``: distribution drift vs the first period (PSI).
    - ``bad_rate_flag``: bad-rate trend vs the first period (relative change).

    ``bad_rate_change`` is the relative change from the first period:
    ``(bad_rate[t] - bad_rate[0]) / bad_rate[0]``.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.  Must contain *target* and *time_col*.
    target : str
        Binary target column (1 = bad).  Rows where *target* is NaN are
        excluded from bad-rate and sample-count calculations.
    time_col : str
        Column identifying time periods (month, quarter, etc.).  Must be
        sortable — the function sorts periods ascending.
    features : list of str, optional
        Columns to analyse.  Defaults to all numeric + categorical columns
        (excludes *target* and *time_col*).
    n_bins : int
        Number of equal-frequency bins for continuous PSI calculation.
    bad_rate_trend_threshold : float
        Relative change in bad rate that triggers a ``"trending"`` flag on
        ``bad_rate_flag``.  For example, 0.5 means a 50% increase or decrease
        from the first period is flagged.
    psi_warn_threshold : float
        PSI threshold above which ``psi_flag`` is set to ``"unstable"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``variable | time_period | n | bad_rate |
        bad_rate_change | psi_vs_first | psi_vs_prev | mean | std |
        psi_flag | bad_rate_flag``.
    """
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in DataFrame")
    if time_col not in df.columns:
        raise KeyError(f"time_col {time_col!r} not in DataFrame")

    skip = {target, time_col}
    feats = features or [c for c in df.columns if c not in skip]

    # Sort time periods
    periods = sorted(df[time_col].dropna().unique())
    if len(periods) < 2:
        raise ValueError(f"Need at least 2 distinct time periods; got {len(periods)}")

    rows: list[dict] = []

    for col in feats:
        series = df[col]
        cat = is_categorical(series)

        # Collect ALL categories for categorical variables (across all periods)
        all_cats = None
        if cat:
            all_cats = sorted(series.dropna().unique())

        # Baseline distribution (first period)
        base_mask = df[time_col] == periods[0]
        base_data = series[base_mask].dropna()
        base_bins = _distribution_bins(base_data, cat, n_bins, all_cats)
        base_dist = _distribution(base_data, base_bins, cat)

        first_bad_rate = None
        prev_dist = None

        for p_idx, period in enumerate(periods):
            mask = df[time_col] == period
            # Exclude rows where target is NaN for bad-rate calculation
            sub = df.loc[mask, [target, col]].dropna(subset=[target])
            sub_target = sub[target]
            sub_data = series[mask].dropna()  # full data for PSI/distribution

            n = len(sub_target)
            bad = sub_target.sum()
            bad_rate = bad / n if n > 0 else np.nan

            if p_idx == 0:
                first_bad_rate = bad_rate

            # Bad rate change vs first period
            if first_bad_rate is not None and first_bad_rate > 0 and not np.isnan(bad_rate):
                br_change = (bad_rate - first_bad_rate) / first_bad_rate
            else:
                br_change = np.nan

            # PSI
            cur_dist = _distribution(sub_data, base_bins, cat)
            psi_first = psi_from_distributions(base_dist, cur_dist)
            psi_prev = (
                psi_from_distributions(prev_dist, cur_dist)
                if prev_dist is not None
                else np.nan
            )

            # Numeric stats
            mean_val = sub_data.mean() if not cat else np.nan
            std_val = sub_data.std() if not cat else np.nan

            rows.append({
                "variable": col,
                "time_period": period,
                "n": int(n),
                "bad_rate": round(bad_rate, 6) if not np.isnan(bad_rate) else np.nan,
                "bad_rate_change": round(br_change, 4) if not np.isnan(br_change) else np.nan,
                "psi_vs_first": round(psi_first, 6),
                "psi_vs_prev": round(psi_prev, 6) if not np.isnan(psi_prev) else np.nan,
                "mean": round(float(mean_val), 4) if not np.isnan(mean_val) else np.nan,
                "std": round(float(std_val), 4) if not np.isnan(std_val) else np.nan,
                "psi_flag": _psi_flag(p_idx, psi_first, psi_warn_threshold),
                "bad_rate_flag": _bad_rate_flag(
                    p_idx, br_change, bad_rate_trend_threshold,
                ),
            })

            prev_dist = cur_dist

    result = pd.DataFrame(rows)
    result["time_period"] = pd.Categorical(
        result["time_period"], categories=periods, ordered=True
    )
    return result.sort_values(["variable", "time_period"]).reset_index(drop=True)


def stability_summary(
    stability_result: pd.DataFrame,
    *,
    metric: str = "bad_rate",
) -> pd.DataFrame:
    """
    Pivot long-form :func:`stability` output to one row per variable.

    Parameters
    ----------
    stability_result : pd.DataFrame
        Output of :func:`stability`.
    metric : str
        Column to pivot: ``"bad_rate"``, ``"psi_vs_first"``, or
        ``"bad_rate_change"``.

    Returns
    -------
    pd.DataFrame
        Index ``variable``; columns are time periods; extra columns
        ``latest_psi_flag`` and ``latest_bad_rate_flag`` from the last period.
    """
    if metric not in stability_result.columns:
        raise KeyError(f"metric {metric!r} not in stability result columns")
    if len(stability_result) == 0:
        return pd.DataFrame()

    wide = stability_result.pivot_table(
        index="variable",
        columns="time_period",
        values=metric,
        aggfunc="first",
    )
    latest = (
        stability_result.sort_values("time_period")
        .groupby("variable", observed=True)
        .last()
    )
    wide["latest_psi_flag"] = latest["psi_flag"].reindex(wide.index)
    wide["latest_bad_rate_flag"] = latest["bad_rate_flag"].reindex(wide.index)
    return wide.reset_index()


# ── internal helpers ───────────────────────────────────────────────────────


def _distribution_bins(
    data: pd.Series,
    cat: bool,
    n_bins: int,
    all_cats: list | None = None,
):
    """Build bin edges from the first-period data (or category list).

    For categorical variables, *all_cats* provides the complete set of
    categories across all periods so that new categories in later periods
    are not lost.
    """
    if cat:
        return all_cats or sorted(data.unique())
    if len(data.unique()) <= n_bins:
        return sorted(data.unique())
    # Let ValueError propagate on degenerate data (Fail Fast)
    _, edges = pd.qcut(data, n_bins, retbins=True, duplicates="drop")
    return edges


def _distribution(data: pd.Series, bins, cat: bool) -> np.ndarray:
    """Compute frequency distribution over *bins*."""
    if len(data) == 0:
        return np.array([])
    if cat or len(bins) <= 5:
        counts = data.value_counts()
        return counts.reindex(bins, fill_value=0).to_numpy(dtype=float)
    else:
        digitized = pd.cut(data, bins=bins, labels=False, include_lowest=True)
        n_bins_actual = len(bins) - 1
        dist = np.zeros(n_bins_actual, dtype=float)
        for idx, cnt in pd.Series(digitized).value_counts(sort=False).items():
            i = int(idx)
            if 0 <= i < n_bins_actual:
                dist[i] = cnt
        return dist


def _psi_flag(p_idx: int, psi_first: float, psi_threshold: float) -> str:
    """PSI-based distribution stability: baseline | stable | unstable."""
    if p_idx == 0:
        return "baseline"
    if not np.isnan(psi_first) and psi_first > psi_threshold:
        return "unstable"
    return "stable"


def _bad_rate_flag(
    p_idx: int,
    br_change: float,
    br_threshold: float,
) -> str:
    """Bad-rate trend stability: baseline | stable | trending_up | trending_down."""
    if p_idx == 0:
        return "baseline"
    if not np.isnan(br_change) and abs(br_change) > br_threshold:
        if br_change > 0:
            return "trending_up"
        return "trending_down"
    return "stable"
