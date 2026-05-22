"""Chi-square binning: merge adjacent bins until the chi-square statistic
exceeds a confidence threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd


def chi_binning(
    feature: np.ndarray,
    target: np.ndarray,
    n_bins: int,
    confidence_val: float = 3.841,
    max_initial_bins: int = 50,
) -> list[float]:
    """
    Chi-square merge binning.

    Continuous features are first pre-binned into at most *max_initial_bins*
    equal-frequency bins.  Adjacent bins are then merged iteratively (lowest
    chi-square first) until *n_bins* is reached or all chi-square values
    exceed *confidence_val*.

    Returns sorted cut points (excluding -inf / +inf).
    """
    if n_bins < 2:
        return []

    # --- pre-bin continuous features ---
    fvals = np.unique(feature)
    n_unique = len(fvals)

    if max_initial_bins and n_unique > max_initial_bins:
        try:
            _, edges = pd.qcut(feature, max_initial_bins, retbins=True, duplicates="drop")
        except ValueError:
            return []
        # edges includes min and max; drop both
        edges = edges[1:-1]
        binned = np.digitize(feature, edges, right=True)
        regroup = _build_regroup_from_bins(binned, target, edges)
    else:
        target_classes = np.unique(target)
        n_classes = len(target_classes)
        counts = np.zeros((n_unique, n_classes), dtype=np.float64)
        for r, v in enumerate(fvals):
            mask = feature == v
            for c, tv in enumerate(target_classes):
                counts[r, c] = (target[mask] == tv).sum()
        regroup = np.c_[fvals.astype(np.float64), counts]

    # --- merge consecutive rows that share a zero-count class ---
    regroup = _merge_zero_class_runs(regroup)
    if regroup.shape[0] <= n_bins:
        return _regroup_cuts(regroup)

    # --- iterative chi-square merging ---
    chi_list = _chi_pairwise(regroup)
    while regroup.shape[0] > n_bins:
        min_idx = int(np.argmin(chi_list))
        if chi_list[min_idx] >= confidence_val:
            break
        regroup, chi_list = _merge_at(regroup, chi_list, min_idx)

    return _regroup_cuts(regroup)


# ── helpers ────────────────────────────────────────────────────────────────


def _build_regroup_from_bins(
    binned: np.ndarray, target: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """
    Build a regroup matrix from pre-binned data.

    Returns an (n_bins × (1 + n_classes)) array where column 0 is the right
    edge of each bin and columns 1.. are class counts.
    """
    target_classes = np.unique(target)
    n_classes = len(target_classes)
    bin_ids = np.unique(binned)
    n_actual = len(bin_ids)

    regroup = np.zeros((n_actual, 1 + n_classes), dtype=np.float64)
    for r, bid in enumerate(bin_ids):
        mask = binned == bid
        # Right boundary: if bid < len(edges), use edges[bid]; else +inf
        if bid < len(edges):
            regroup[r, 0] = float(edges[bid])
        else:
            regroup[r, 0] = np.inf
        for c, tv in enumerate(target_classes):
            regroup[r, 1 + c] = (target[mask] == tv).sum()
    return regroup


def _regroup_cuts(regroup: np.ndarray) -> list[float]:
    """Extract cut points from the regroup matrix (last value of each bin)."""
    if regroup.shape[0] < 2:
        return []
    return sorted(set(float(v) for v in regroup[:-1, 0]))


def _chi2(regroup: np.ndarray, i: int) -> float:
    """
    Chi-square for rows *i* and *i+1* (binary target: column 1=bad, column 2=good).

    Standard 2×2 formula:
        χ² = (ad - bc)² · N / [(a+b)(c+d)(a+c)(b+d)]
    where a,b are (bad, good) of row i and c,d are (bad, good) of row i+1.
    """
    a = regroup[i, 1]     # bad in row i
    b = regroup[i, 2]     # good in row i
    c = regroup[i + 1, 1] # bad in row i+1
    d = regroup[i + 1, 2] # good in row i+1

    n = a + b + c + d
    if n == 0:
        return 0.0

    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom <= 0:
        return float("inf")

    return (a * d - b * c) ** 2 * n / denom


def _chi_pairwise(regroup: np.ndarray) -> np.ndarray:
    """Compute chi-square for each adjacent pair of rows."""
    n = regroup.shape[0]
    return np.array([_chi2(regroup, i) for i in range(n - 1)], dtype=np.float64)


def _merge_at(
    regroup: np.ndarray, chi_list: np.ndarray, idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge row *idx* and *idx+1* in *regroup*, keeping the larger boundary
    value.  Update *chi_list* to reflect only the affected neighbours.
    """
    orig_n = regroup.shape[0]

    # Merge counts
    regroup[idx, 1:] = regroup[idx, 1:] + regroup[idx + 1, 1:]
    regroup[idx, 0] = regroup[idx + 1, 0]
    regroup = np.delete(regroup, idx + 1, axis=0)

    # Recompute chi values for affected neighbours
    if idx == orig_n - 2:  # merged the two last rows
        chi_list = np.delete(chi_list, idx)
        if idx > 0:
            chi_list[idx - 1] = _chi2(regroup, idx - 1)
    elif idx == 0:  # merged the two first rows
        chi_list[0] = _chi2(regroup, 0)
        chi_list = np.delete(chi_list, 1)
    else:  # middle merge
        chi_list[idx - 1] = _chi2(regroup, idx - 1)
        chi_list[idx] = _chi2(regroup, idx)
        chi_list = np.delete(chi_list, idx + 1)

    return regroup, chi_list


def _merge_zero_class_runs(regroup: np.ndarray) -> np.ndarray:
    """
    Merge consecutive rows that both have zero count in the **same** target class.

    The original chi-merge algorithm merges any run of rows sharing a zero-count
    class column, because the chi-square formula would otherwise produce a
    division by zero.
    """
    n_classes = regroup.shape[1] - 1  # columns after feature value
    i = 0
    while i <= regroup.shape[0] - 2:
        merge_col = None
        for col in range(n_classes):
            if regroup[i, 1 + col] == 0 and regroup[i + 1, 1 + col] == 0:
                merge_col = col
                break

        if merge_col is not None:
            # Find the full run of rows with zero in this class
            end = i + 1
            while end + 1 <= regroup.shape[0] - 1:
                if regroup[end + 1, 1 + merge_col] == 0:
                    end += 1
                else:
                    break
            # Merge rows i .. end into row i
            regroup[i, 1:] = regroup[i : end + 1, 1:].sum(axis=0)
            regroup[i, 0] = regroup[end, 0]  # keep rightmost boundary
            regroup = np.delete(regroup, list(range(i + 1, end + 1)), axis=0)
            i -= 1  # re-check from this position
        i += 1
    return regroup
