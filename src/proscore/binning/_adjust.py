"""Trend adjustment for binning results: monotonicity, UV detection, and
quality checks."""

from __future__ import annotations

import numpy as np

from proscore.binning._woe import calc_iv_woe


def monotonicity_check(bad_rates: list | np.ndarray) -> int:
    """
    Check whether bad rates are monotonic.

    Returns
    -------
    0 : no trend
    1 : strictly increasing
    2 : strictly decreasing
    """
    br = np.asarray(bad_rates, dtype=float)
    if len(br) < 2:
        return 0
    if np.all(np.diff(br) >= 0):
        return 1
    if np.all(np.diff(br) <= 0):
        return 2
    return 0


def uv_check(
    regroup: np.ndarray,
    side_bin_p: float = 0.2,
    side_sample_p: float = 0.2,
    high_diff_p: float = 0.5,
) -> int:
    """
    Detect whether bad_rate forms a U (valley) or inverted-U (peak) shape.

    Conditions for U/V detection:
        - The lowest (or highest) bad-rate bin lies in the "middle region"
          (excludes *side_bin_p* fraction of bins on each end).
        - Each side has at least *side_sample_p* fraction of total samples.
        - The height difference between the two ends is ≤ *high_diff_p*
          of the total height range.

    Returns
    -------
    0 : no UV shape
    3 : U-shape  (lowest bad_rate in middle, monotonic on both sides)
    4 : inverted-U (highest bad_rate in middle, monotonic on both sides)
    """
    n_bins = regroup.shape[0]
    if n_bins < 3:
        return 0

    side_len = max(int(n_bins * side_bin_p), 1)
    mid_range = range(side_len, n_bins - side_len)

    total = regroup[:, 1] + regroup[:, 2]
    bad_rates = np.divide(
        regroup[:, 1], total, out=np.zeros_like(total), where=total > 0
    )

    min_idx = int(np.argmin(bad_rates))
    max_idx = int(np.argmax(bad_rates))

    # Candidate core: min or max inside middle region
    core_idx: int | None = None
    if min_idx in mid_range:
        core_idx = min_idx
    elif max_idx in mid_range:
        core_idx = max_idx

    if core_idx is None:
        return 0

    left = bad_rates[: core_idx + 1]
    right = bad_rates[core_idx:]

    if not (monotonicity_check(left) and monotonicity_check(right)):
        return 0

    total_samples = float(total.sum())
    left_samples = float(total[:core_idx].sum())
    right_samples = float(total[core_idx + 1 :].sum())

    if left_samples < side_sample_p * total_samples:
        return 0
    if right_samples < side_sample_p * total_samples:
        return 0

    height_range = bad_rates[max_idx] - bad_rates[min_idx]
    if height_range <= 0:
        return 0
    end_diff = abs(float(bad_rates[0]) - float(bad_rates[-1]))
    if end_diff / height_range > high_diff_p:
        return 0

    if monotonicity_check(left) == 2:  # decreasing → then increasing = U
        return 3
    return 4


def get_comb_idx(trend_flag: int, diffs: np.ndarray) -> int:
    """
    Choose which adjacent bin pair to merge to move toward the desired trend.

    Parameters
    ----------
    trend_flag : int
        0 = no preset (merge smallest absolute diff).
        1 = increasing (merge pairs with negative diff to eliminate reversals).
        2 = decreasing (merge pairs with positive diff).
        3 = U-shape (prefer merging end pairs first).
        4 = inverted-U.
    diffs : np.ndarray
        Signed bad-rate differences: ``bad_rate[i+1] - bad_rate[i]``.
    """
    if trend_flag == 0:
        return int(np.argmin(np.abs(diffs)))
    if trend_flag == 1:  # increasing: remove negative steps
        neg = np.where(diffs < 0)[0]
        if len(neg) > 0:
            return int(neg[np.argmin(np.abs(diffs[neg]))])
        return int(np.argmin(np.abs(diffs)))
    if trend_flag == 2:  # decreasing: remove positive steps
        pos = np.where(diffs > 0)[0]
        if len(pos) > 0:
            return int(pos[np.argmin(np.abs(diffs[pos]))])
        return int(np.argmin(np.abs(diffs)))
    if trend_flag == 3:  # U-shape
        if diffs[0] > 0:
            return 0
        if diffs[-1] < 0:
            return len(diffs) - 1
        return int(np.argmin(np.abs(diffs)))
    if trend_flag == 4:  # inverted-U
        if diffs[0] < 0:
            return 0
        if diffs[-1] > 0:
            return len(diffs) - 1
        return int(np.argmin(np.abs(diffs)))
    return int(np.argmin(np.abs(diffs)))


def _merge_at(regroup: np.ndarray, idx: int) -> np.ndarray:
    """Merge row *idx* and *idx+1*, keeping the rightmost boundary."""
    regroup[idx, 1:] = regroup[idx, 1:] + regroup[idx + 1, 1:]
    regroup[idx, 0] = regroup[idx + 1, 0]
    return np.delete(regroup, idx + 1, axis=0)


def _find_small_bin(regroup: np.ndarray, total_n: int, min_bin_pct: float) -> int | None:
    """Index of the smallest bin below threshold, or None."""
    bin_sizes = regroup[:, 1] + regroup[:, 2]
    rates = bin_sizes / total_n
    if np.min(rates) >= min_bin_pct:
        return None
    return int(np.argmin(rates))


def _find_lowest_woe_diff(regroup: np.ndarray, min_woe_diff: float) -> int | None:
    """Index of adjacent pair with smallest WOE difference below threshold."""
    _, _, woe = calc_iv_woe(regroup)
    diffs = np.abs(np.diff(woe))
    min_diff = np.min(diffs)
    if min_diff >= min_woe_diff:
        return None
    return int(np.argmin(diffs))


def trend_adjust(
    regroup: np.ndarray,
    total_n: int,
    *,
    min_bin_pct: float = 0.05,
    min_woe_diff: float = 0.1,
    trend_flag: int = 0,
) -> tuple[np.ndarray, list[float], bool, int]:
    """
    Adjust binning to satisfy quality constraints.

    Phases: (1) merge undersized bins, (2) enforce monotonicity / UV,
    (3) merge bins with too-similar WOE.

    Parameters
    ----------
    regroup : np.ndarray
        Shape (n_bins, 3). Col 0 = boundary, 1 = bad, 2 = good.
    total_n : int
        Total number of observations (including specials/missing for pct calc).
    min_bin_pct : float
        Minimum fraction of total observations per bin.
    min_woe_diff : float
        Minimum WOE difference between adjacent bins.
    trend_flag : int
        0=no constraint, 1=increasing bad_rate, 2=decreasing,
        3=U-shape, 4=inverted-U.

    Returns
    -------
    regroup : np.ndarray
        Adjusted regroup matrix.
    cutoffs : list[float]
    is_valid : bool
        True if the final trend matches *trend_flag* or *trend_flag* was 0.
    real_trend : int
        Actual trend code of the final binning.
    """
    # --- Phase 1: merge undersized bins ---
    if min_bin_pct > 0:
        while regroup.shape[0] > 2:
            idx = _find_small_bin(regroup, total_n, min_bin_pct)
            if idx is None:
                break
            # Merge small bin with its closer neighbour (by bad rate)
            if idx == 0:
                regroup = _merge_at(regroup, 0)
            elif idx == regroup.shape[0] - 1:
                regroup = _merge_at(regroup, idx - 1)
            else:
                br = regroup[:, 1] / (regroup[:, 1] + regroup[:, 2] + 1e-12)
                diff_left = abs(float(br[idx]) - float(br[idx - 1]))
                diff_right = abs(float(br[idx + 1]) - float(br[idx]))
                merge_idx = idx - 1 if diff_left <= diff_right else idx
                regroup = _merge_at(regroup, merge_idx)

    # Detect actual trend
    total = regroup[:, 1] + regroup[:, 2]
    bad_rates = np.divide(
        regroup[:, 1], total, out=np.zeros_like(total), where=total > 0
    )
    real_trend = monotonicity_check(bad_rates)
    if not real_trend:
        real_trend = uv_check(regroup)

    # --- Phase 2: enforce shape ---
    if trend_flag != 0:
        while regroup.shape[0] > 2:
            bad_rates = np.divide(
                regroup[:, 1],
                regroup[:, 1] + regroup[:, 2] + 1e-12,
            )
            current_trend = monotonicity_check(bad_rates)
            if current_trend == trend_flag:
                break

            uv_code = uv_check(regroup)
            if (trend_flag in (3, 4)) and uv_code == trend_flag:
                break

            # Compute signed differences for merge decision
            diffs = np.diff(bad_rates)
            if np.all(diffs == 0):
                break

            merge_idx = get_comb_idx(trend_flag, diffs)
            regroup = _merge_at(regroup, merge_idx)

    # Re-detect final trend
    total = regroup[:, 1] + regroup[:, 2]
    bad_rates = np.divide(
        regroup[:, 1], total, out=np.zeros_like(total), where=total > 0
    )
    real_trend = monotonicity_check(bad_rates)
    if not real_trend:
        real_trend = uv_check(regroup)

    # --- Phase 3: merge bins with too-similar WOE ---
    if min_woe_diff > 0:
        while regroup.shape[0] > 2:
            idx = _find_lowest_woe_diff(regroup, min_woe_diff)
            if idx is None:
                break
            regroup = _merge_at(regroup, idx)

    cutoffs = [float(v) for v in regroup[:-1, 0]]
    is_valid = trend_flag == 0 or real_trend == trend_flag
    return regroup, cutoffs, is_valid, real_trend
