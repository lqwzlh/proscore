"""Equidistant (equal-width) binning."""

from __future__ import annotations

import numpy as np


def distance_binning(feature: np.ndarray, n_bins: int) -> list[float]:
    """
    Equal-width binning: split the value range into *n_bins* equal intervals.

    Returns sorted cut points (excluding -inf / +inf).
    """
    lo = np.nanmin(feature)
    hi = np.nanmax(feature)

    if lo == hi or n_bins < 2:
        return []

    step = (hi - lo) / n_bins
    cuts = list(np.arange(lo + step, hi, step))
    # Deduplicate due to floating-point rounding
    return sorted(set(round(c, 12) for c in cuts))
