"""Equal-frequency (quantile) binning."""

from __future__ import annotations

import numpy as np
import pandas as pd


def frequency_binning(feature: np.ndarray, n_bins: int) -> list[float]:
    """
    Equal-frequency binning: each bin contains approximately the same number
    of observations.

    Returns sorted cut points (excluding -inf / +inf).
    """
    if n_bins < 2:
        return []

    # pd.qcut handles duplicates and edge cases
    try:
        _, edges = pd.qcut(
            feature, n_bins, retbins=True, duplicates="drop"
        )
    except ValueError:
        return []

    # edges[0] = min, edges[-1] = max — drop both
    cuts = edges[1:-1].tolist()
    # Map back to original values for cleaner cut points
    return [_map_to_actual(feature, c) for c in cuts]


def _map_to_actual(feature: np.ndarray, cutoff: float) -> float:
    """Map a computed cut point to the closest actual value in the data."""
    candidates = feature[feature <= cutoff]
    if len(candidates) == 0:
        return cutoff
    return float(np.nanmax(candidates))
