"""Population Stability Index (PSI) — shared implementation."""

from __future__ import annotations

import numpy as np


def psi_from_distributions(
    expected: np.ndarray,
    actual: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Population Stability Index between two frequency distributions.

    *expected* is the reference (train / first period) and *actual* is the
    comparison (test / subsequent period).  Both are normalised internally
    so raw counts are accepted.

    Parameters
    ----------
    expected : np.ndarray
        Reference distribution (counts or proportions).
    actual : np.ndarray
        Comparison distribution.
    eps : float
        Small constant to avoid log(0) or division by zero.

    Returns
    -------
    float
        PSI = Σ (a_i - e_i) × ln(a_i / e_i).  Returns ``np.nan`` when
        either distribution sums to zero.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)

    e_sum = e.sum()
    a_sum = a.sum()
    if e_sum <= 0 or a_sum <= 0:
        return np.nan

    e = e / e_sum
    a = a / a_sum

    e = np.clip(e, eps, 1.0)
    a = np.clip(a, eps, 1.0)

    return float(np.sum((a - e) * np.log(a / e)))
