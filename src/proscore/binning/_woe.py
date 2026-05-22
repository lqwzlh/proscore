"""WOE and IV calculation from a regroup matrix."""

from __future__ import annotations

import math

import numpy as np

# Legacy scorecard convention: cap extreme negative WOE at -2 (was -inf).
_WOE_NEG_EXTREME: float = -2.0
# All-bad bin (no good in bin): positive extreme before normalize.
_WOE_POS_RAW: float = 20.0


def normalize_woe(woe: float) -> float:
    """Map non-finite / extreme WOE to stable values (legacy ``woe_inf_change``).

    * ``-inf`` or below ``_WOE_NEG_EXTREME`` → ``-2``
    * ``+inf`` → raise (incorrect binning / separation)
    """
    if math.isinf(woe) and woe > 0:
        raise ValueError("异常变量或不正确分箱: WOE 为 +inf，请检查分箱或样本分布")
    if math.isinf(woe) and woe < 0:
        return _WOE_NEG_EXTREME
    if woe < _WOE_NEG_EXTREME:
        return _WOE_NEG_EXTREME
    return float(woe)


def calc_iv_woe(regroup: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate total IV, per-bin IV, and per-bin WOE.

    Parameters
    ----------
    regroup : np.ndarray, shape (n_bins, 3)
        Column 0: bin boundary (not used in calculation).
        Column 1: count of bad (target=1).
        Column 2: count of good (target=0).

    Returns
    -------
    iv_total : float
        Sum of per-bin IV.
    iv_per_bin : np.ndarray, shape (n_bins,)
    woe_per_bin : np.ndarray, shape (n_bins,)
    """
    bad = regroup[:, 1].astype(np.float64)
    good = regroup[:, 2].astype(np.float64)

    total_bad = bad.sum()
    total_good = good.sum()

    if total_bad == 0 or total_good == 0:
        n = regroup.shape[0]
        return 0.0, np.zeros(n), np.zeros(n)

    bad_dist = bad / total_bad
    good_dist = good / total_good

    woe = np.zeros_like(bad_dist)
    iv = np.zeros_like(bad_dist)

    for i in range(len(bad_dist)):
        if bad_dist[i] > 0 and good_dist[i] > 0:
            woe[i] = normalize_woe(float(np.log(bad_dist[i] / good_dist[i])))
        elif bad_dist[i] > 0:
            woe[i] = normalize_woe(_WOE_POS_RAW)
        elif good_dist[i] > 0:
            woe[i] = normalize_woe(-_WOE_POS_RAW)
        else:
            woe[i] = 0.0

        iv[i] = float((bad_dist[i] - good_dist[i]) * woe[i])

    return float(iv.sum()), iv, woe
