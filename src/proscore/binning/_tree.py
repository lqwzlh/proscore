"""Decision-tree-based binning."""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree


def tree_binning(
    feature: np.ndarray, target: np.ndarray, n_bins: int, min_samples_leaf: int | None = None
) -> list[float]:
    """
    Binning via a shallow decision tree.

    The tree split points become the bin boundaries.

    Returns sorted cut points (excluding -inf / +inf).
    """
    if n_bins < 2:
        return []

    if min_samples_leaf is None:
        min_samples_leaf = max(1, int(len(feature) * 0.01))

    clf = DecisionTreeClassifier(
        max_leaf_nodes=n_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    clf.fit(feature.reshape(-1, 1), target)

    thresholds = clf.tree_.threshold
    thresholds = thresholds[thresholds != _tree.TREE_UNDEFINED]
    return sorted(set(float(t) for t in thresholds))
