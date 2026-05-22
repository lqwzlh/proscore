"""Matplotlib-based visualization functions for scorecard diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import auc, roc_curve

if TYPE_CHECKING:
    from proscore.binning._base import BinTable


def _ensure_mpl():
    """Lazy-import matplotlib; raise a clear error if not installed."""
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualisation. Install with: pip install matplotlib"
        )


# ── binning distribution ──────────────────────────────────────────────────


def plot_binning(bt: BinTable, figsize: tuple[float, float] = (9, 4)):
    """
    Plot per-bin sample counts (bar) and bad rate (line) for a single variable.

    Parameters
    ----------
    bt : BinTable
        A fitted binning result for one variable.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt

    bins = [b for b in bt.bins if b.bin_label != "missing"]
    if not bins:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{bt.var} — no bins")
        return fig

    labels = [b.bin_label for b in bins]
    counts = [b.count for b in bins]
    bad_rates = [b.bad_rate for b in bins]
    woes = [b.woe for b in bins]

    fig, ax1 = plt.subplots(figsize=figsize)

    x = np.arange(len(labels))
    bars = ax1.bar(x, counts, color="steelblue", alpha=0.7, label="Count")
    ax1.set_ylabel("Count", color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    ax2 = ax1.twinx()
    line = ax2.plot(x, bad_rates, "o-", color="darkorange", linewidth=2, label="Bad rate")
    ax2.set_ylabel("Bad rate", color="darkorange")

    # WOE annotation
    for i, (br, w) in enumerate(zip(bad_rates, woes)):
        ax2.annotate(
            f"WOE={w:.2f}",
            (x[i], br),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=7,
            color="gray",
        )

    ax1.set_title(f"{bt.var}  (IV={bt.iv_total:.4f}, trend={bt.monotonic})")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    return fig


# ── KS curve ──────────────────────────────────────────────────────────────


def plot_ks(
    y_true: list | np.ndarray,
    prob: list | np.ndarray,
    labels: tuple[str, ...] | None = None,
    figsize: tuple[float, float] = (7, 5),
):
    """
    Plot KS curve (cumulative bad / good distributions).

    Parameters
    ----------
    y_true : list or np.ndarray
        Binary target (1 = bad).
    prob : list or np.ndarray
        Predicted probabilities.
    labels : tuple of str, optional
        Curve label(s).  Default ``("Model",)``.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt

    yt = np.asarray(y_true, dtype=float).ravel()
    pr = np.asarray(prob, dtype=float).ravel()
    if labels is None:
        labels = ("Model",)

    # Sort by predicted probability
    order = np.argsort(pr)
    yt_sorted = yt[order]

    total_bad = yt_sorted.sum()
    total_good = len(yt_sorted) - total_bad
    if total_bad == 0 or total_good == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("KS — insufficient class balance")
        return fig

    cum_bad = np.cumsum(yt_sorted) / total_bad
    cum_good = np.cumsum(1 - yt_sorted) / total_good
    ks_val = np.abs(cum_bad - cum_good).max()
    ks_x = np.arange(len(yt_sorted)) / len(yt_sorted)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ks_x, cum_bad, color="darkred", linewidth=2, label=f"{labels[0]} — Bad")
    ax.plot(ks_x, cum_good, color="steelblue", linewidth=2, label=f"{labels[0]} — Good")
    ax.fill_between(ks_x, cum_bad, cum_good, alpha=0.1, color="gray")

    # Mark KS point
    ks_idx = np.argmax(np.abs(cum_bad - cum_good))
    ax.axvline(ks_x[ks_idx], color="gray", linestyle="--", alpha=0.5)
    ax.text(
        ks_x[ks_idx] + 0.02, 0.5,
        f"KS = {ks_val:.4f}",
        fontsize=10,
        color="black",
    )

    ax.set_xlabel("Fraction of population")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"KS Curve  (KS = {ks_val:.4f})")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    return fig


# ── ROC curve ─────────────────────────────────────────────────────────────


def plot_roc(
    y_true: list | np.ndarray,
    prob: list | np.ndarray,
    labels: tuple[str, ...] | None = None,
    figsize: tuple[float, float] = (6, 6),
):
    """
    Plot ROC curve with AUC annotation.

    Parameters
    ----------
    y_true : list or np.ndarray
        Binary target.
    prob : list or np.ndarray
        Predicted probabilities.
    labels : tuple of str, optional
        Curve label.  Default ``("Model",)``.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt

    yt = np.asarray(y_true, dtype=float).ravel()
    pr = np.asarray(prob, dtype=float).ravel()
    if labels is None:
        labels = ("Model",)

    fig, ax = plt.subplots(figsize=figsize)

    fpr, tpr, _ = roc_curve(yt, pr)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, linewidth=2, color="darkred", label=f"{labels[0]} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    fig.tight_layout()
    return fig


# ── score distribution ────────────────────────────────────────────────────


def plot_score_distribution(
    train_scores: list | np.ndarray,
    test_scores: list | np.ndarray | None = None,
    n_bins: int = 20,
    figsize: tuple[float, float] = (8, 4),
):
    """
    Plot score distribution histogram (train, optional test overlay).

    Parameters
    ----------
    train_scores : list or np.ndarray
        Training-set scores.
    test_scores : list or np.ndarray, optional
        Test/OOT scores for overlay comparison.
    n_bins : int
        Number of histogram bins.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt

    trn = np.asarray(train_scores, dtype=float).ravel()
    tst = np.asarray(test_scores, dtype=float).ravel() if test_scores is not None else None

    # Use common bin edges
    all_scores = trn if tst is None else np.concatenate([trn, tst])
    bins = np.histogram_bin_edges(all_scores, bins=n_bins)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(trn, bins=bins, alpha=0.6, color="steelblue", label=f"Train (n={len(trn)})")
    if tst is not None:
        ax.hist(tst, bins=bins, alpha=0.5, color="darkorange", label=f"Test (n={len(tst)})")

    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig
