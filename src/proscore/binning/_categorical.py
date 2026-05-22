"""Categorical variable binning strategies.

Each strategy returns a ``dict[value, bin_no]`` mapping original category
values to zero-based bin indices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def woe_per_value(series: pd.Series, target: pd.Series, **kwargs) -> dict:
    """
    Each distinct category becomes its own bin.

    Returns a mapping from original category value → bin index (0-based),
    ordered by category value.
    """
    values = sorted(series.dropna().unique())
    return {v: i for i, v in enumerate(values)}


def badrate_merge(
    series: pd.Series,
    target: pd.Series,
    min_bin_pct: float = 0.05,
    badrate_gap: float = 0.05,
    **kwargs,
) -> dict:
    """
    Merge categories with similar bad rates.

    Categories are sorted by bad rate, then adjacent categories are merged
    when either the current group is too small or the bad-rate gap to the
    next category is narrow.

    Returns a ``{value: bin_no}`` mapping.
    """
    stats = (
        pd.DataFrame({"var": series, "target": target})
        .groupby("var", observed=True)
        .agg(count=("target", "count"), bad=("target", "sum"))
        .assign(bad_rate=lambda x: x["bad"] / x["count"])
        .sort_values("bad_rate")
        .reset_index()
    )

    total_n = len(series.dropna())
    if total_n == 0:
        return {}

    # Greedy merge by bad-rate proximity
    bin_labels: list[list] = []
    cur_group: list = [stats["var"].iloc[0]]
    cur_count = int(stats["count"].iloc[0])
    cur_br = float(stats["bad_rate"].iloc[0])

    for i in range(1, len(stats)):
        row = stats.iloc[i]
        if (
            cur_count / total_n < min_bin_pct
            or abs(float(row["bad_rate"]) - cur_br) < badrate_gap
        ):
            cur_group.append(row["var"])
            cur_count += int(row["count"])
            # Weighted-average bad rate
            new_n = int(row["count"])
            cur_br = (cur_br * (cur_count - new_n) + float(row["bad_rate"]) * new_n) / cur_count
        else:
            bin_labels.append(cur_group)
            cur_group = [row["var"]]
            cur_count = int(row["count"])
            cur_br = float(row["bad_rate"])

    if cur_group:
        bin_labels.append(cur_group)

    mapping: dict = {}
    for bin_no, group in enumerate(bin_labels):
        for v in group:
            mapping[v] = bin_no
    return mapping


def freq_merge(
    series: pd.Series,
    target: pd.Series,
    top_n: int = 10,
    **kwargs,
) -> dict:
    """
    Keep the *top_n* most frequent categories as individual bins; merge the
    rest into an "Other" bin.

    Returns a ``{value: bin_no}`` mapping.
    """
    freq = series.value_counts()
    if len(freq) <= top_n:
        top = freq.index.tolist()
    else:
        top = freq.head(top_n - 1).index.tolist()

    mapping: dict = {}
    for i, v in enumerate(top):
        mapping[v] = i
    if len(freq) > len(top):
        other_idx = len(top)
        for v in freq.index:
            if v not in mapping:
                mapping[v] = other_idx
    return mapping
