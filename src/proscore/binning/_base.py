"""Base classes and data structures for binning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BinRecord:
    """A single bin produced by binning."""

    bin_no: int
    min_val: float | None
    max_val: float | None
    count: int
    count_bad: int
    count_good: int
    bad_rate: float
    woe: float
    iv: float
    bin_label: str = ""


@dataclass
class BinTable:
    """Complete binning result for a single variable."""

    var: str
    bins: list[BinRecord] = field(default_factory=list)
    cutoffs: list[float] = field(default_factory=list)
    iv_total: float = 0.0
    method: str = ""
    n_bins: int = 0
    monotonic: int = 0  # actual trend: 0=none,1=increasing,2=decreasing,3=U,4=inverted-U
    trend_preset: int = 0  # user-requested trend (0 = no constraint)
    trend_match: bool = True  # True when actual trend matches preset (or no preset)
    dtype: str = "continuous"  # continuous | categorical
    special_values: list[Any] = field(default_factory=list)
    has_missing: bool = False
    missing_merged: bool = False
    cat_mapping: dict = field(default_factory=dict)  # value → bin_no (categorical only)
