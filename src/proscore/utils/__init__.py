"""Shared utilities for ProScore."""

from __future__ import annotations

import numpy as np
import pandas as pd

from proscore.utils._presets import PresetResult


def load_presets(path: str, *, sheet_name: str = "variables") -> PresetResult:
    """Load variable presets from an Excel file.

    Requires ``openpyxl`` (install with ``pip install proscore[excel]``).

    Returns a :class:`PresetResult` with ``feature_config`` (for
    :class:`~proscore.binning.BinningProcess`) and ``feature_belong`` (for
    :class:`~proscore.selection.StepwiseSelector`).
    """
    from proscore.utils._presets import load_presets as _load

    return _load(path, sheet_name=sheet_name)


# ── Column / DataFrame validation ──────────────────────────────────────────


def require_unique_column_labels(df: pd.DataFrame) -> None:
    """Raise ValueError if *df* has duplicate column labels."""
    if df.columns.duplicated().any():
        dup_labels = (
            df.columns[df.columns.duplicated(keep=False)].unique().tolist()
        )
        raise ValueError(
            "DataFrame has duplicate column labels; rename columns to unique "
            "names before calling this function. "
            f"Duplicated labels include: {dup_labels[:30]!r}"
            + (" ..." if len(dup_labels) > 30 else "")
        )


def require_unique_feature_list(features: list[str], *, arg_name: str = "features") -> None:
    """Raise ValueError if *features* contains duplicate entries."""
    if len(features) != len(set(features)):
        raise ValueError(
            f"{arg_name} contains duplicate entries; pass each column name at most once."
        )


# ── Dtype inference ────────────────────────────────────────────────────────


def is_categorical(series: pd.Series, max_categories: int = 20) -> bool:
    """Heuristic: treat as categorical if few unique values, object dtype, or bool."""
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return True
    n_nonnull = series.count()
    if n_nonnull == 0:
        return False
    return series.nunique() <= min(max_categories, int(n_nonnull * 0.05))
