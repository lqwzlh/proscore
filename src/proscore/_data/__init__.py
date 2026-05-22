"""Data loading and validation utilities for ProScore."""

from __future__ import annotations

import numpy as np
import pandas as pd


class DataReader:
    """
    Load and validate a DataFrame for the scorecard pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target : str
        Target column name.
    id_col : str, optional
        Primary-key / identifier column (excluded from modelling).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        id_col: str | None = None,
    ):
        if target not in df.columns:
            raise KeyError(f"Target column {target!r} not found in DataFrame")
        if id_col is not None and id_col not in df.columns:
            raise KeyError(f"ID column {id_col!r} not found in DataFrame")

        self.df = df
        self.target = target
        self.id_col = id_col

    @property
    def features_(self) -> list[str]:
        """Column names available for modelling (excludes target and id)."""
        skip = {self.target}
        if self.id_col:
            skip.add(self.id_col)
        return [c for c in self.df.columns if c not in skip]

    @property
    def X(self) -> pd.DataFrame:
        """Feature DataFrame (excludes target and id)."""
        return self.df[self.features_]

    @property
    def y(self) -> pd.Series:
        """Target Series."""
        return self.df[self.target]

    @property
    def shape(self) -> tuple[int, int]:
        """(n_rows, n_features) — *n_features* excludes *target* and *id_col*."""
        return (len(self.df), len(self.features_))

    def summary(self) -> pd.DataFrame:
        """Quick overview: dtype, missing rate, unique count per column."""
        rows = []
        for col in self.features_:
            s = self.df[col]
            rows.append({
                "variable": col,
                "dtype": str(s.dtype),
                "n_missing": int(s.isna().sum()),
                "missing_pct": round(s.isna().mean() * 100, 2),
                "n_unique": int(s.nunique()),
            })
        return pd.DataFrame(rows).sort_values("missing_pct", ascending=False).reset_index(drop=True)

    def __repr__(self) -> str:
        return (
            f"DataReader(n_rows={len(self.df)}, n_features={len(self.features_)}, "
            f"target={self.target!r}, id={self.id_col!r})"
        )

    # ── factory ────────────────────────────────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        path: str,
        target: str,
        id_col: str | None = None,
        **kwargs,
    ) -> DataReader:
        """Create a DataReader from a CSV file."""
        df = pd.read_csv(path, **kwargs)
        return cls(df, target=target, id_col=id_col)
