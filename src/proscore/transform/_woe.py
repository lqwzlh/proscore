"""WOE Transformer: convert raw features to Weight-of-Evidence values."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from proscore.binning._base import BinTable


_UNSEEN_STRATEGIES = frozenset({"worst", "most_common", "missing", "zero"})


class WOETransformer:
    """
    Convert raw features to WOE values using a fitted binning result.

    Parameters
    ----------
    unseen_strategy : str
        How to handle categories seen at *transform* time that were not
        present during *fit*:

        - ``"worst"`` (default) — use the WOE of the bin with the highest
          bad rate (most conservative for risk modelling).
        - ``"most_common"`` — use the WOE of the bin with the largest sample.
        - ``"missing"`` — use the WOE of the missing-value bin (if one
          exists; otherwise WOE=0).
        - ``"zero"`` — assign WOE=0 (neutral, no directional signal).
    """

    def __init__(self, unseen_strategy: str = "worst"):
        if unseen_strategy not in _UNSEEN_STRATEGIES:
            raise ValueError(
                f"Unknown unseen_strategy: {unseen_strategy!r}. "
                f"Valid options: {sorted(_UNSEEN_STRATEGIES)}"
            )
        self.unseen_strategy = unseen_strategy

        self._bin_tables: dict[str, BinTable] = {}
        self._woe_map: dict[str, dict] = {}       # col → {interval/value: woe}
        self._special_woe: dict[str, dict] = {}    # col → {special_value: woe}
        self._missing_woe: dict[str, float] = {}   # col → missing_woe
        self._unseen_woe: dict[str, float] = {}    # col → fallback WOE
        self._fitted = False

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(self, bin_tables: dict[str, BinTable]) -> WOETransformer:
        """
        Build WOE mappings from a pre-fitted binning result.

        Parameters
        ----------
        bin_tables : dict[str, BinTable]
            The ``bin_table_`` attribute from a fitted :class:`Binning`
            or :class:`BinningProcess` instance.
        """
        self._bin_tables = bin_tables
        self._woe_map = {}
        self._special_woe = {}
        self._missing_woe = {}
        self._unseen_woe = {}

        for col, bt in bin_tables.items():
            if bt.dtype == "continuous":
                self._woe_map[col] = self._build_continuous_map(bt)
            else:
                self._woe_map[col] = self._build_categorical_map(bt)

            # Special values → WOE
            spec = {}
            for sv in bt.special_values:
                for b in bt.bins:
                    if b.bin_label == str(sv) or b.min_val == sv:
                        spec[sv] = b.woe
                        break
            self._special_woe[col] = spec

            # Missing → WOE
            miss_woe = 0.0
            if bt.has_missing:
                for b in bt.bins:
                    if b.bin_label == "missing" or (b.min_val is None and b.max_val is None):
                        miss_woe = b.woe
                        break
            self._missing_woe[col] = miss_woe

            # Unseen category fallback
            self._unseen_woe[col] = self._resolve_unseen_woe(bt, miss_woe)

        self._fitted = True
        return self

    def _build_continuous_map(self, bt: BinTable) -> dict:
        """Build {(lo, hi, bin_no): woe} for a continuous variable."""
        mapping = {}
        for b in bt.bins:
            # Skip special-value bins (min==max) and missing bin
            if b.bin_label in ("missing",):
                continue
            if b.min_val is not None and b.max_val is not None and b.min_val == b.max_val:
                continue  # special-value bin
            lo = -float("inf") if b.min_val is None or np.isneginf(float(b.min_val)) else float(b.min_val)
            hi = float("inf") if b.max_val is None or np.isposinf(float(b.max_val)) else float(b.max_val)
            mapping[(lo, hi, b.bin_no)] = b.woe
        return mapping

    def _build_categorical_map(self, bt: BinTable) -> dict:
        """Build {original_value: woe} for a categorical variable.

        Uses ``cat_mapping`` (value → bin_no) and bin WOE values to build
        a direct lookup, preserving original value types.
        """
        mapping = {}
        # Build bin_no → woe lookup from normal bins
        woe_by_bin: dict[int, float] = {}
        for b in bt.bins:
            if b.bin_label == "missing":
                continue
            woe_by_bin[b.bin_no] = b.woe
        # Map each original value to its bin's WOE
        for val, bin_no in bt.cat_mapping.items():
            if bin_no in woe_by_bin:
                mapping[val] = woe_by_bin[bin_no]
        return mapping

    def _resolve_unseen_woe(self, bt: BinTable, miss_woe: float) -> float:
        """Compute the fallback WOE for unseen categories."""
        strategy = self.unseen_strategy

        if strategy == "zero":
            return 0.0

        if strategy == "missing":
            return miss_woe

        # Collect normal bins (exclude special-value and missing bins)
        normal = [
            b for b in bt.bins
            if b.bin_label != "missing"
            and not (b.min_val is not None and b.max_val is not None and b.min_val == b.max_val)
        ]
        if not normal:
            return miss_woe

        if strategy == "worst":
            return max(normal, key=lambda b: b.bad_rate).woe

        if strategy == "most_common":
            return max(normal, key=lambda b: b.count).woe

        return 0.0

    # ── transform ──────────────────────────────────────────────────────────

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert each column in *X* to WOE values.

        Columns not present in the fitted binning are passed through unchanged.
        """
        _check_fitted(self)
        result = X.copy()
        for col, bt in self._bin_tables.items():
            if col not in result.columns:
                continue
            result[col] = self._transform_one(result[col], col, bt)
        return result

    def _transform_one(
        self, series: pd.Series, col: str, bt: BinTable
    ) -> pd.Series:
        out = pd.Series(np.nan, index=series.index, dtype=float)

        if bt.dtype == "continuous":
            out = self._transform_continuous(series, col, bt)
        else:
            out = self._transform_categorical(series, col, bt)

        # Special values
        for sv, woe_val in self._special_woe.get(col, {}).items():
            out[series == sv] = woe_val

        # Missing
        miss_woe = self._missing_woe.get(col, 0.0)
        out[series.isna()] = miss_woe

        return out

    def _transform_continuous(
        self, series: pd.Series, col: str, bt: BinTable
    ) -> pd.Series:
        """Map via pd.cut → bin_no → WOE."""
        if not bt.cutoffs:
            # Single bin
            woe_map = self._woe_map.get(col, {})
            woe = next(iter(woe_map.values()), 0.0) if woe_map else 0.0
            return pd.Series(woe, index=series.index, dtype=float)

        bins = [-float("inf")] + bt.cutoffs + [float("inf")]
        bin_idx = pd.cut(series, bins=bins, labels=False, right=True)

        # Build bin_no → woe lookup
        woe_by_bin = {
            bin_tup[2]: w
            for bin_tup, w in self._woe_map.get(col, {}).items()
        }
        return bin_idx.map(woe_by_bin).astype(float)

    def _transform_categorical(
        self, series: pd.Series, col: str, bt: BinTable
    ) -> pd.Series:
        """Map via woe_map (value → WOE), with unseen fallback."""
        woe_map = self._woe_map.get(col, {})  # {original_value: woe}
        unseen_woe = self._unseen_woe.get(col, 0.0)

        def _map(val: Any) -> float:
            if isinstance(val, float) and np.isnan(val):
                return np.nan
            return woe_map.get(val, unseen_woe)

        return series.apply(_map).astype(float)

    def fit_transform(
        self, bin_tables: dict[str, BinTable], X: pd.DataFrame
    ) -> pd.DataFrame:
        """Fit, then transform."""
        self.fit(bin_tables)
        return self.transform(X)

    # ── properties ─────────────────────────────────────────────────────────

    @property
    def bin_tables_(self) -> dict[str, BinTable]:
        """The fitted binning tables."""
        _check_fitted(self)
        return self._bin_tables

    @property
    def woe_map_(self) -> dict[str, dict]:
        """Per-column WOE mapping (interval or value → WOE)."""
        _check_fitted(self)
        return self._woe_map


# ── helpers ───────────────────────────────────────────────────────────────


def _check_fitted(obj) -> None:
    if not obj._fitted:
        raise RuntimeError("Call fit() before using this property or transform().")
