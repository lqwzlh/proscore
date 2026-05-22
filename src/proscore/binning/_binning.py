"""Binning orchestrator: algorithm dispatch, adjustment, IV/WOE, transform."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from proscore.binning._base import BinRecord, BinTable
from proscore.binning._categorical import badrate_merge, freq_merge, woe_per_value
from proscore.binning._chi import chi_binning
from proscore.binning._distance import distance_binning
from proscore.binning._frequency import frequency_binning
from proscore.binning._tree import tree_binning
from proscore.binning._adjust import monotonicity_check, trend_adjust, uv_check
from proscore.binning._woe import calc_iv_woe, normalize_woe
from proscore.utils import is_categorical, require_unique_column_labels


_VALID_METHODS = frozenset({"chi", "frequency", "distance", "tree", "optimal"})

_ALGORITHMS = {
    "chi": chi_binning,
    "frequency": frequency_binning,
    "distance": distance_binning,
    "tree": tree_binning,
}

_VALID_CATEGORICAL_MODES = frozenset({"woe_per_value", "badrate_merge", "freq_merge", "custom"})

_CATEGORICAL_MODES = {
    "woe_per_value": woe_per_value,
    "badrate_merge": badrate_merge,
    "freq_merge": freq_merge,
}


class Binning:
    """
    Bin continuous and categorical features for scorecard development.

    Parameters
    ----------
    method : str
        Binning algorithm: ``"chi"``, ``"frequency"``, ``"distance"``, or
        ``"tree"``.  ``"optimal"`` requires the optional ``optbinning``
        package and is not yet implemented.
    n_bins : int
        Target number of bins.
    min_bin_pct : float
        Minimum fraction of total observations per bin.  Bins below this
        threshold are merged during adjustment.
    min_woe_diff : float
        Minimum WOE difference between adjacent bins.  Pairs below this
        threshold are merged.
    monotonic : bool, int, str, or None
        Trend constraint on bad-rate across bins:
        ``None`` / ``False`` / ``0`` = no constraint;
        ``True`` / ``1`` / ``"increasing"`` / ``"ascending"`` = increasing;
        ``-1`` / ``2`` / ``"decreasing"`` / ``"descending"`` = decreasing;
        ``3`` / ``"u"`` / ``"valley"`` = U-shape (valley in middle);
        ``4`` / ``"inverted_u"`` / ``"peak"`` = inverted-U (peak in middle).
        A warning is issued when the final bins do not match this preset.
    confidence_val : float
        Chi-square confidence threshold (only used when *method* is
        ``"chi"``).
    categorical_mode : str
        Strategy for categorical features: ``"woe_per_value"``,
        ``"badrate_merge"``, or ``"freq_merge"``.
    special_values : dict, optional
        ``{column_name: [values]}`` — values treated as separate bins
        (excluded from the main binning algorithm).
    skip_values : dict, optional
        ``{column_name: [values]}`` — values excluded from binning entirely.
    manual_cutoffs : dict, optional
        ``{column_name: [cut_points]}`` — expert cut points (skips the
        binning algorithm).
    adjust_shape : bool
        When True (default), enforce monotonicity / UV-shape detection and
        merge undersized bins.
    min_nobs : int or float, default 30
        Minimum number of non-null observations required to attempt binning.
        Columns with fewer non-null values are skipped (``_fit_one`` returns
        ``None``).  If a float in (0, 1), it is interpreted as a fraction of
        total rows.
    """

    def __init__(
        self,
        method: str = "chi",
        n_bins: int = 10,
        min_bin_pct: float = 0.05,
        min_woe_diff: float = 0.1,
        monotonic: bool | int | str | None = None,
        confidence_val: float = 3.841,
        categorical_mode: str = "woe_per_value",
        special_values: dict | None = None,
        skip_values: dict | None = None,
        manual_cutoffs: dict | None = None,
        adjust_shape: bool = True,
        max_categories: int = 20,
        min_nobs: int | float = 30,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(
                f"Binning got unexpected keyword argument(s): "
                f"{', '.join(repr(k) for k in kwargs)}"
            )

        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown binning method: {method!r}. "
                f"Valid options: {sorted(_VALID_METHODS)}"
            )
        if method == "optimal":
            raise NotImplementedError(
                "optimal binning requires the optbinning package. "
                "Install with: pip install optbinning"
            )

        if categorical_mode not in _VALID_CATEGORICAL_MODES:
            raise ValueError(
                f"Unknown categorical_mode: {categorical_mode!r}. "
                f"Valid options: {sorted(_VALID_CATEGORICAL_MODES)}"
            )
        if categorical_mode == "custom":
            raise NotImplementedError(
                "categorical_mode='custom' requires a user-provided mapping. "
                "Not yet implemented."
            )

        if min_nobs < 0:
            raise ValueError("min_nobs must be non-negative")

        self.method = method
        self.n_bins = n_bins
        self.min_bin_pct = min_bin_pct
        self.min_woe_diff = min_woe_diff
        self.monotonic = monotonic
        self.confidence_val = confidence_val
        self.categorical_mode = categorical_mode
        self.special_values = special_values or {}
        self.skip_values = skip_values or {}
        self.manual_cutoffs = manual_cutoffs or {}
        self.adjust_shape = adjust_shape
        self.max_categories = max_categories
        self.min_nobs = min_nobs

        self._bin_tables: dict[str, BinTable] = {}
        self._fitted = False

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: str | pd.Series) -> Binning:
        """
        Fit binning for every eligible column in *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data (column labels must be unique).
        y : str or pd.Series
            Binary target with values in {0, 1} where 1 = "bad".
            If a string, it names a column in *X*; that column is excluded
            from binning.

        Returns
        -------
        self : Binning
        """
        require_unique_column_labels(X)

        if isinstance(y, str):
            target_col = y
            if target_col not in X.columns:
                raise KeyError(f"target column {target_col!r} not in DataFrame")
            y_series = X[target_col]
            feature_cols = [c for c in X.columns if c != target_col]
        else:
            target_col = None
            y_series = y
            feature_cols = list(X.columns)

        y_vals = np.unique(y_series.dropna().values)
        if len(y_vals) != 2:
            raise ValueError(
                f"Target must be binary (2 unique values), got {len(y_vals)}: "
                f"{sorted(y_vals)[:10]}"
            )
        if not (set(y_vals) <= {0, 1}):
            raise ValueError(
                f"Target values must be in {{0, 1}} (1 = bad). Got: {sorted(y_vals)}"
            )

        self._bin_tables = {}

        for col in feature_cols:
            skip_vals = set(self.skip_values.get(col, []))
            bt = self._fit_one(X[col], y_series, col, skip_vals)
            if bt is not None:
                self._bin_tables[col] = bt

        self._fitted = True
        return self

    def _fit_one(
        self,
        series: pd.Series,
        y: pd.Series,
        col: str,
        skip_vals: set,
    ) -> BinTable | None:
        # Early exit for extremely sparse columns
        n_nonnull = int(series.count())
        min_nobs = self.min_nobs
        if isinstance(min_nobs, float) and 0 < min_nobs < 1:
            threshold = max(1, int(len(series) * min_nobs))
        else:
            threshold = int(min_nobs)
        if n_nonnull < threshold:
            return None

        cat = is_categorical(series, self.max_categories)
        special_vals = set(self.special_values.get(col, []))

        # Separate special / missing / clean
        not_null = series.notna()
        is_special = series.isin(special_vals)
        clean_mask = not_null & ~is_special
        for sv in skip_vals:
            clean_mask = clean_mask & (series != sv)

        X_clean = series[clean_mask].values
        y_clean = y[clean_mask].values
        total_n = len(series)

        if len(X_clean) == 0:
            return None

        assign_cuts = self.manual_cutoffs.get(col)

        if cat:
            bt = self._fit_categorical(X_clean, y_clean, col, total_n)
        else:
            bt = self._fit_numeric(X_clean, y_clean, col, total_n, assign_cuts)

        # --- add special-value bins ---
        for sv in sorted(special_vals):
            mask = series == sv
            if mask.sum() == 0:
                continue
            bad = int(y[mask].sum())
            good = int(mask.sum() - bad)
            bt.bins.append(
                BinRecord(
                    bin_no=len(bt.bins),
                    min_val=sv,
                    max_val=sv,
                    count=bad + good,
                    count_bad=bad,
                    count_good=good,
                    bad_rate=bad / (bad + good) if (bad + good) > 0 else 0.0,
                    woe=0.0,
                    iv=0.0,
                    bin_label=str(sv),
                )
            )
            bt.special_values.append(sv)

        # --- add missing bin ---
        null_mask = series.isna()
        if null_mask.any():
            bad = int(y[null_mask].sum())
            good = int(null_mask.sum() - bad)
            bt.has_missing = True
            bt.bins.append(
                BinRecord(
                    bin_no=len(bt.bins),
                    min_val=None,
                    max_val=None,
                    count=bad + good,
                    count_bad=bad,
                    count_good=good,
                    bad_rate=bad / (bad + good) if (bad + good) > 0 else 0.0,
                    woe=0.0,
                    iv=0.0,
                    bin_label="missing",
                )
            )

        # --- final WOE / IV for all bins ---
        _recalc_woe_iv(bt)

        # --- trend matching check ---
        preset = self._trend_flag()
        bt.trend_preset = preset
        if preset != 0:
            bt.trend_match = (bt.monotonic == preset)
            if not bt.trend_match:
                if col in self.manual_cutoffs:
                    reason = "manual_cutoffs provided"
                elif not self.adjust_shape:
                    reason = "adjust_shape=False"
                else:
                    reason = "data may not support the requested shape"
                warnings.warn(
                    f"Variable '{col}': final bins trend is {_trend_name(bt.monotonic)} "
                    f"(code={bt.monotonic}), but monotonic={self.monotonic!r} "
                    f"({_trend_name(preset)}) was preset. ({reason})",
                    stacklevel=2,
                )

        self._bin_tables[col] = bt
        return bt

    def _fit_categorical(
        self,
        X: np.ndarray,
        y: np.ndarray,
        col: str,
        total_n: int,
    ) -> BinTable:
        strat = _CATEGORICAL_MODES[self.categorical_mode]
        mapping = strat(pd.Series(X), pd.Series(y), min_bin_pct=self.min_bin_pct)

        # Build regroup from mapping: each bin → {boundary, bad, good}
        n_bins = len(set(mapping.values()))
        regroup = np.zeros((n_bins, 3), dtype=np.float64)
        # Also track which values belong to each bin (for labels and transform)
        bin_values: dict[int, list] = {i: [] for i in range(n_bins)}
        for val, bin_no in mapping.items():
            mask = X == val
            regroup[bin_no, 1] += y[mask].sum()  # bad
            regroup[bin_no, 2] += (y[mask] == 0).sum()  # good
            regroup[bin_no, 0] = float(bin_no)
            bin_values[bin_no].append(val)

        iv_total, iv_bins, woe_bins = calc_iv_woe(regroup)

        bins = []
        for i in range(regroup.shape[0]):
            vals = bin_values.get(i, [])
            label = ", ".join(str(v) for v in vals[:5])
            if len(vals) > 5:
                label += f", ... (+{len(vals) - 5})"
            bins.append(
                BinRecord(
                    bin_no=i,
                    min_val=None,
                    max_val=None,
                    count=int(regroup[i, 1] + regroup[i, 2]),
                    count_bad=int(regroup[i, 1]),
                    count_good=int(regroup[i, 2]),
                    bad_rate=(
                        regroup[i, 1] / (regroup[i, 1] + regroup[i, 2])
                        if (regroup[i, 1] + regroup[i, 2]) > 0
                        else 0.0
                    ),
                    woe=float(woe_bins[i]),
                    iv=float(iv_bins[i]),
                    bin_label=label,
                )
            )

        return BinTable(
            var=col,
            bins=bins,
            cutoffs=[],
            iv_total=float(iv_total),
            method=f"categorical_{self.categorical_mode}",
            n_bins=len(bins),
            monotonic=0,
            dtype="categorical",
            cat_mapping=mapping,
        )

    def _fit_numeric(
        self,
        X: np.ndarray,
        y: np.ndarray,
        col: str,
        total_n: int,
        assign_cuts: list[float] | None,
    ) -> BinTable:
        # Get cutoffs
        if assign_cuts is not None:
            cutoffs = sorted(assign_cuts)
        elif self.method in _ALGORITHMS:
            algo = _ALGORITHMS[self.method]
            if self.method == "chi":
                cutoffs = algo(X, y, self.n_bins, confidence_val=self.confidence_val)
            elif self.method == "tree":
                cutoffs = algo(X, y, self.n_bins)
            else:
                cutoffs = algo(X, self.n_bins)
        else:
            raise ValueError(f"Unknown binning method: {self.method}")

        if len(cutoffs) == 0:
            # Fallback: single bin
            cutoffs = []

        # Digitize and build regroup
        if cutoffs:
            binned = np.digitize(X, cutoffs, right=True)
        else:
            binned = np.zeros(len(X), dtype=int)

        n_bins = len(cutoffs) + 1
        regroup = np.zeros((n_bins, 3), dtype=np.float64)
        for i in range(n_bins):
            mask = binned == i
            regroup[i, 1] = y[mask].sum()
            regroup[i, 2] = (y[mask] == 0).sum()
            regroup[i, 0] = cutoffs[i] if i < len(cutoffs) else float("inf")

        # Apply adjustment (skip when user provided manual cutoffs)
        if self.adjust_shape and assign_cuts is None:
            regroup, cutoffs, is_valid, real_trend = trend_adjust(
                regroup,
                total_n,
                min_bin_pct=self.min_bin_pct,
                min_woe_diff=self.min_woe_diff,
                trend_flag=self._trend_flag(),
            )
        else:
            # Detect actual trend from the un-adjusted regroup (manual cutoffs)
            total = regroup[:, 1] + regroup[:, 2]
            br = np.divide(regroup[:, 1], total, out=np.zeros_like(total), where=total > 0)
            real_trend = monotonicity_check(br)
            if not real_trend:
                real_trend = uv_check(regroup)

        iv_total, iv_bins, woe_bins = calc_iv_woe(regroup)

        bins = []
        for i in range(regroup.shape[0]):
            lo = cutoffs[i - 1] if i > 0 else -float("inf")
            hi = cutoffs[i] if i < len(cutoffs) else float("inf")
            count = int(regroup[i, 1] + regroup[i, 2])
            bad = int(regroup[i, 1])
            good = int(regroup[i, 2])
            bins.append(
                BinRecord(
                    bin_no=i,
                    min_val=lo,
                    max_val=hi,
                    count=count,
                    count_bad=bad,
                    count_good=good,
                    bad_rate=bad / (bad + good) if (bad + good) > 0 else 0.0,
                    woe=float(woe_bins[i]),
                    iv=float(iv_bins[i]),
                    bin_label=f"({lo:.4f}, {hi:.4f}]" if np.isfinite(hi) else f"({lo:.4f}, inf]",
                )
            )

        return BinTable(
            var=col,
            bins=bins,
            cutoffs=[float(c) for c in cutoffs],
            iv_total=float(iv_total),
            method=self.method,
            n_bins=len(bins),
            monotonic=real_trend,
            dtype="continuous",
        )

    # ── transform ──────────────────────────────────────────────────────────

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert each feature to bin indices (0-based).

        Special-value and missing bins receive indices beyond the main bins.
        """
        _check_fitted(self)
        result = X.copy()
        for col, bt in self._bin_tables.items():
            if col not in result.columns:
                continue
            result[col] = self._transform_one(result[col], bt)
        return result

    def _transform_one(self, series: pd.Series, bt: BinTable) -> pd.Series:
        return apply_bin_index(series, bt)

    def fit_transform(self, X: pd.DataFrame, y: str | pd.Series) -> pd.DataFrame:
        """Fit binning, then transform *X*."""
        self.fit(X, y)
        return self.transform(X)

    # ── properties ─────────────────────────────────────────────────────────

    @property
    def bin_table_(self) -> dict[str, BinTable]:
        """Mapping from column name to :class:`BinTable`."""
        _check_fitted(self)
        return self._bin_tables

    @property
    def iv_(self) -> pd.DataFrame:
        """DataFrame of per-variable IV."""
        _check_fitted(self)
        return pd.DataFrame(
            [
                {"variable": col, "iv": bt.iv_total}
                for col, bt in self._bin_tables.items()
            ]
        ).sort_values("iv", ascending=False).reset_index(drop=True)

    @property
    def cutoffs_(self) -> dict[str, list[float]]:
        """Mapping from column name to cut points."""
        _check_fitted(self)
        return {col: bt.cutoffs for col, bt in self._bin_tables.items()}

    @property
    def woe_(self) -> dict[str, dict[int, float]]:
        """Mapping from column name to ``{bin_no: woe}``."""
        _check_fitted(self)
        return {
            col: {b.bin_no: b.woe for b in bt.bins}
            for col, bt in self._bin_tables.items()
        }

    # ── helpers ────────────────────────────────────────────────────────────

    def _trend_flag(self) -> int:
        """Normalise the *monotonic* argument to an internal trend code (0–4)."""
        m = self.monotonic
        if m is None or m is False or m == 0:
            return 0
        if m is True or m == 1:
            return 1
        if m == -1 or m == 2:
            return 2
        if m == 3:
            return 3
        if m == 4:
            return 4
        if isinstance(m, str):
            s = m.strip().lower()
            if s in ("increasing", "ascending"):
                return 1
            if s in ("decreasing", "descending"):
                return 2
            if s in ("u", "valley"):
                return 3
            if s in ("inverted_u", "inverted-u", "inverted u", "peak"):
                return 4
        raise ValueError(
            f"Unknown monotonic value: {m!r}. Expected one of: "
            "None, True/False, 0/1/2/3/4, 'increasing', 'decreasing', 'u', 'inverted_u'."
        )


# ── Module-level transform utility ──────────────────────────────────────────


def apply_bin_index(series: pd.Series, bt: BinTable) -> pd.Series:
    """
    Convert a feature column to bin indices (0-based) using a fitted BinTable.

    This is a pure function usable by both :class:`Binning` and
    :class:`BinningProcess` without requiring a class instance.
    """
    n_main = bt.n_bins
    out = pd.Series(np.full(len(series), np.nan, dtype=float), index=series.index)

    if bt.dtype == "continuous":
        if bt.cutoffs:
            out = pd.cut(
                series,
                bins=[-float("inf")] + bt.cutoffs + [float("inf")],
                labels=False,
                right=True,
            ).astype(float)
        else:
            out = pd.Series(np.zeros(len(series), dtype=float), index=series.index)

    elif bt.dtype == "categorical":
        if bt.cat_mapping:
            for val, bin_no in bt.cat_mapping.items():
                out[series == val] = float(bin_no)

    # Special values → bin indices beyond the main bins
    for i, sv in enumerate(bt.special_values):
        out[series == sv] = n_main + i

    # Missing → last bin
    if bt.has_missing:
        out[series.isna()] = n_main + len(bt.special_values)

    return out


# ── BinningProcess ─────────────────────────────────────────────────────────


class BinningProcess:
    """
    Batch-process all features with per-feature configuration overrides.

    Parameters
    ----------
    feature_config : dict, optional
        ``{column_name: {param: value, ...}}`` — per-feature overrides for
        any parameter accepted by :class:`Binning`.
    default_method : str
        Default binning algorithm.
    default_n_bins : int
        Default target number of bins.
    **default_kwargs
        Additional default parameters forwarded to :class:`Binning`.
    """

    def __init__(
        self,
        feature_config: dict | None = None,
        default_method: str = "chi",
        default_n_bins: int = 10,
        **default_kwargs,
    ):
        self.feature_config = feature_config or {}
        self.default_method = default_method
        self.default_n_bins = default_n_bins
        self.default_kwargs = default_kwargs
        self._bin_tables: dict[str, BinTable] = {}
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: str | pd.Series) -> BinningProcess:
        """Fit binning for every column, respecting *feature_config* overrides."""
        if isinstance(y, str):
            target_col = y
            feature_cols = [c for c in X.columns if c != target_col]
        else:
            target_col = None
            feature_cols = list(X.columns)

        self._bin_tables = {}

        for col in feature_cols:
            cfg = self.feature_config.get(col, {})
            kw = {k: v for k, v in cfg.items() if k not in ("method", "n_bins")}
            # Binning expects special_values / skip_values as {col: [vals]} dicts,
            # but feature_config passes them as plain lists. Wrap per-column.
            for dict_param in ("special_values", "skip_values"):
                if dict_param in kw and not isinstance(kw[dict_param], dict):
                    kw[dict_param] = {col: kw[dict_param]}
            binner = Binning(
                method=cfg.get("method", self.default_method),
                n_bins=cfg.get("n_bins", self.default_n_bins),
                **{**self.default_kwargs, **kw},
            )
            binner.fit(X[[col] + ([target_col] if target_col else [])], y)
            if col in binner._bin_tables:
                self._bin_tables[col] = binner._bin_tables[col]

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform all features using their fitted binning."""
        _check_fitted(self)
        result = X.copy()
        for col, bt in self._bin_tables.items():
            if col not in result.columns:
                continue
            result[col] = apply_bin_index(result[col], bt)
        return result

    def fit_transform(self, X: pd.DataFrame, y: str | pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    @property
    def bin_table_(self) -> dict[str, BinTable]:
        _check_fitted(self)
        return self._bin_tables

    @property
    def iv_(self) -> pd.DataFrame:
        _check_fitted(self)
        return pd.DataFrame(
            [
                {"variable": col, "iv": bt.iv_total}
                for col, bt in self._bin_tables.items()
            ]
        ).sort_values("iv", ascending=False).reset_index(drop=True)


# ── internal helpers ───────────────────────────────────────────────────────


def _trend_name(code: int) -> str:
    """Human-readable name for a trend code."""
    return {0: "none", 1: "increasing", 2: "decreasing", 3: "U-shape", 4: "inverted-U"}.get(
        code, f"unknown({code})"
    )


def _check_fitted(obj) -> None:
    if not obj._fitted:
        raise RuntimeError("Call fit() before using this property.")


def _recalc_woe_iv(bt: BinTable) -> None:
    """Update WOE/IV for all bins in *bt* and set *iv_total*."""
    bad = np.array([b.count_bad for b in bt.bins], dtype=np.float64)
    good = np.array([b.count_good for b in bt.bins], dtype=np.float64)
    total_bad = bad.sum()
    total_good = good.sum()

    if total_bad == 0 or total_good == 0:
        bt.iv_total = 0.0
        for b in bt.bins:
            b.woe = 0.0
            b.iv = 0.0
        return

    iv_total = 0.0
    for b in bt.bins:
        bad_dist = b.count_bad / total_bad if total_bad > 0 else 0.0
        good_dist = b.count_good / total_good if total_good > 0 else 0.0
        if bad_dist > 0 and good_dist > 0:
            b.woe = round(normalize_woe(float(np.log(bad_dist / good_dist))), 6)
        elif bad_dist > 0:
            b.woe = round(normalize_woe(20.0), 6)
        elif good_dist > 0:
            b.woe = round(normalize_woe(-20.0), 6)
        else:
            b.woe = 0.0
        b.iv = round(float((bad_dist - good_dist) * b.woe), 6)
        iv_total += b.iv
    bt.iv_total = round(float(iv_total), 6)


def bin_table_to_dataframe(bin_table: dict[str, BinTable]) -> pd.DataFrame:
    """Flatten a ``bin_table_`` dict into a per-bin DataFrame for inspection."""
    rows: list[dict[str, Any]] = []
    for var, bt in bin_table.items():
        for b in bt.bins:
            rows.append({
                "variable": var,
                "bin_no": b.bin_no,
                "bin_label": b.bin_label,
                "count": b.count,
                "count_bad": b.count_bad,
                "count_good": b.count_good,
                "bad_rate": round(b.bad_rate, 6),
                "woe": b.woe,
                "iv": b.iv,
                "iv_total": bt.iv_total,
                "method": bt.method,
                "monotonic": bt.monotonic,
                "trend_match": bt.trend_match,
            })
    if not rows:
        return pd.DataFrame(
            columns=[
                "variable", "bin_no", "bin_label", "count", "count_bad",
                "count_good", "bad_rate", "woe", "iv", "iv_total",
                "method", "monotonic", "trend_match",
            ]
        )
    out = pd.DataFrame(rows)
    out["variable"] = pd.Categorical(
        out["variable"], categories=list(bin_table.keys()), ordered=True
    )
    return out.sort_values(["variable", "bin_no"]).reset_index(drop=True)
