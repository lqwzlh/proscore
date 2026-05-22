"""Feature Filter module for initial screening.

Performs quality-based feature selection using missing rate, one-value rate,
IV, single-variable AUC, PSI, correlation, and VIF.  Binning integration is
available for accurate PSI / IV calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from proscore.binning import BinningProcess
from proscore.inspect._quality import _calc_iv
from proscore.selection._screen import FeatureScreenWarning
from proscore.utils import is_categorical
from proscore.utils._psi import psi_from_distributions

if TYPE_CHECKING:
    from proscore.binning._base import BinTable


class Filter:
    """Feature pre-screener.

    Applies sequential filters: missing rate → one-value rate → IV →
    single-variable AUC → PSI → correlation → VIF → top-N by IV.

    All thresholds are parameterised with defaults suitable for scorecard
    development.  When a pre-fitted *bin_table* is provided, IV and PSI use
    model binning; otherwise IV falls back to the same equal-frequency /
    categorical rules as :func:`~proscore.inspect.quality`.
    """

    def __init__(
        self,
        max_missing_rate: float = 0.8,
        max_one_value_rate: float = 0.95,
        iv_range: tuple[float, float | None] | None = (0.02, None),
        min_auc: float | None = None,
        max_psi: float | None = None,
        max_corr: float = 0.7,
        max_vif: float | None = None,
        n_selected: int | None = None,
    ) -> None:
        """Initialise filter thresholds.

        Args:
            max_missing_rate: Drop features whose missing fraction exceeds this.
            max_one_value_rate: Drop features whose most-common value exceeds
                this fraction of non-null rows.
            iv_range: ``(low, high)`` — drop features with IV below *low* or
                above *high*.  ``high=None`` means no upper cap.  ``None``
                skips the IV check (typical for coarse :meth:`prefilter`).
            min_auc: Minimum single-variable AUC (computed in-sample; biased).
                ``None`` skips the check.
            max_psi: Maximum PSI (requires *bin_table* or *X_test*).
                ``None`` skips the check.
            max_corr: Drop the weaker (by IV) of each pair exceeding this
                absolute correlation.
            max_vif: Drop features whose VIF exceeds this. ``None`` skips.
            n_selected: Keep at most this many features (by IV rank).
        """
        self.max_missing_rate = max_missing_rate
        self.max_one_value_rate = max_one_value_rate
        self.iv_range = iv_range
        self.min_auc = min_auc
        self.max_psi = max_psi
        self.max_corr = max_corr
        self.max_vif = max_vif
        self.n_selected = n_selected

        self._support: list[str] = []
        self._quality: pd.DataFrame | None = None
        self._iv: pd.DataFrame | None = None
        self._psi_cutoffs: dict[str, list[float]] = {}
        self._bin_table: dict[str, Any] | None = None
        self._drop_reasons: dict[str, list[str]] = {}
        self._n_candidates_in: int = 0
        self._fitted: bool = False

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame | None = None,
        bin_table: dict[str, Any] | None = None,
    ) -> Filter:
        """Run the filter pipeline.

        Args:
            X: Feature DataFrame (raw numeric or WOE-transformed).
            y: Binary target Series (1 = bad).
            X_test: Optional test/OOT DataFrame (same columns as *X*).
                Required for PSI; also used for single-variable AUC when
                provided (result may be biased if *y_test* differs from
                train target — use *AUC* as a coarse screen only).
            bin_table: Pre-fitted ``Binning.bin_table_`` result, a dict
                ``{column: BinTable}``.  When provided, IV and PSI are
                computed from real binning distributions.

        Returns:
            self
        """
        if y is None:
            raise ValueError("y must be provided")

        self._bin_table = bin_table
        features = list(X.columns) if X is not None else []
        self._n_candidates_in = len(features)

        if len(features) == 0:
            warnings.warn(
                "Filter.fit received no feature columns; support_ is empty. "
                "This is a normal outcome when upstream screening removed all "
                "candidates — skip refine and downstream modelling steps.",
                FeatureScreenWarning,
                stacklevel=2,
            )
            self._support = []
            self._drop_reasons = {}
            self._iv = pd.DataFrame()
            self._build_quality(pd.DataFrame(columns=[]))
            self._fitted = True
            return self

        # Pre-compute PSI cutoffs with a single BinningProcess when needed
        if self.max_psi is not None:
            if X_test is None:
                warnings.warn(
                    "max_psi requires test data but none was provided. "
                    "Skipping PSI check.",
                    stacklevel=2,
                )
                self.max_psi = None
            else:
                self._ensure_psi_cutoffs(X, y, features)

        # 1. Missing rate
        missing_out = self._check_missing(X, features)
        features = [f for f in features if f not in missing_out]

        # 2. One-value rate
        one_val_out = self._check_one_value(X, features)
        features = [f for f in features if f not in one_val_out]

        # 3. IV
        iv_out, iv_df = self._check_iv(X, y, features)
        self._iv = iv_df
        features = [f for f in features if f not in iv_out]

        # 4. AUC (single-variable, on train — in-sample, biased but rank-valid)
        if self.min_auc is not None:
            auc_out = self._check_auc(X, y, features)
            features = [f for f in features if f not in auc_out]

        # 5. PSI
        if self.max_psi is not None and X_test is not None:
            psi_out = self._check_psi(X, X_test, features)
            features = [f for f in features if f not in psi_out]

        # 6. Correlation
        if self.max_corr is not None:
            corr_out = self._check_corr(X, features)
            features = [f for f in features if f not in corr_out]

        # 7. VIF
        if self.max_vif is not None:
            vif_out = self._check_vif(X, features)
            features = [f for f in features if f not in vif_out]

        # 8. Top-N by IV
        if self.n_selected is not None and len(features) > self.n_selected:
            if self._iv is not None:
                iv_sorted = self._iv[self._iv["feature"].isin(features)].sort_values(
                    "iv", ascending=False
                )
                keep = set(iv_sorted["feature"].head(self.n_selected))
                features = list(keep)

        self._support = features
        self._collect_all_reasons(X, y, X_test)
        self._build_quality(X)
        self._fitted = True
        return self

    def _collect_all_reasons(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame | None,
    ) -> None:
        """Record every failed check per feature (for the quality table)."""
        all_feats = list(X.columns)
        self._drop_reasons = {f: [] for f in all_feats}
        self._mark_drop(self._check_missing(X, all_feats), "missing")
        self._mark_drop(self._check_one_value(X, all_feats), "one_value")
        if self.iv_range is not None:
            iv_fail, _ = self._check_iv(X, y, all_feats)
            self._mark_drop(iv_fail, "iv")
        if self.min_auc is not None:
            self._mark_drop(self._check_auc(X, y, all_feats), "auc")
        if self.max_psi is not None and X_test is not None:
            self._mark_drop(self._check_psi(X, X_test, all_feats), "psi")
        if self.max_corr is not None:
            self._mark_drop(self._check_corr(X, all_feats), "corr")
        if self.max_vif is not None:
            self._mark_drop(self._check_vif(X, all_feats), "vif")

    def _mark_drop(self, dropped: list[str], reason: str) -> None:
        for f in dropped:
            if reason not in self._drop_reasons[f]:
                self._drop_reasons[f].append(reason)

    # ── individual checks ──────────────────────────────────────────────────

    def _check_missing(self, X: pd.DataFrame, features: list[str]) -> list[str]:
        return [f for f in features if X[f].isna().mean() > self.max_missing_rate]

    def _check_one_value(self, X: pd.DataFrame, features: list[str]) -> list[str]:
        out = []
        for f in features:
            non_null = X[f].dropna()
            if len(non_null) == 0:
                continue
            rate = non_null.value_counts().iloc[0] / len(non_null)
            if rate > self.max_one_value_rate:
                out.append(f)
        return out

    def _check_iv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: list[str],
    ) -> tuple[list[str], pd.DataFrame]:
        """IV from *bin_table* when available; else equal-frequency (quality rules)."""
        bt = self._bin_table or {}
        y_arr = np.asarray(y)
        rows = []
        out = []

        for f in features:
            iv_val = 0.0
            source = "none"
            if isinstance(bt, dict) and f in bt:
                tbl = bt[f]
                if hasattr(tbl, "iv_total"):
                    iv_val = float(tbl.iv_total)
                    source = "bin_table"
            else:
                iv_val = float(_calc_iv(X[f], y_arr, is_categorical(X[f])))
                source = "equalfreq"

            rows.append({"feature": f, "iv": iv_val, "source": source})
            if self.iv_range is not None:
                low, high = self.iv_range
                if iv_val < low or (high is not None and iv_val > high):
                    out.append(f)

        cols = ["feature", "iv", "source"]
        df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame({c: [] for c in cols})
        return out, df

    def _check_auc(
        self, X: pd.DataFrame, y: pd.Series, features: list[str]
    ) -> list[str]:
        """Single-variable in-sample AUC.  Biased — coarse screen only."""
        out = []
        for f in features:
            try:
                auc = roc_auc_score(y, X[f])
            except ValueError:
                auc = 0.5
            if auc < self.min_auc:
                out.append(f)
        return out

    def _ensure_psi_cutoffs(
        self, X: pd.DataFrame, y: pd.Series, features: list[str]
    ) -> None:
        """Compute cutoffs for all features in one pass via BinningProcess."""
        bt = self._bin_table or {}
        need_cutoffs = [
            f
            for f in features
            if f not in self._psi_cutoffs
            and not (isinstance(bt, dict) and f in bt and hasattr(bt[f], "cutoffs"))
        ]
        if not need_cutoffs:
            return

        bp = BinningProcess(
            default_method="frequency",
            default_n_bins=10,
            monotonic=False,
            adjust_shape=False,
        )
        bp.fit(X[need_cutoffs], y)
        for f in need_cutoffs:
            if f in bp.bin_table_ and hasattr(bp.bin_table_[f], "cutoffs"):
                self._psi_cutoffs[f] = list(bp.bin_table_[f].cutoffs)

    def _check_psi(
        self, X: pd.DataFrame, X_test: pd.DataFrame, features: list[str]
    ) -> list[str]:
        """PSI check using pre-computed cutoffs."""
        bt = self._bin_table or {}
        out = []
        for f in features:
            cutoffs: list[float] | None = self._psi_cutoffs.get(f)
            if cutoffs is None and isinstance(bt, dict) and f in bt:
                tbl = bt[f]
                if hasattr(tbl, "cutoffs"):
                    cutoffs = list(tbl.cutoffs)
            if cutoffs is None:
                continue
            psi = _calc_psi_with_cutoffs(X[f], X_test[f], cutoffs)
            if psi > self.max_psi:
                out.append(f)
        return out

    def _check_corr(self, X: pd.DataFrame, features: list[str]) -> list[str]:
        """Drop the weaker (by IV) of each highly-correlated pair."""
        num_feats = [f for f in features if pd.api.types.is_numeric_dtype(X[f])]
        if len(num_feats) < 2:
            return []
        corr_mat = X[num_feats].replace([np.inf, -np.inf], np.nan).corr().abs()
        np.fill_diagonal(corr_mat.values, 0)
        out: set[str] = set()
        for i in range(len(num_feats)):
            for j in range(i + 1, len(num_feats)):
                if corr_mat.iloc[i, j] > self.max_corr:
                    f1, f2 = num_feats[i], num_feats[j]
                    iv1 = (
                        self._iv[self._iv["feature"] == f1]["iv"].values[0]
                        if self._iv is not None
                        else 0.0
                    )
                    iv2 = (
                        self._iv[self._iv["feature"] == f2]["iv"].values[0]
                        if self._iv is not None
                        else 0.0
                    )
                    out.add(f2 if iv1 >= iv2 else f1)
        return list(out)

    def _check_vif(self, X: pd.DataFrame, features: list[str]) -> list[str]:
        """Drop features exceeding the VIF threshold."""
        num_feats = [f for f in features if pd.api.types.is_numeric_dtype(X[f])]
        if len(num_feats) < 2 or self.max_vif is None:
            return []
        Xv = X[num_feats].replace([np.inf, -np.inf], np.nan).dropna()
        Xv["const"] = 1.0
        out = []
        for i, f in enumerate(num_feats):
            vif = variance_inflation_factor(Xv.values, i)
            if vif > self.max_vif:
                out.append(f)
        return out

    # ── quality table ──────────────────────────────────────────────────────

    def _build_quality(self, X: pd.DataFrame) -> None:
        """Build the quality overview table."""
        rows = []
        for f in X.columns:
            miss = X[f].isna().mean()
            non_null = X[f].dropna()
            one_val = (
                non_null.value_counts().iloc[0] / len(non_null)
                if len(non_null) > 0
                else 1.0
            )
            iv_val = 0.0
            iv_src = ""
            if self._iv is not None and "feature" in self._iv.columns and not self._iv.empty:
                match = self._iv[self._iv["feature"] == f]
                if not match.empty:
                    iv_val = match["iv"].values[0]
                    iv_src = match["source"].values[0] if "source" in match.columns else ""
            reasons = self._drop_reasons.get(f, [])
            rows.append({
                "feature": f,
                "missing_rate": miss,
                "one_value_rate": one_val,
                "iv": iv_val,
                "iv_source": iv_src,
                "selected": f in self._support,
                "dropped": len(reasons) > 0,
                "reason": ";".join(reasons),
            })
        cols = [
            "feature", "missing_rate", "one_value_rate", "iv", "iv_source",
            "selected", "dropped", "reason",
        ]
        self._quality = pd.DataFrame(rows, columns=cols if not rows else None)

    # ── properties ─────────────────────────────────────────────────────────

    @property
    def support_(self) -> list[str]:
        """Features that passed all filters."""
        return self._support

    @property
    def iv_(self) -> pd.DataFrame:
        """Per-feature IV table (from bin_table when available)."""
        return self._iv if self._iv is not None else pd.DataFrame()

    @property
    def quality_(self) -> pd.DataFrame:
        """Variable quality overview."""
        return self._quality if self._quality is not None else pd.DataFrame()

    @property
    def exhausted_(self) -> bool:
        """``True`` after :meth:`fit` when no features remain."""
        return self._fitted and len(self._support) == 0

    @property
    def n_candidates_in_(self) -> int:
        """Number of feature columns passed to the last :meth:`fit`."""
        return self._n_candidates_in


# ── module-level helpers ───────────────────────────────────────────────────


def _calc_psi_with_cutoffs(
    train: pd.Series, test: pd.Series, cutoffs: list[float]
) -> float:
    """PSI from cut-points defined on the training distribution."""
    bins = [-np.inf] + cutoffs + [np.inf]
    trn_cnt = pd.cut(train.dropna(), bins=bins, right=False).value_counts(sort=False)
    tst_cnt = pd.cut(test.dropna(), bins=bins, right=False).value_counts(sort=False)
    all_bins = trn_cnt.index.union(tst_cnt.index)
    e_arr = trn_cnt.reindex(all_bins, fill_value=0).to_numpy(dtype=float)
    a_arr = tst_cnt.reindex(all_bins, fill_value=0).to_numpy(dtype=float)
    return psi_from_distributions(e_arr, a_arr)
