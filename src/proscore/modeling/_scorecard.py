"""Scorecard modelling: Logistic Regression fit + scorecard conversion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from proscore.binning._base import BinTable


class ScoreCard:
    """
    Fit a logistic regression on WOE-transformed features and convert to a
    standard scorecard.

    Parameters
    ----------
    odds : float
        Target odds (good:bad ratio) at the base score.  Default 50 means
        the base score is assigned to borrowers whose good:bad odds are 50:1.
    pdo : float
        Points to Double the Odds.  Default 10 means every 10-point drop
        halves the odds (doubles the risk).
    base_score : float
        Score assigned at the target *odds*.  Default 600.
    """

    def __init__(self, odds: float = 50.0, pdo: float = 10.0, base_score: float = 600.0):
        self.odds = odds
        self.pdo = pdo
        self.base_score = base_score

        self._model: sm.Logit | None = None  # type: ignore[valid-type]
        self._features: list[str] = []
        self._score_table: pd.DataFrame | None = None
        self._fitted = False
        self._base_offset: float | None = None

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: str | pd.Series,
        features: list[str] | None = None,
    ) -> ScoreCard:
        """
        Fit a logistic regression on WOE-transformed data.

        Parameters
        ----------
        X : pd.DataFrame
            WOE-transformed features.
        y : str or pd.Series
            Binary target (1 = bad).  If a string, it names a column in *X*;
            that column is excluded from *features*.
        features : list of str, optional
            Features to include.  Defaults to all columns in *X* (except *y*
            when *y* is a column name).

        Notes
        -----
        If the data is perfectly separable (or nearly so), statsmodels may
        raise a ``LinAlgError`` or emit a ``PerfectSeparationWarning``.
        These are deliberately not suppressed — check your features for
        quasi-complete separation before fitting.
        """
        if isinstance(y, str):
            target_col = y
            y_series = X[target_col]
            self._features = features or [c for c in X.columns if c != target_col]
        else:
            y_series = y
            self._features = features or list(X.columns)

        if not self._features:
            raise ValueError("No features to fit. Check that X has columns besides the target.")

        Xc = sm.add_constant(X[self._features], has_constant="add")
        self._model = sm.Logit(y_series, Xc).fit(disp=False, maxiter=200)
        self._fitted = True
        return self

    # ── scorecard ──────────────────────────────────────────────────────────

    def scorecard(self, bin_tables: dict[str, BinTable]) -> pd.DataFrame:
        """
        Convert fitted model coefficients and bin WOE values to score points.

        The total score for a borrower is::

            total = base_offset + Σ bin_points
            base_offset = base_score + factor × (ln(odds) + intercept)

        where ``factor = pdo / ln(2)`` and each bin's points are
        ``-factor × coef × woe`` (higher score = lower risk).

        Parameters
        ----------
        bin_tables : dict[str, BinTable]
            The ``bin_table_`` from a fitted :class:`Binning` instance.  Must
            contain every feature in the model.

        Returns
        -------
        pd.DataFrame
            Columns: ``variable | bin_label | bin_no | count | bad_rate |
            woe | coef | points | is_extra``.  Rows with ``is_extra=True``
            correspond to special-value and missing bins.
        """
        _check_fitted(self)
        if self._model is None:
            raise RuntimeError("Model not fitted.")

        factor = self.pdo / np.log(2)
        # Score = base_offset - factor × Σ(coef × woe)  →  higher score, lower risk
        # base_offset = base_score + factor × (ln(odds) + intercept)
        base_offset = self.base_score + factor * (np.log(self.odds) + float(self._model.params.iloc[0]))
        self._base_offset = base_offset

        rows = []
        for var in self._features:
            coef = float(self._model.params[var])
            bt = bin_tables[var]

            for b in bt.bins:
                # Special-value bins are identified by min_val == max_val (both set
                # to the special value by the binning module).  Missing bins use the
                # label "missing".  Both are tagged is_extra for downstream filtering.
                is_extra = (
                    b.bin_label == "missing"
                    or (b.min_val is not None and b.max_val is not None and b.min_val == b.max_val)
                )
                points = -factor * coef * b.woe
                rows.append({
                    "variable": var,
                    "bin_label": b.bin_label,
                    "bin_no": b.bin_no,
                    "count": b.count,
                    "bad_rate": round(b.bad_rate, 6),
                    "woe": b.woe,
                    "coef": round(coef, 6),
                    "points": round(points, 2),
                    "is_extra": is_extra,
                })

        self._score_table = pd.DataFrame(rows)
        self._score_table["variable"] = pd.Categorical(
            self._score_table["variable"], categories=self._features, ordered=True
        )
        return self._score_table.sort_values(["variable", "bin_no"]).reset_index(drop=True)

    # ── predict ────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Compute scores for new WOE-transformed data.

        Returns a Series of score values (higher = lower risk).

        Uses ``score = base_offset - factor × Σ(coef × woe)`` (same sign as
        :meth:`scorecard` partial points).
        """
        _check_fitted(self)
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        factor = self.pdo / np.log(2)
        if self._base_offset is not None:
            base_offset = self._base_offset
        else:
            base_offset = self.base_score + factor * (np.log(self.odds) + float(self._model.params.iloc[0]))

        score = pd.Series(base_offset, index=X.index, dtype=float)
        for col in self._features:
            coef = float(self._model.params[col])
            score -= factor * coef * X[col]
        return score

    # ── properties ─────────────────────────────────────────────────────────

    @property
    def model_(self) -> sm.Logit:  # type: ignore[valid-type]
        """The fitted statsmodels Logit result."""
        _check_fitted(self)
        return self._model

    @property
    def intercept_(self) -> float:
        """Intercept of the logistic regression."""
        _check_fitted(self)
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return float(self._model.params.iloc[0])

    @property
    def coef_(self) -> dict[str, float]:
        """Coefficient for each feature."""
        _check_fitted(self)
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return {f: float(self._model.params[f]) for f in self._features}

    @property
    def score_table_(self) -> pd.DataFrame | None:
        """Scorecard table (available after calling :meth:`scorecard`)."""
        return self._score_table


def _check_fitted(obj) -> None:
    if not obj._fitted:
        raise RuntimeError("Call fit() before using this property or method.")
