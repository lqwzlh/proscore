"""Stepwise bidirectional feature selection module.

Core differentiator: forward + backward iteration with full parameterized LR checks,
perturbation, source belong control, and force fill.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


class StepwiseSelector:
    """Bidirectional stepwise feature selector.

    Supports forward addition + backward elimination with fully parameterized
    LR checks (set to ``None`` to disable), objective function, source-belong
    control, perturbation, and variable-count constraints.
    """

    def __init__(
        self,
        pvalue_threshold: float | None = 0.05,
        coef_sign: str | None = "positive",
        vif_threshold: float | None = 10.0,
        corr_threshold: float | None = 0.8,
        feature_belong: dict[str, list[str]] | None = None,
        belong_max_pct: float | None = None,
        perturbation: bool = True,
        perturbation_pct: float = 0.1,
        perturbation_add: int = 2,
        max_iter_round: int = 100,
        max_iter_time: int = 600,
        same_round_exit: int = 4,
        r: float = 0.8,
        objective: str | Callable[[dict[str, float]], float] = "ks_reduce",
        goal_threshold: float = 0.01,
        n_min: int = 5,
        n_max: int = 15,
        force_fill: bool = True,
    ) -> None:
        """Initialise the selector.

        Args:
            pvalue_threshold: P-value threshold (``None`` disables the check).
            coef_sign: Coefficient sign constraint — ``'positive'``,
                ``'negative'``, or ``None`` (no constraint).
            vif_threshold: VIF threshold (``None`` disables).
            corr_threshold: Pairwise correlation threshold (``None`` disables).
            feature_belong: Source-belong mapping ``{source: [features]}``.
            belong_max_pct: Maximum fraction of features from a single source.
            perturbation: Enable random perturbation when stuck.
            perturbation_pct: Fraction of current features to drop.
            perturbation_add: Maximum number of new candidates to randomly add
                during each perturbation (0 disables adding).
            max_iter_round: Maximum iteration rounds.
            max_iter_time: Maximum wall-clock seconds.
            same_round_exit: Consecutive unchanged rounds before exiting.
            r: Overfitting penalty weight in the objective (0 = ignore
                overfitting, 1 = penalise heavily).
            objective: ``'ks_reduce'``, ``'auc_reduce'``, or a callable
                ``f(perf_dict) -> float``.
            goal_threshold: Minimum absolute score improvement to accept a trial.
                When using ``'ks'`` / ``'ks_reduce'`` (KS ∈ [0, 1]), a typical
                range is 0.001–0.02.  For ``'auc'`` / ``'auc_reduce'``
                (AUC ∈ [0.5, 1]), 0.0005–0.01 is typical.  Adjust downward for
                custom objectives with smaller score ranges.
            n_min: Minimum number of variables in a valid solution.
            n_max: Maximum number of variables in a valid solution.
            force_fill: When ``True``, pad up to *n_min* after iteration.
        """
        self.pvalue_threshold = pvalue_threshold
        self.coef_sign = coef_sign
        self.vif_threshold = vif_threshold
        self.corr_threshold = corr_threshold
        self.feature_belong = feature_belong or {}
        self.belong_max_pct = belong_max_pct
        self.perturbation = perturbation
        self.perturbation_pct = perturbation_pct
        self.perturbation_add = perturbation_add
        self.max_iter_round = max_iter_round
        self.max_iter_time = max_iter_time
        self.same_round_exit = same_round_exit
        self.r = r
        self.objective = objective
        self.goal_threshold = goal_threshold
        self.n_min = n_min
        self.n_max = n_max
        self.force_fill = force_fill

        self._support: list[str] = []
        self._record: dict[int, dict[str, Any]] = {}
        self._best_perf: dict[str, Any] = {}
        self._model: Any = None
        self._best_score: float = -np.inf
        self._iter_count = 0
        self._lr_cache: dict[tuple, Any] = {}  # frozenset(vars) → fitted model

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        candidates: list[str] | None = None,
        force_in: list[str] | None = None,
        sort_df: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ) -> StepwiseSelector:
        """Run bidirectional stepwise selection.

        Args:
            X: WOE-transformed feature DataFrame (train).
            y: Binary target Series (train; 1 = bad).
            candidates: Candidate features (default: all cols except *force_in*).
            force_in: Features that must stay in every trial.
            sort_df: Pre-sort reference table (columns: ``feature`` + metric).
            X_test: WOE-transformed test/OOT DataFrame (optional).
            y_test: Test target Series (required when *X_test* is given).

        Returns:
            self
        """
        if X_test is not None and y_test is None:
            raise ValueError("y_test must be provided when X_test is given")

        # When no test set, _reduce objectives cannot compute ks_reduce/auc_reduce
        if (X_test is None or y_test is None) and isinstance(self.objective, str):
            if self.objective.endswith("_reduce"):
                fallback = self.objective.replace("_reduce", "")
                warnings.warn(
                    f"objective='{self.objective}' requires test data but none was "
                    f"provided. Falling back to '{fallback}' (train-only).",
                    stacklevel=2,
                )
                self.objective = fallback

        # Validate feature_belong keys exist in X
        if self.feature_belong:
            unknown = set()
            for src, feats in self.feature_belong.items():
                unknown.update(f for f in feats if f not in X.columns)
            if unknown:
                raise KeyError(
                    f"feature_belong contains features not in X: {sorted(unknown)}"
                )

        start_time = time.time()
        force_in = force_in or []
        all_features = list(X.columns)
        candidates = candidates or [f for f in all_features if f not in force_in]

        current = list(force_in)
        history: list[tuple[str, ...]] = []

        while self._iter_count < self.max_iter_round:
            self._iter_count += 1
            elapsed = time.time() - start_time
            if elapsed > self.max_iter_time:
                break

            # Forward
            current = self._forward_step(X, y, current, candidates, sort_df)

            # Backward
            current = self._backward_step(X, y, current, force_in)

            # Evaluate
            score, perf, passed = self._evaluate_trial(X, y, current, X_test, y_test)

            record = {
                "round": self._iter_count,
                "vars": current.copy(),
                "n_vars": len(current),
                "score": score,
                "passed": passed,
                "trn_ks": perf.get("trn_ks", 0.0),
                "test_ks": perf.get("test_ks", 0.0),
                "ks_reduce": perf.get("ks_reduce", 0.0),
                "trn_auc": perf.get("trn_auc", 0.5),
                "test_auc": perf.get("test_auc", 0.5),
                "auc_reduce": perf.get("auc_reduce", 0.0),
            }
            self._record[self._iter_count] = record
            history.append(tuple(current))

            if passed and score > getattr(self, "_best_score", -np.inf):
                self._best_score = score
                self._support = current.copy()
                self._best_perf = perf
                self._model = self._fit_logit(X, current, y)

            if self._should_exit_early(history, passed, len(current)):
                break

            # Perturbation
            if (
                self.perturbation
                and not passed
                and self._consecutive_unchanged(history) >= 2
            ):
                current = self._perturb(current, all_features)
                history.append(tuple(current))

            # Source-belong control
            current = self._apply_belong_control(current)

        # Force fill
        if self.force_fill and len(self._support) < self.n_min:
            self._support = self._force_fill(X, y, self._support, candidates)
            if self._support:
                self._best_perf = self._get_perf(X, y, self._support, X_test, y_test)
                self._model = self._fit_logit(X, self._support, y)

        if not self._support:
            self._support = force_in[: self.n_min] if force_in else []

        if not self._best_perf:
            self._best_perf = self._get_perf(X, y, self._support, X_test, y_test)
            if self._model is None and self._support:
                self._model = self._fit_logit(X, self._support, y)

        return self

    # ── core iteration ─────────────────────────────────────────────────────

    def _forward_step(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        current: list[str],
        candidates: list[str],
        sort_df: pd.DataFrame | None,
    ) -> list[str]:
        """Add one feature per pass; repeat while score improves."""
        remaining = [f for f in candidates if f not in current]
        if sort_df is not None and "feature" in sort_df.columns:
            ordered = sort_df[sort_df["feature"].isin(remaining)]["feature"].tolist()
            remaining = ordered + [f for f in remaining if f not in ordered]

        improved = True
        while improved and remaining:
            improved = False
            best_score, _, _ = self._evaluate_trial(X, y, current)
            best_add: str | None = None
            for f in remaining[:]:
                trial = current + [f]
                if self._quick_check(X, y, trial):
                    score, _, _ = self._evaluate_trial(X, y, trial)
                    if score - best_score >= self.goal_threshold:
                        best_add = f
                        best_score = score
                        improved = True
            if best_add:
                current.append(best_add)
                remaining.remove(best_add)
        return current

    def _backward_step(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        current: list[str],
        force_in: list[str],
    ) -> list[str]:
        """Remove one feature per pass; repeat while score improves."""
        if len(current) <= len(force_in):
            return current
        improved = True
        while improved:
            improved = False
            best_score, _, _ = self._evaluate_trial(X, y, current)
            to_remove: str | None = None
            for f in current:
                if f in force_in:
                    continue
                trial = [v for v in current if v != f]
                if self._quick_check(X, y, trial):
                    score, _, _ = self._evaluate_trial(X, y, trial)
                    if score - best_score >= self.goal_threshold:
                        to_remove = f
                        best_score = score
                        improved = True
            if to_remove:
                current.remove(to_remove)
        return current

    def _evaluate_trial(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        vars: list[str],
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ) -> tuple[float, dict[str, float], bool]:
        """Return ``(score, perf_dict, passed)`` for a variable set."""
        perf = self._get_perf(X, y, vars, X_test, y_test)
        score = self._objective_score(perf)
        trial_model = self._lr_cache.get(tuple(sorted(vars)))
        passed = self._check_pass(X, vars, perf, model=trial_model)
        return score, perf, passed

    # ── objective ──────────────────────────────────────────────────────────

    def _objective_score(self, perf: dict[str, float]) -> float:
        """Compute the objective score from a performance dictionary.

        Built-in modes:
        - ``'ks'`` / ``'ks_reduce'`` — maximise KS, with overfitting penalty.
        - ``'auc'`` / ``'auc_reduce'`` — maximise AUC, with overfitting penalty.
        - Custom: any ``Callable[[dict], float]``.
        """
        if callable(self.objective):
            return float(self.objective(perf))

        obj = self.objective

        if obj in ("auc", "auc_reduce"):
            trn_auc = perf.get("trn_auc", 0.5)
            test_auc = perf.get("test_auc", trn_auc)
            auc_reduce = perf.get("auc_reduce", 0.0)
            return test_auc + (1 - self.r) * trn_auc - self.r * auc_reduce

        # default: ks / ks_reduce
        trn_ks = perf.get("trn_ks", 0.0)
        test_ks = perf.get("test_ks", trn_ks)
        ks_reduce = perf.get("ks_reduce", 0.0)
        return test_ks + (1 - self.r) * trn_ks - self.r * ks_reduce

    # ── performance evaluation ─────────────────────────────────────────────

    def _get_perf(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        vars: list[str],
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ) -> dict[str, float]:
        """Fit LR on train, evaluate on train (and test if provided)."""
        empty = {
            "trn_ks": 0.0, "test_ks": 0.0, "ks_reduce": 0.0,
            "trn_auc": 0.5, "test_auc": 0.5, "auc_reduce": 0.0,
            "model_vars": vars,
        }
        if not vars:
            return empty

        model = self._fit_logit(X, vars, y)
        if model is None:
            return empty

        pred_trn = model.predict(sm.add_constant(X[vars], has_constant="add"))
        y_trn_arr = np.asarray(y)
        trn_auc = float(roc_auc_score(y_trn_arr, pred_trn))
        trn_ks = _ks_from_probs(pred_trn, y_trn_arr)

        if X_test is not None and y_test is not None:
            pred_test = model.predict(sm.add_constant(X_test[vars], has_constant="add"))
            y_tst_arr = np.asarray(y_test)
            test_auc = float(roc_auc_score(y_tst_arr, pred_test))
            test_ks = _ks_from_probs(pred_test, y_tst_arr)
            auc_reduce = trn_auc - test_auc
            ks_reduce = trn_ks - test_ks
        else:
            test_auc = trn_auc
            test_ks = trn_ks
            auc_reduce = 0.0
            ks_reduce = 0.0

        return {
            "trn_ks": trn_ks,
            "test_ks": test_ks,
            "ks_reduce": ks_reduce,
            "trn_auc": trn_auc,
            "test_auc": test_auc,
            "auc_reduce": auc_reduce,
            "model_vars": vars,
        }

    def _fit_logit(self, X: pd.DataFrame, vars: list[str], y: pd.Series) -> Any:
        """Fit a statsmodels Logit, with LRU cache by variable set.

        Caching means _quick_check and _get_perf can share the same fit
        without double computation — the second call hits the cache.
        """
        key = tuple(sorted(vars))
        if key in self._lr_cache:
            return self._lr_cache[key]

        if not vars:
            return None
        Xc = sm.add_constant(X[vars], has_constant="add")
        try:
            model = sm.Logit(y, Xc).fit(disp=False, maxiter=100)
        except (np.linalg.LinAlgError, ValueError):
            model = None
        self._lr_cache[key] = model
        return model

    # ── checks ─────────────────────────────────────────────────────────────

    def _quick_check(self, X: pd.DataFrame, y: pd.Series, vars: list[str]) -> bool:
        """Fast LR condition check (coefficients + VIF only)."""
        if not vars:
            return True
        model = self._fit_logit(X, vars, y)
        if model is None:
            return False
        return self._check_coefficients(model, vars) and self._check_vif_quick(X, vars)

    def _check_pass(
        self, X: pd.DataFrame, vars: list[str], perf: dict[str, float], model: Any = None
    ) -> bool:
        """Full acceptance check (coefficients, VIF, correlation, belong, count)."""
        if not vars:
            return False
        # Use the trial-specific model if provided; otherwise fall back to best model
        _model = model if model is not None else self._model
        coef_ok = self._check_coefficients(_model, vars) if _model else True
        # VIF / correlation: enforced here at the final check (not per-trial)
        vif_ok = self._check_vif_quick(X, vars)
        corr_ok = self._check_corr_quick(X, vars)
        belong_ok = self._check_belong(vars)
        num_ok = self.n_min <= len(vars) <= self.n_max
        return coef_ok and vif_ok and corr_ok and belong_ok and num_ok

    def _check_coefficients(self, model: Any, vars: list[str]) -> bool:
        """Check p-value and coefficient sign."""
        if self.pvalue_threshold is None and self.coef_sign is None:
            return True
        pvalues = model.pvalues[1:] if len(model.pvalues) > 1 else pd.Series()
        if self.pvalue_threshold is not None:
            if (pvalues > self.pvalue_threshold).any():
                return False
        if self.coef_sign == "positive":
            coef = model.params[1:] if len(model.params) > 1 else pd.Series()
            if (coef < 0).any():
                return False
        elif self.coef_sign == "negative":
            coef = model.params[1:] if len(model.params) > 1 else pd.Series()
            if (coef > 0).any():
                return False
        return True

    def _check_vif_quick(self, X: pd.DataFrame, vars: list[str]) -> bool:
        """VIF check for a trial variable set."""
        if self.vif_threshold is None or len(vars) < 2:
            return True
        Xv = X[vars].copy()
        Xv["c"] = 1.0
        for i in range(len(vars)):
            vif = variance_inflation_factor(Xv.values, i)
            if vif > self.vif_threshold:
                return False
        return True

    def _check_corr_quick(self, X: pd.DataFrame, vars: list[str]) -> bool:
        """Pairwise correlation check."""
        if self.corr_threshold is None or len(vars) < 2:
            return True
        corr = X[vars].corr().abs()
        np.fill_diagonal(corr.values, 0)
        return (corr > self.corr_threshold).sum().sum() == 0

    def _check_belong(self, vars: list[str]) -> bool:
        """Source-belong coverage check."""
        if not self.feature_belong or self.belong_max_pct is None:
            return True
        counts: dict[str, int] = {k: 0 for k in self.feature_belong}
        for v in vars:
            for k, flist in self.feature_belong.items():
                if v in flist:
                    counts[k] += 1
        max_allowed = (
            int(self.n_max * self.belong_max_pct)
            if self.belong_max_pct < 1
            else self.belong_max_pct
        )
        return all(c <= max_allowed for c in counts.values())

    def _apply_belong_control(self, vars: list[str]) -> list[str]:
        """Trim features from over-represented sources.

        For each source exceeding *belong_max_pct*, drop the features with
        the lowest per-feature IV (approximated by order in *vars* — later
        entries are assumed weaker).
        """
        if not self.feature_belong or self.belong_max_pct is None:
            return vars

        max_allowed = (
            int(self.n_max * self.belong_max_pct)
            if self.belong_max_pct < 1
            else int(self.belong_max_pct)
        )

        for source, flist in self.feature_belong.items():
            in_vars = [f for f in vars if f in flist]
            if len(in_vars) > max_allowed:
                # Drop excess features (keep the first ones — assumed higher IV)
                to_drop = in_vars[max_allowed:]
                vars = [f for f in vars if f not in to_drop]
        return vars

    # ── perturbation / force-fill ──────────────────────────────────────────

    def _perturb(self, current: list[str], all_feats: list[str]) -> list[str]:
        """Drop a fraction of features AND add a few random ones to escape local optima."""
        if not self.perturbation or len(current) < 3:
            return current
        n_remove = max(1, int(len(current) * self.perturbation_pct))
        keep = np.random.choice(
            current, size=len(current) - n_remove, replace=False
        ).tolist()

        # Also randomly add up to perturbation_add new candidates
        remaining = [f for f in all_feats if f not in keep]
        if remaining and self.perturbation_add > 0:
            n_add = min(self.perturbation_add, len(remaining))
            additions = np.random.choice(remaining, size=n_add, replace=False).tolist()
            keep.extend(additions)
        return keep

    def _force_fill(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        current: list[str],
        candidates: list[str],
    ) -> list[str]:
        """Pad *current* up to *n_min* with passing candidates."""
        remaining = [f for f in candidates if f not in current]
        while len(current) < self.n_min and remaining:
            for f in remaining:
                trial = current + [f]
                if self._quick_check(X, y, trial):
                    current.append(f)
                    remaining.remove(f)
                    break
            else:
                break
        return current

    # ── exit logic ─────────────────────────────────────────────────────────

    def _should_exit_early(
        self,
        history: list[tuple[str, ...]],
        passed: bool,
        n_vars: int,
    ) -> bool:
        """Determine whether iteration should stop."""
        if not passed:
            return False
        if n_vars < self.n_min or n_vars > self.n_max:
            return False
        if self._consecutive_unchanged(history) >= self.same_round_exit:
            return True
        return False

    def _consecutive_unchanged(self, history: list[tuple[str, ...]]) -> int:
        """Count consecutive rounds with identical variable sets.

        Returns the run length (number of consecutive identical states),
        not the number of comparisons.  For history [A, A, A, A], returns 4.
        """
        if len(history) < 2:
            return 0
        n = 0
        for i in range(len(history) - 1, 0, -1):
            if history[i] == history[i - 1]:
                n += 1
            else:
                break
        return n + 1 if n > 0 else 0

    # ── properties ─────────────────────────────────────────────────────────

    @property
    def support_(self) -> list[str]:
        """Final selected features."""
        return self._support

    @property
    def record_(self) -> dict[int, dict[str, Any]]:
        """Per-round iteration record."""
        return self._record

    @property
    def best_performance_(self) -> dict[str, Any]:
        """Performance dict of the best solution."""
        return self._best_perf

    @property
    def model_(self) -> Any:
        """Best fitted Logit model."""
        return self._model


# ── module-level helpers ───────────────────────────────────────────────────


def _ks_from_probs(prob: np.ndarray, y: np.ndarray) -> float:
    """Compute KS statistic from predicted probabilities and binary target."""
    # Manual CDF-based KS. For sklearn roc_curve KS, see _metrics._ks.
    df = pd.DataFrame({"pred": prob, "y": y})
    df = df.sort_values("pred").reset_index(drop=True)
    total_bad = df["y"].sum()
    total_good = len(df) - total_bad
    if total_bad == 0 or total_good == 0:
        return 0.0
    cum_bad = df["y"].cumsum() / total_bad
    cum_good = (1 - df["y"]).cumsum() / total_good
    return float((cum_bad - cum_good).abs().max())
