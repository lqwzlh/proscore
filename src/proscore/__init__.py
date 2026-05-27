"""ProScore — Scorecard modelling toolkit."""

from __future__ import annotations

import warnings

import pandas as pd

from proscore import inspect
from proscore._pipeline_config import (
    PipelineConfig,  # noqa: F401
    run_pipeline,  # noqa: F401
)
from proscore._spec import PipelineSpec
from proscore.binning import Binning, BinningProcess
from proscore.evaluate import evaluate as _evaluate
from proscore.modeling import ScoreCard
from proscore.rules import RuleMiner
from proscore.selection import Filter, StepwiseSelector, assess_screen
from proscore.transform import WOETransformer

__version__ = "0.1.1"


class ProScore:
    """
    Chain-style entry point for the full scorecard pipeline.

    Recommended order (aligned with modular notebooks)::

        ps = ProScore()
        ps.read(train=..., test=..., oot=..., target="bad_flag") \\
          .detect() \\
          .prefilter(max_corr=0.75, max_vif=10) \\
          .bin(method="chi", n_bins=5) \\
          .refine(iv_range=(0.02, None), max_psi=0.25) \\
          .mine_rules(...)   # optional: mine decision rules from raw features \\
          .transform() \\
          .select() \\
          .fit(odds=50, pdo=10) \\
          .scorecard() \\
          .evaluate()
    """

    def __init__(self) -> None:
        self.target: str = ""
        self.id_col: str | None = None
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.oot_df: pd.DataFrame | None = None

        self.detect_result: pd.DataFrame | None = None
        self.quality_result: pd.DataFrame | None = None
        self._prefilter: Filter | None = None
        self._filter: Filter | None = None
        self._binner: Binning | BinningProcess | None = None
        self._transformer: WOETransformer | None = None
        self._selector: StepwiseSelector | None = None
        self._scorecard: ScoreCard | None = None
        self.eval_result: dict | None = None

        self._halted: bool = False
        self._halt_message: str = ""
        self._refine_skipped: bool = False
        self._screen_outcomes: list = []
        self._spec: PipelineSpec | None = None
        self._rulemine: RuleMiner | None = None
        self._rule_features: list[str] = []

    # ── helpers ───────────────────────────────────────────────────────────────

    def _merge_kw(self, section: str, **explicit) -> dict:
        """Merge explicit kwargs with PipelineSpec defaults (explicit wins)."""
        if self._spec is None:
            return explicit
        spec_kw = getattr(self._spec, section, {})
        return {**spec_kw, **explicit}

    def _train_X(self, features: list[str]) -> pd.DataFrame:
        return self.train_df[features]

    def _train_y(self) -> pd.Series:
        return self.train_df[self.target]

    def _test_X(self, features: list[str]) -> pd.DataFrame | None:
        if self.test_df is None:
            return None
        return self.test_df[features]

    def _test_y(self) -> pd.Series | None:
        if self.test_df is None:
            return None
        return self.test_df[self.target]

    def _oot_X(self, features: list[str]) -> pd.DataFrame | None:
        if self.oot_df is None:
            return None
        return self.oot_df[features]

    def _oot_y(self) -> pd.Series | None:
        if self.oot_df is None:
            return None
        return self.oot_df[self.target]

    def _categorical_features(self) -> list[str]:
        return [
            c for c in self.train_df.columns
            if c != self.target and not pd.api.types.is_numeric_dtype(self.train_df[c])
        ]

    def _initial_features(self) -> list[str]:
        """Numeric columns — the Filter candidate pool."""
        return [
            c for c in self.train_df.columns
            if c != self.target and pd.api.types.is_numeric_dtype(self.train_df[c])
        ]

    def _numeric_after_prefilter(self) -> list[str]:
        if self._prefilter is not None and self._prefilter.support_:
            return self._prefilter.support_
        return self._initial_features()

    def _current_numeric(self) -> list[str]:
        """Numeric features after refine (or prefilter / fallback)."""
        if self._selector is not None:
            return [f for f in self._selector.support_ if f in self._initial_features()]
        if self._filter is not None and self._filter.support_:
            return self._filter.support_
        return self._numeric_after_prefilter()

    def _features_for_binning_fit(self) -> list[str]:
        """Columns passed to :meth:`bin` — prefilter survivors + categoricals."""
        return self._numeric_after_prefilter() + self._categorical_features()

    def _features_for_modeling(self) -> list[str]:
        """Columns for WOE / stepwise — refine survivors + categoricals."""
        return self._current_numeric() + self._categorical_features()

    # ── spec injection ────────────────────────────────────────────────────────

    def apply(self, spec: PipelineSpec) -> ProScore:
        """Apply a :class:`PipelineSpec` as default parameters for the pipeline.

        Explicit kwargs on individual chain methods override spec defaults.
        """
        self._spec = spec
        return self

    # ── read ──────────────────────────────────────────────────────────────────

    def read(
        self,
        train: pd.DataFrame,
        *,
        target: str,
        test: pd.DataFrame | None = None,
        oot: pd.DataFrame | None = None,
        id_col: str | None = None,
    ) -> ProScore:
        """Load data (already split into train / test / oot)."""
        self.train_df = train
        self.test_df = test
        self.oot_df = oot
        self.target = target
        self.id_col = id_col

        base_cols = set(train.columns)
        if test is not None and set(test.columns) != base_cols:
            raise ValueError("test columns must exactly match train columns")
        if oot is not None and set(oot.columns) != base_cols:
            raise ValueError("oot columns must exactly match train columns")

        return self

    # ── inspect (train only) ──────────────────────────────────────────────────

    def detect(self, **kwargs) -> ProScore:
        _check_read(self)
        self.detect_result = inspect.detect(self.train_df, target=self.target, **kwargs)
        return self

    def quality(self, **kwargs) -> ProScore:
        _check_read(self)
        self.quality_result = inspect.quality(self.train_df, target=self.target, **kwargs)
        return self

    # ── prefilter (coarse — no bin_table required) ───────────────────────────

    def prefilter(self, **kwargs) -> ProScore:
        """
        Coarse feature screen on numeric columns (no binning required).

        Typical kwargs: ``max_missing_rate``, ``max_one_value_rate``,
        ``min_auc``, ``max_corr``, ``max_vif``.  Leave ``iv_range`` and
        ``max_psi`` unset (``None``) until :meth:`refine`.
        """
        kwargs = self._merge_kw("prefilter", **kwargs)
        _check_read(self)
        features = self._initial_features()
        self._prefilter = Filter(**kwargs)
        self._prefilter.fit(
            self._train_X(features), self._train_y(),
            X_test=self._test_X(features) if self.test_df is not None else None,
            bin_table=None,
        )
        outcome = assess_screen(
            self._prefilter.support_,
            stage="prefilter",
            n_candidates=len(features),
        )
        self._screen_outcomes.append(outcome)
        if not outcome.ok and len(self._categorical_features()) == 0:
            self._halted = True
            self._halt_message = outcome.message
        return self

    # ── bin (train — survivors of prefilter + categoricals) ───────────────────

    def bin(self, method: str = "chi", n_bins: int = 10, **kwargs) -> ProScore:
        kwargs = self._merge_kw("binning", **kwargs)
        method = kwargs.pop("method", method)
        n_bins = kwargs.pop("n_bins", n_bins)
        _check_read(self)
        if self._halted:
            _warn_halted(self)
            return self
        features = self._features_for_binning_fit()
        if len(features) == 0:
            self._halted = True
            self._halt_message = "分箱阶段无可选特征（数值与类别均为空）。"
            _warn_halted(self)
            return self
        X = pd.concat([self.train_df[features], self.train_df[self.target]], axis=1)

        feature_config = kwargs.pop("feature_config", None)
        if feature_config:
            self._binner = BinningProcess(
                feature_config=feature_config,
                default_method=method,
                default_n_bins=n_bins,
                **kwargs,
            )
            self._binner.fit(X, y=self.target)
        else:
            self._binner = Binning(method=method, n_bins=n_bins, **kwargs)
            self._binner.fit(X, y=self.target)
        return self

    # ── refine (fine — IV/PSI from bin_table_; requires :meth:`bin`) ─────────

    def refine(self, **kwargs) -> ProScore:
        """
        Fine screen on numeric columns that passed :meth:`prefilter`.

        Pass ``iv_range``, ``max_psi`` (Train vs Test), etc.  Uses
        :attr:`Binning.bin_table_` from the preceding :meth:`bin` call.
        """
        kwargs = self._merge_kw("refine", **kwargs)
        _check_read(self)
        if self._halted:
            _warn_halted(self)
            return self
        _check_binner(self)
        features = self._numeric_after_prefilter()
        if len(features) == 0:
            import warnings

            from proscore.selection._screen import FeatureScreenWarning

            warnings.warn(
                "refine skipped: no numeric features after prefilter.",
                FeatureScreenWarning,
                stacklevel=2,
            )
            self._refine_skipped = True
            self._filter = None
            return self
        self._refine_skipped = False
        self._filter = Filter(**kwargs)
        self._filter.fit(
            self._train_X(features), self._train_y(),
            X_test=self._test_X(features) if self.test_df is not None else None,
            bin_table=self._binner.bin_table_,
        )
        outcome = assess_screen(
            self._filter.support_,
            stage="refine",
            n_candidates=len(features),
        )
        self._screen_outcomes.append(outcome)
        modeling = self._features_for_modeling()
        if not outcome.ok and len(modeling) == 0:
            self._halted = True
            self._halt_message = outcome.message
        return self

    def filter(self, **kwargs) -> ProScore:
        """Alias for :meth:`refine` (backward compatibility)."""
        return self.refine(**kwargs)

    # ── transform ─────────────────────────────────────────────────────────────

    def transform(self, unseen_strategy: str = "worst", **kwargs) -> ProScore:
        if self._halted:
            _warn_halted(self)
            return self
        _check_binner(self)
        _check_refine(self)
        features = self._features_for_modeling()
        if len(features) == 0:
            self._halted = True
            self._halt_message = "无可建模特征（数值+类别均为空）。"
            _warn_halted(self)
            return self
        self._transformer = WOETransformer(unseen_strategy=unseen_strategy, **kwargs)
        tables = {k: v for k, v in self._binner.bin_table_.items() if k in features}
        self._transformer.fit(tables)
        return self

    # ── mine_rules (optional — before transform / select) ────────────────────

    def mine_rules(self, **kwargs) -> ProScore:
        """Mine decision rules from refined candidates (before WOE transform).

        Rules use raw feature values (not WOE).  Mined features are
        automatically excluded from :meth:`select`.
        """
        kwargs = self._merge_kw("rules", **kwargs)
        _check_read(self)
        if self._binner is None:
            raise RuntimeError("Call bin() before mine_rules().")
        features = self._current_numeric()
        if len(features) == 0:
            warnings.warn("No numeric features available for rule mining.", stacklevel=2)
            return self

        self._rulemine = RuleMiner(**kwargs)
        self._rulemine.fit(
            X=self._train_X(features),
            y=self._train_y(),
            bin_table=self._binner.bin_table_,
        )
        self._rule_features = self._rulemine.used_features_

        return self

    # ── select ────────────────────────────────────────────────────────────────

    def select(self, **kwargs) -> ProScore:
        kwargs = self._merge_kw("select", **kwargs)
        if kwargs.pop("method", None) is not None:
            warnings.warn(
                "select(method=...) is ignored; only stepwise selection is supported.",
                UserWarning,
                stacklevel=2,
            )
        if self._halted:
            _warn_halted(self)
            return self
        _check_transformer(self)
        features = [c for c in self._features_for_modeling()
                    if c not in self._rule_features]
        train_woe = self._transformer.transform(self._train_X(features))
        train_woe[self.target] = self._train_y().values

        test_woe = None
        y_test = None
        if self.test_df is not None:
            test_woe = self._transformer.transform(self._test_X(features))
            y_test = self._test_y().values

        force_in = kwargs.pop("force_in", None)
        self._selector = StepwiseSelector(**kwargs)
        self._selector.fit(
            train_woe, self._train_y(), candidates=features,
            force_in=force_in, X_test=test_woe, y_test=y_test,
        )
        return self

    # ── fit / scorecard / evaluate ────────────────────────────────────────────

    def fit(self, odds: float = 50, pdo: float = 10, base_score: float = 600, **kwargs) -> ProScore:
        kwargs = self._merge_kw("model", **kwargs)
        odds = kwargs.pop("odds", odds)
        pdo = kwargs.pop("pdo", pdo)
        base_score = kwargs.pop("base_score", base_score)
        if self._halted:
            _warn_halted(self)
            return self
        _check_selector(self)
        features = self._selector.support_
        train_woe = self._transformer.transform(self._train_X(features))
        train_woe[self.target] = self._train_y().values

        self._scorecard = ScoreCard(odds=odds, pdo=pdo, base_score=base_score, **kwargs)
        self._scorecard.fit(train_woe, y=self.target, features=features)
        return self

    def scorecard(self) -> ProScore:
        _check_scorecard(self)
        features = self._selector.support_
        tables = {k: v for k, v in self._binner.bin_table_.items() if k in features}
        self._scorecard.scorecard(tables)
        return self

    def evaluate(self, n_bins: int = 10) -> ProScore:
        _check_scorecard(self)
        features = self._selector.support_

        train_woe = self._transformer.transform(self._train_X(features))

        test_woe = None
        if self.test_df is not None:
            test_woe = self._transformer.transform(self._test_X(features))

        oot_woe = None
        if self.oot_df is not None:
            oot_woe = self._transformer.transform(self._oot_X(features))

        self.eval_result = _evaluate(
            self._scorecard.model_,
            train_woe, self._train_y(),
            X_test=test_woe, y_test=self._test_y(),
            X_oot=oot_woe, y_oot=self._oot_y(),
            features=features, n_bins=n_bins,
        )
        return self

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def prefilter_(self) -> Filter | None:
        """Coarse filter result (:meth:`prefilter`)."""
        return self._prefilter

    @property
    def filter_(self) -> Filter | None:
        """Fine filter result (:meth:`refine`); same object as ``refine_``."""
        return self._filter

    @property
    def refine_(self) -> Filter | None:
        return self._filter

    @property
    def binner_(self) -> Binning | BinningProcess | None:
        return self._binner

    @property
    def transformer_(self) -> WOETransformer | None:
        return self._transformer

    @property
    def selector_(self) -> StepwiseSelector | None:
        return self._selector

    @property
    def scorecard_(self) -> ScoreCard | None:
        return self._scorecard

    @property
    def support_(self) -> list[str]:
        if self._selector is None:
            return []
        return self._selector.support_

    @property
    def bin_tables_(self):
        if self._binner is None:
            raise RuntimeError("Call bin() first.")
        return self._binner.bin_table_

    @property
    def score_table_(self):
        if self._scorecard is None:
            raise RuntimeError("Call fit() first.")
        return self._scorecard.score_table_

    @property
    def model_(self):
        if self._scorecard is None:
            raise RuntimeError("Call fit() first.")
        return self._scorecard.model_

    @property
    def halted_(self) -> bool:
        """``True`` when the pipeline should stop (no modelling features)."""
        return self._halted

    @property
    def halt_message_(self) -> str:
        return self._halt_message

    @property
    def screen_outcomes_(self) -> list:
        return list(self._screen_outcomes)

    @property
    def rulemine_(self) -> RuleMiner | None:
        """The fitted :class:`RuleMiner` instance (or ``None``)."""
        return self._rulemine

    @property
    def rules_table_(self):
        """Mined rules evaluation table (or empty DataFrame)."""
        if self._rulemine is None:
            return pd.DataFrame()
        return self._rulemine.rules_table_


def _warn_halted(ps: ProScore) -> None:
    import warnings

    from proscore.selection._screen import FeatureScreenWarning

    warnings.warn(
        f"Pipeline halted: {ps._halt_message}",
        FeatureScreenWarning,
        stacklevel=3,
    )


def _check_read(ps: ProScore) -> None:
    if ps.train_df is None:
        raise RuntimeError("Call read() first.")


def _check_binner(ps: ProScore) -> None:
    if ps._binner is None:
        raise RuntimeError("Call bin() first.")


def _check_refine(ps: ProScore) -> None:
    if ps._filter is not None:
        return  # refine() was called and completed normally
    if getattr(ps, "_refine_skipped", False):
        return  # refine() was called but had nothing to do
    if ps._prefilter is not None:
        return  # prefilter was run, refine intentionally skipped (valid path)
    raise RuntimeError("Call refine() first — or at least prefilter().")


def _check_transformer(ps: ProScore) -> None:
    if ps._transformer is None:
        raise RuntimeError("Call transform() first.")


def _check_selector(ps: ProScore) -> None:
    if ps._selector is None:
        raise RuntimeError("Call select() first.")


def _check_scorecard(ps: ProScore) -> None:
    if ps._scorecard is None:
        raise RuntimeError("Call fit() first.")
