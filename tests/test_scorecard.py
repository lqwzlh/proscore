"""Tests for proscore.modeling — ScoreCard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proscore.modeling import ScoreCard
from proscore.selection import StepwiseSelector


@pytest.fixture(scope="module")
def fitted_scorecard(woe_train_test, binning_result):
    """Fit a quick stepwise selection + scorecard for testing."""
    train_woe, test_woe, _ = woe_train_test
    candidates = [c for c in train_woe.columns if c != "bad_flag"]

    ss = StepwiseSelector(
        n_min=3, n_max=8,
        force_fill=True,
        max_iter_round=8,
    )
    ss.fit(train_woe, train_woe["bad_flag"], candidates=candidates)

    sc = ScoreCard(odds=20, pdo=20, base_score=600)
    sc.fit(train_woe, y="bad_flag", features=ss.support_)
    sc.scorecard(binning_result.bin_table_)
    return sc, ss.support_


class TestScoreCard:
    def test_fit_sets_model(self, fitted_scorecard):
        sc, _ = fitted_scorecard
        assert sc.model_ is not None

    def test_score_table_is_dataframe(self, fitted_scorecard):
        sc, _ = fitted_scorecard
        assert isinstance(sc.score_table_, pd.DataFrame)
        assert not sc.score_table_.empty

    def test_score_table_has_expected_columns(self, fitted_scorecard):
        sc, _ = fitted_scorecard
        cols = sc.score_table_.columns
        assert "variable" in cols
        assert "points" in cols or "score" in cols

    def test_scores_are_finite(self, fitted_scorecard, woe_train_test):
        sc, support = fitted_scorecard
        train_woe, _, _ = woe_train_test
        scores = sc.predict(train_woe[support])
        assert np.isfinite(scores).all()
        assert isinstance(scores, (pd.Series, np.ndarray))

    def test_scores_reasonable_range(self, fitted_scorecard, woe_train_test):
        sc, support = fitted_scorecard
        train_woe, _, _ = woe_train_test
        scores = sc.predict(train_woe[support])
        mean_score = np.mean(scores)
        # With base=600, odds=20, pdo=20, scores should be roughly around 600
        assert 400 < mean_score < 800, f"mean score {mean_score} out of range"

    def test_scores_have_predictive_power(self, fitted_scorecard, woe_train_test):
        """Higher score = lower risk → negative correlation with bad_flag."""
        sc, support = fitted_scorecard
        train_woe, _, _ = woe_train_test
        scores = sc.predict(train_woe[support])
        y = train_woe["bad_flag"].values

        corr = np.corrcoef(scores, y)[0, 1]
        assert corr < -0.05, (
            f"Score-bad_flag correlation ({corr:.4f}) should be negative "
            "(higher score = lower risk)"
        )

    def test_train_scores_non_negative(self, fitted_scorecard, woe_train_test):
        sc, support = fitted_scorecard
        train_woe, _, _ = woe_train_test
        scores = sc.predict(train_woe[support])
        assert scores.min() >= 0, f"min score {scores.min()} should be non-negative"
