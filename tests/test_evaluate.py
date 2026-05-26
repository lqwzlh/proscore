"""Tests for proscore.evaluate — evaluate."""

from __future__ import annotations

import pytest

from proscore.evaluate import evaluate
from proscore.selection import StepwiseSelector


@pytest.fixture(scope="module")
def eval_fixtures(woe_train_test):
    """Fit a model and return evaluation inputs."""
    from proscore.modeling import ScoreCard

    train_woe, test_woe, _ = woe_train_test
    candidates = [c for c in train_woe.columns if c != "bad_flag"]

    ss = StepwiseSelector(n_min=3, n_max=8, force_fill=True, max_iter_round=8)
    ss.fit(train_woe, train_woe["bad_flag"], candidates=candidates)

    sc = ScoreCard(odds=20, pdo=20, base_score=600)
    sc.fit(train_woe, y="bad_flag", features=ss.support_)

    return sc.model_, train_woe, test_woe, ss.support_


class TestEvaluate:
    def test_returns_dict(self, eval_fixtures):
        model, train_woe, test_woe, support = eval_fixtures
        result = evaluate(
            model,
            train_woe[support], train_woe["bad_flag"],
            test_woe[support], test_woe["bad_flag"],
        )
        assert isinstance(result, dict)

    def test_has_expected_keys(self, eval_fixtures):
        model, train_woe, test_woe, support = eval_fixtures
        result = evaluate(
            model,
            train_woe[support], train_woe["bad_flag"],
            test_woe[support], test_woe["bad_flag"],
        )
        for key in ["trn_ks", "test_ks", "trn_auc", "test_auc", "psi"]:
            assert key in result, f"Missing key: {key}"

    def test_ks_in_range(self, eval_fixtures):
        model, train_woe, test_woe, support = eval_fixtures
        result = evaluate(
            model,
            train_woe[support], train_woe["bad_flag"],
            test_woe[support], test_woe["bad_flag"],
        )
        assert 0 < result["trn_ks"] < 1, f"trn_ks={result['trn_ks']}"
        assert 0 < result["test_ks"] < 1, f"test_ks={result['test_ks']}"

    def test_auc_in_range(self, eval_fixtures):
        model, train_woe, test_woe, support = eval_fixtures
        result = evaluate(
            model,
            train_woe[support], train_woe["bad_flag"],
            test_woe[support], test_woe["bad_flag"],
        )
        assert 0.5 < result["trn_auc"] <= 1.0, f"trn_auc={result['trn_auc']}"
        assert 0.5 < result["test_auc"] <= 1.0, f"test_auc={result['test_auc']}"

    def test_psi_nonnegative(self, eval_fixtures):
        model, train_woe, test_woe, support = eval_fixtures
        result = evaluate(
            model,
            train_woe[support], train_woe["bad_flag"],
            test_woe[support], test_woe["bad_flag"],
        )
        assert result["psi"] >= 0

    def test_score_table_present(self, eval_fixtures):
        model, train_woe, test_woe, support = eval_fixtures
        result = evaluate(
            model,
            train_woe[support], train_woe["bad_flag"],
            test_woe[support], test_woe["bad_flag"],
        )
        assert "score_table" in result
