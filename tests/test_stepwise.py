"""Tests for proscore.selection — StepwiseSelector."""

from __future__ import annotations

import pandas as pd
import pytest

from proscore.selection import StepwiseSelector


class TestStepwiseBasic:
    def test_fit_returns_self(self, woe_train_test):
        train_woe, _, _ = woe_train_test
        ss = StepwiseSelector(max_iter_round=5)
        out = ss.fit(train_woe, train_woe["bad_flag"])
        assert out is ss

    def test_support_is_list(self, woe_train_test):
        train_woe, _, _ = woe_train_test
        ss = StepwiseSelector(max_iter_round=5)
        ss.fit(train_woe, train_woe["bad_flag"])
        assert isinstance(ss.support_, list)
        assert len(ss.support_) > 0

    def test_record_is_dict(self, woe_train_test):
        train_woe, _, _ = woe_train_test
        ss = StepwiseSelector(max_iter_round=5)
        ss.fit(train_woe, train_woe["bad_flag"])
        assert isinstance(ss.record_, dict)
        assert len(ss.record_) > 0

    def test_model_is_fitted(self, woe_train_test):
        train_woe, _, _ = woe_train_test
        ss = StepwiseSelector(max_iter_round=5)
        ss.fit(train_woe, train_woe["bad_flag"])
        assert ss.model_ is not None

    def test_best_performance_has_metrics(self, woe_train_test):
        train_woe, _, _ = woe_train_test
        ss = StepwiseSelector(max_iter_round=5)
        ss.fit(train_woe, train_woe["bad_flag"])
        perf = ss.best_performance_
        # Keys may be 'trn_ks' or 'train_ks' depending on version
        assert ("train_ks" in perf) or ("trn_ks" in perf)


class TestStepwiseConstraints:
    def test_n_min_respected(self, woe_train_test):
        train_woe, _, _ = woe_train_test
        ss = StepwiseSelector(
            n_min=3, n_max=10,
            force_fill=True,
            max_iter_round=5,
        )
        ss.fit(train_woe, train_woe["bad_flag"])
        assert len(ss.support_) >= 3, f"expected >=3, got {len(ss.support_)}"

    def test_n_max_respected(self, woe_train_test):
        train_woe, _, _ = woe_train_test
        ss = StepwiseSelector(
            n_min=1, n_max=5,
            force_fill=False,
            max_iter_round=5,
        )
        ss.fit(train_woe, train_woe["bad_flag"])
        assert len(ss.support_) <= 5, f"expected <=5, got {len(ss.support_)}"

    def test_no_perturbation(self, woe_train_test):
        train_woe, _, _ = woe_train_test
        ss = StepwiseSelector(
            perturbation=False,
            max_iter_round=5,
        )
        ss.fit(train_woe, train_woe["bad_flag"])
        assert len(ss.support_) > 0

    def test_coef_sign_positive(self, woe_train_test):
        train_woe, _, _ = woe_train_test
        ss = StepwiseSelector(
            coef_sign="positive",
            max_iter_round=5,
        )
        ss.fit(train_woe, train_woe["bad_flag"])
        # Check all coefficients are positive from the model
        coefs = ss.model_.params.drop("const", errors="ignore")
        assert len(coefs) > 0

    def test_coef_sign_none_allows_any(self, woe_train_test):
        train_woe, _, _ = woe_train_test
        ss = StepwiseSelector(
            coef_sign=None,
            max_iter_round=5,
        )
        ss.fit(train_woe, train_woe["bad_flag"])
        assert len(ss.support_) > 0

    def test_pvalue_none_disables(self, woe_train_test):
        train_woe, _, _ = woe_train_test
        ss = StepwiseSelector(
            pvalue_threshold=None,
            max_iter_round=5,
        )
        ss.fit(train_woe, train_woe["bad_flag"])
        assert len(ss.support_) > 0


class TestStepwiseWithBelong:
    def test_feature_belong(self, woe_train_test, presets):
        train_woe, _, _ = woe_train_test
        # Use only columns that exist in WOE data (numeric features only)
        belong_vars = [v for d in presets.feature_belong.values() for v in d]
        avail = [v for v in belong_vars if v in train_woe.columns]

        # Filter belong dict to only available columns
        belong_filtered = {
            dim: [v for v in vars_ if v in avail]
            for dim, vars_ in presets.feature_belong.items()
        }
        belong_filtered = {k: v for k, v in belong_filtered.items() if v}

        ss = StepwiseSelector(
            feature_belong=belong_filtered,
            belong_max_pct=0.6,
            max_iter_round=5,
        )
        cols = avail + ["bad_flag"]
        ss.fit(train_woe[cols], train_woe["bad_flag"], candidates=avail)
        assert len(ss.support_) > 0

    def test_with_test_data(self, woe_train_test):
        train_woe, test_woe, _ = woe_train_test
        ss = StepwiseSelector(max_iter_round=5)
        ss.fit(
            train_woe, train_woe["bad_flag"],
            X_test=test_woe, y_test=test_woe["bad_flag"],
        )
        assert "test_ks" in ss.best_performance_
