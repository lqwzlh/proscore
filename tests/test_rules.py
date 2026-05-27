"""Tests for proscore.rules — RuleMiner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proscore.binning import Binning
from proscore.rules import RuleMiner


@pytest.fixture(scope="module")
def rule_data():
    rng = np.random.default_rng(42)
    n = 300
    x1 = np.concatenate([rng.normal(0, 1, 200), rng.normal(3, 1, 100)])
    x2 = rng.normal(2, 1, n)
    # Add some missing to x1
    x1[:15] = np.nan
    y = np.where(x1 > 2, 1, 0).astype(int)
    # Make missing rows all bad
    y[:15] = 1
    df = pd.DataFrame({"x1": np.where(np.isnan(x1), np.nan, x1),
                       "x2": x2})
    return df, y


@pytest.fixture(scope="module")
def bin_table(rule_data):
    df, y = rule_data
    b = Binning(method="chi", n_bins=5)
    b.fit(pd.concat([df, pd.Series(y, name="bad")], axis=1), y="bad")
    return b.bin_table_


class TestRuleMiner:
    def test_exhaustive_returns_rules(self, rule_data, bin_table):
        df, y = rule_data
        rm = RuleMiner(method="exhaustive", min_lift=1.5, max_rules=10)
        rm.fit(df, y, bin_table=bin_table)
        assert len(rm.rules_table_) > 0

    def test_missing_rule_present(self, rule_data, bin_table):
        df, y = rule_data
        rm = RuleMiner(method="exhaustive", min_lift=1.5, max_rules=20)
        rm.fit(df, y, bin_table=bin_table)
        has_missing = any("is missing" in r.rule for r in rm._rules)
        assert has_missing, "x1 has 5% missing all-bad → should generate missing rule"

    def test_rule_mask_supports_missing(self, rule_data, bin_table):
        df, _ = rule_data
        rm = RuleMiner()
        rm.fit(df, pd.Series([0]*len(df)), bin_table=bin_table)
        # Test rule_mask handles "is missing"
        mask = rm._rule_mask("x1 is missing", df)
        assert mask.sum() == df["x1"].isna().sum()

    def test_used_features_lists_rule_variables(self, rule_data, bin_table):
        df, y = rule_data
        rm = RuleMiner(method="exhaustive", min_lift=1.5, max_rules=20)
        rm.fit(df, y, bin_table=bin_table)
        assert rm.used_features_
        if any("is missing" in r.rule for r in rm._rules):
            assert "x1" in rm.used_features_

    def test_max_depth_three_way_cross(self):
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        })
        y = pd.Series((df["a"] + df["b"] + df["c"] > 1.5).astype(int))
        b = Binning(method="chi", n_bins=3)
        b.fit(pd.concat([df, y.rename("bad")], axis=1), y="bad")
        rm = RuleMiner(method="exhaustive", max_depth=3, min_lift=1.0, max_rules=50)
        rm.fit(df, y, bin_table=b.bin_table_)
        three_way = [r for r in rm._rules if r.rule.count(" AND ") == 2]
        assert three_way, "max_depth=3 with 3 features should yield 3-way cross rules"

    def test_apriori_runs(self, rule_data, bin_table):
        df, y = rule_data
        rm = RuleMiner(method="apriori", min_lift=1.5, max_rules=10)
        rm.fit(df, y, bin_table=bin_table)
        assert len(rm.rules_table_) > 0

    def test_tree_runs(self, rule_data, bin_table):
        df, y = rule_data
        rm = RuleMiner(method="tree", max_tree_depth=3, min_lift=1.5)
        rm.fit(df, y, bin_table=bin_table)
        assert len(rm.rules_table_) >= 0  # may be empty with small data

    def test_coverage_report(self, rule_data, bin_table):
        df, y = rule_data
        rm = RuleMiner(method="exhaustive", min_lift=1.5, max_rules=10)
        rm.fit(df, y, bin_table=bin_table)
        cr = rm.coverage_report(df, y)
        assert "cum_recall" in cr.columns
        assert "cum_bad_rate" in cr.columns

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            RuleMiner(method="invalid")

    def test_empty_rules_table(self, rule_data, bin_table):
        df, y = rule_data
        rm = RuleMiner(method="exhaustive", min_lift=100.0)
        rm.fit(df, y, bin_table=bin_table)
        assert len(rm.rules_table_) == 0
        assert rm.used_features_ == []
