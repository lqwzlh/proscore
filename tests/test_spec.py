"""Tests for PipelineSpec and ProScore.apply integration."""

from __future__ import annotations

import warnings

from proscore import PipelineSpec, ProScore


class TestPipelineSpec:
    def test_merge_shallow_per_section(self):
        base = PipelineSpec(binning={"method": "chi", "n_bins": 5})
        merged = base.merge(binning={"n_bins": 8})
        assert merged.binning == {"method": "chi", "n_bins": 8}
        assert base.binning == {"method": "chi", "n_bins": 5}

    def test_merge_kw_explicit_wins(self):
        ps = ProScore()
        ps._spec = PipelineSpec(binning={"n_bins": 99})
        kw = ps._merge_kw("binning", n_bins=3)
        assert kw["n_bins"] == 3

    def test_apply_binning_defaults_used(self, full_df):
        train = full_df.drop(columns=["apply_date"])
        spec = PipelineSpec(
            prefilter={"max_corr": 0.9, "iv_range": None, "max_psi": None},
            binning={"method": "chi", "n_bins": 4},
            refine={"iv_range": (0.02, None), "max_corr": 0.9},
        )
        p = (
            ProScore()
            .read(train=train, target="bad_flag")
            .apply(spec)
            .prefilter()
            .bin()
        )
        assert p._binner is not None
        assert p._binner.method == "chi"
        assert p._binner.n_bins == 4

    def test_apply_rules_via_mine_rules(self, full_df):
        train = full_df.drop(columns=["apply_date"])
        spec = PipelineSpec(
            prefilter={"max_corr": 0.9, "iv_range": None, "max_psi": None},
            binning={"method": "chi", "n_bins": 4},
            refine={"iv_range": (0.02, None)},
            rules={"method": "tree", "max_tree_depth": 2, "min_lift": 1.0, "max_rules": 5},
        )
        p = (
            ProScore()
            .read(train=train, target="bad_flag")
            .apply(spec)
            .prefilter()
            .bin()
            .refine()
            .mine_rules()
        )
        assert p._rulemine is not None
        assert p._rulemine.method == "tree"
        assert p._rulemine.max_tree_depth == 2

    def test_select_ignores_method_kwarg(self, full_df):
        train = full_df.drop(columns=["apply_date"])
        p = ProScore()
        p.read(train=train, target="bad_flag")
        p.prefilter(max_corr=0.9, iv_range=None, max_psi=None)
        p.bin(method="chi", n_bins=4)
        p.refine(iv_range=(0.02, None))
        p.transform()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            p.select(method="stepwise", n_min=2, n_max=4, force_fill=True, max_iter_round=3)
        assert any("method=" in str(w.message) for w in caught)
        assert p._selector is not None
