"""Integration tests — full pipeline chain."""

from __future__ import annotations

import pytest

import proscore as ps


def _train_test(full_df, train_until=2022, test_year=2023):
    """Helper: split full_df by year."""
    train = full_df[full_df["apply_date"].dt.year <= train_until].drop(
        columns=["apply_date"]
    )
    test = full_df[full_df["apply_date"].dt.year == test_year].drop(
        columns=["apply_date"]
    )
    return train, test


class TestChainAPI:
    def test_full_chain_smoke(self, full_df):
        """End-to-end chain API with train + test."""
        train, test = _train_test(full_df)
        p = ps.ProScore()

        p.read(train=train, test=test, target="bad_flag")
        p.detect()
        p.prefilter(max_corr=0.9, max_vif=15, iv_range=None, max_psi=None)
        assert len(p.prefilter_.support_) > 0

        p.bin(method="chi", n_bins=5)
        p.refine(iv_range=(0.02, None), max_corr=0.9, max_vif=15)
        assert len(p.filter_.support_) > 0

        p.transform(unseen_strategy="worst")
        p.select( n_min=3, n_max=8, force_fill=True, max_iter_round=6)
        p.fit(odds=20, pdo=20, base_score=600)
        p.scorecard()
        p.evaluate()

        assert "test_ks" in p.eval_result

    def test_full_chain_train_only(self, full_df):
        train = full_df.drop(columns=["apply_date"])
        p = ps.ProScore()
        p.read(train=train, target="bad_flag")
        p.prefilter(max_corr=0.9, max_vif=15, iv_range=None, max_psi=None)
        p.bin(method="chi", n_bins=5)
        p.refine(iv_range=(0.02, None), max_corr=0.9, max_vif=15)
        p.transform()
        p.select( n_min=3, n_max=8, force_fill=True, max_iter_round=6)
        p.fit(odds=20, pdo=20, base_score=600)
        p.scorecard()
        p.evaluate()
        assert "trn_ks" in p.eval_result

    def test_full_chain_with_oot(self, full_df):
        train = full_df[full_df["apply_date"].dt.year <= 2021].drop(columns=["apply_date"])
        test = full_df[full_df["apply_date"].dt.year == 2022].drop(columns=["apply_date"])
        oot = full_df[full_df["apply_date"].dt.year == 2023].drop(columns=["apply_date"])

        p = ps.ProScore()
        p.read(train=train, test=test, oot=oot, target="bad_flag")
        p.prefilter(max_corr=0.9, iv_range=None, max_psi=None)
        p.bin(method="chi", n_bins=5)
        p.refine(iv_range=(0.02, None), max_corr=0.9)
        p.transform()
        p.select( n_min=3, n_max=8, force_fill=True, max_iter_round=6)
        p.fit(odds=20, pdo=20, base_score=600)
        p.scorecard()
        p.evaluate()
        assert "oot_ks" in p.eval_result

    def test_prefilter_without_bin(self, full_df):
        train = full_df.drop(columns=["apply_date"])
        p = ps.ProScore()
        p.read(train=train, target="bad_flag")
        p.prefilter(max_corr=0.9, iv_range=None, max_psi=None)
        assert p.prefilter_ is not None

    def test_refine_requires_bin(self, full_df):
        train = full_df.drop(columns=["apply_date"])
        p = ps.ProScore()
        p.read(train=train, target="bad_flag")
        p.prefilter(iv_range=None, max_psi=None)
        with pytest.raises(RuntimeError, match="bin"):
            p.refine()

    def test_transform_requires_refine(self, full_df):
        train = full_df.drop(columns=["apply_date"])
        p = ps.ProScore()
        p.read(train=train, target="bad_flag")
        p.bin(method="chi", n_bins=5)
        with pytest.raises(RuntimeError, match="refine"):
            p.transform()

    def test_filter_alias_for_refine(self, full_df):
        train = full_df.drop(columns=["apply_date"])
        p = ps.ProScore()
        p.read(train=train, target="bad_flag")
        p.prefilter(iv_range=None, max_psi=None)
        p.bin(method="chi", n_bins=5)
        p.filter(iv_range=(0.02, None))
        assert p.filter_ is p.refine_

    def test_properties_before_calls(self):
        p = ps.ProScore()
        assert p.support_ == []
