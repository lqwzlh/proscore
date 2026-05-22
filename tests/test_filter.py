"""Tests for proscore.selection — Filter."""

from __future__ import annotations

import pandas as pd
import pytest

from proscore.selection import Filter


class TestFilter:
    def test_fit_returns_self(self, full_df, num_features):
        f = Filter()
        out = f.fit(full_df[num_features], full_df["bad_flag"])
        assert out is f

    def test_support_is_list(self, full_df, num_features):
        f = Filter()
        f.fit(full_df[num_features], full_df["bad_flag"])
        assert isinstance(f.support_, list)
        assert len(f.support_) > 0

    def test_iv_range_filters(self, full_df, num_features):
        f = Filter(iv_range=(0.05, 100), max_corr=None, max_vif=None)
        f.fit(full_df[num_features], full_df["bad_flag"])
        # Should keep some, drop some
        assert len(f.support_) <= len(num_features)

    def test_max_corr_filters(self, full_df, num_features):
        f = Filter(max_corr=0.9, max_vif=None)
        f.fit(full_df[num_features], full_df["bad_flag"])
        assert isinstance(f.support_, list)

    def test_max_vif_filters(self, full_df, num_features):
        f = Filter(max_vif=15, max_corr=None)
        f.fit(full_df[num_features], full_df["bad_flag"])
        assert isinstance(f.support_, list)

    def test_quality_dataframe(self, full_df, num_features):
        f = Filter()
        f.fit(full_df[num_features], full_df["bad_flag"])
        assert isinstance(f.quality_, pd.DataFrame)
        assert "selected" in f.quality_.columns
        assert "reason" in f.quality_.columns

    def test_with_bin_table(self, full_df, num_features, binning_result):
        f = Filter(iv_range=(0.02, 100))
        f.fit(
            full_df[num_features], full_df["bad_flag"],
            bin_table=binning_result.bin_table_,
        )
        assert isinstance(f.support_, list)
        # Variables with real bin_table IV should have source='bin_table'
        assert "source" in f.iv_.columns

    def test_n_selected_top_n(self, full_df, num_features):
        f = Filter(n_selected=3, max_corr=None, max_vif=None,
                   iv_range=(0.0, None))
        f.fit(full_df[num_features], full_df["bad_flag"])
        assert len(f.support_) == 3

    def test_iv_equalfreq_without_bin_table(self, full_df, num_features):
        f = Filter(iv_range=(0.02, None), max_corr=None, max_vif=None)
        f.fit(full_df[num_features], full_df["bad_flag"])
        assert (f.iv_["source"] == "equalfreq").any()

    def test_reason_nonempty_when_dropped(self, full_df, num_features):
        f = Filter(iv_range=(0.5, None), max_corr=None, max_vif=None)
        f.fit(full_df[num_features], full_df["bad_flag"])
        dropped = f.quality_[f.quality_["dropped"]]
        assert dropped["reason"].str.contains("iv").any()
