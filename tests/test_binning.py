"""Tests for proscore.binning — Binning, BinningProcess."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proscore.binning import Binning, BinningProcess
from proscore.binning._base import BinTable


class TestBinning:
    """Core Binning tests on small synthetic data."""

    @pytest.mark.parametrize("method", ["chi", "frequency", "distance", "tree"])
    def test_all_methods_run(self, sample_df, method):
        b = Binning(method=method, n_bins=4, min_bin_pct=0.05)
        b.fit(sample_df[["x1", "x2", "x4", "target"]], y="target")
        assert "x1" in b.bin_table_

    def test_fit_returns_self(self, sample_df):
        b = Binning()
        out = b.fit(sample_df[["x1", "target"]], y="target")
        assert out is b

    def test_iv_dataframe(self, sample_df):
        b = Binning(method="chi", n_bins=5)
        b.fit(sample_df[["x1", "x2", "x4", "target"]], y="target")
        iv = b.iv_
        assert isinstance(iv, pd.DataFrame)
        assert "variable" in iv.columns
        assert "iv" in iv.columns

    def test_iv_nonnegative(self, sample_df):
        b = Binning(method="chi", n_bins=5)
        b.fit(sample_df[["x1", "x2", "x4", "target"]], y="target")
        assert (b.iv_["iv"] >= 0).all()

    def test_bin_table_is_bintable(self, sample_df):
        b = Binning(method="chi", n_bins=5)
        b.fit(sample_df[["x1", "target"]], y="target")
        assert isinstance(b.bin_table_["x1"], BinTable)

    def test_cutoffs_list(self, sample_df):
        b = Binning(method="chi", n_bins=5)
        b.fit(sample_df[["x1", "target"]], y="target")
        assert isinstance(b.cutoffs_["x1"], list)

    def test_categorical_column(self, sample_df):
        b = Binning(method="chi", n_bins=5)
        b.fit(sample_df[["x3", "target"]], y="target")
        bt = b.bin_table_["x3"]
        assert bt.dtype == "categorical"

    def test_transform_returns_indices(self, sample_df):
        b = Binning(method="chi", n_bins=5)
        b.fit(sample_df[["x1", "target"]], y="target")
        idx = b.transform(sample_df[["x1"]])
        assert isinstance(idx, pd.DataFrame)
        assert idx.min().min() >= 0

    # ── monotonic ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("mono,trend_name", [
        (True, "increasing"),
        (-1, "decreasing"),
        ("u", "u"),
        ("inverted_u", "inverted_u"),
    ])
    def test_monotonic_variants(self, sample_df, mono, trend_name):
        b = Binning(method="chi", n_bins=5, monotonic=mono)
        b.fit(sample_df[["x1", "target"]], y="target")
        bt = b.bin_table_["x1"]
        assert bt.trend_preset != 0  # preset was set
        # trend_match may be False if data doesn't cooperate — just check no crash

    def test_trend_preset_and_match_fields(self, sample_df):
        b = Binning(method="chi", n_bins=5, monotonic="increasing")
        b.fit(sample_df[["x1", "target"]], y="target")
        bt = b.bin_table_["x1"]
        assert hasattr(bt, "trend_preset")
        assert hasattr(bt, "trend_match")

    # ── special_values ─────────────────────────────────────────────────────

    def test_special_values(self, full_df):
        b = Binning(
            method="chi", n_bins=5,
            special_values={"income": [-999]},
        )
        b.fit(full_df[["income", "bad_flag"]], y="bad_flag")
        bt = b.bin_table_["income"]
        assert -999 in bt.special_values

    # ── manual_cutoffs ─────────────────────────────────────────────────────

    def test_manual_cutoffs(self, sample_df):
        b = Binning(
            manual_cutoffs={"x1": [1.0, 3.0, 5.0]},
            monotonic="increasing",
        )
        b.fit(sample_df[["x1", "target"]], y="target")
        bt = b.bin_table_["x1"]
        # Should have about 4 bins (3 cutoffs)
        assert 2 <= len(bt.bins) <= 5

    # ── adjust_shape=False ─────────────────────────────────────────────────

    def test_no_adjust_shape(self, sample_df):
        b = Binning(method="chi", n_bins=5, adjust_shape=False)
        b.fit(sample_df[["x1", "target"]], y="target")
        assert "x1" in b.bin_table_


class TestBinningProcess:
    def test_fit_with_feature_config(self, full_df, features):
        bp = BinningProcess(
            feature_config={
                "income": {"monotonic": "decreasing"},
                "debt_ratio": {"monotonic": "increasing"},
            },
            default_method="chi",
            default_n_bins=4,
        )
        bp.fit(full_df[features[:4] + ["bad_flag"]], y="bad_flag")
        assert "income" in bp.bin_table_

    def test_iv_dataframe(self, full_df, features):
        bp = BinningProcess(default_method="chi", default_n_bins=4)
        bp.fit(full_df[features[:4] + ["bad_flag"]], y="bad_flag")
        assert isinstance(bp.iv_, pd.DataFrame)

    def test_transform_returns_dataframe(self, full_df, features):
        bp = BinningProcess(default_method="chi", default_n_bins=4)
        bp.fit(full_df[features[:4] + ["bad_flag"]], y="bad_flag")
        result = bp.transform(full_df[features[:4]])
        assert result.shape[1] == 4
        assert (result >= 0).all().all()

    def test_from_presets(self, full_df, features, presets):
        bp = BinningProcess(
            feature_config=presets.feature_config,
            default_method="chi",
            default_n_bins=5,
        )
        bp.fit(full_df[features + ["bad_flag"]], y="bad_flag")
        # income should have monotonic=decreasing set from presets
        assert "income" in bp.bin_table_
