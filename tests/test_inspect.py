"""Tests for proscore.inspect — detect, quality, stability, vif, correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proscore.inspect import correlation, detect, quality, stability, vif


class TestDetect:
    def test_returns_dataframe(self, sample_df):
        result = detect(sample_df, target="target")
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_columns_present(self, sample_df):
        result = detect(sample_df, target="target")
        for col in ["variable", "dtype", "count", "missing_pct", "n_unique"]:
            assert col in result.columns

    def test_target_excluded(self, sample_df):
        result = detect(sample_df, target="target")
        assert "target" not in result["variable"].values

    def test_missing_rate_range(self, sample_df):
        result = detect(sample_df, target="target")
        assert result["missing_pct"].between(0, 100).all()

    def test_detect_on_full_data(self, full_df):
        result = detect(full_df, target="bad_flag")
        assert len(result) >= 10


class TestQuality:
    def test_returns_dataframe(self, sample_df):
        result = quality(sample_df, target="target")
        assert isinstance(result, pd.DataFrame)

    def test_iv_auc_ks_present(self, sample_df):
        result = quality(sample_df, target="target")
        for col in ["iv", "auc", "ks"]:
            assert col in result.columns

    def test_iv_range(self, sample_df):
        result = quality(sample_df, target="target")
        iv_vals = result["iv"].dropna()
        assert (iv_vals >= 0).all()

    def test_auc_range(self, sample_df):
        result = quality(sample_df, target="target")
        auc_vals = result["auc"].dropna()
        assert auc_vals.between(0, 1).all()

    def test_ks_range(self, sample_df):
        result = quality(sample_df, target="target")
        ks_vals = result["ks"].dropna()
        assert ks_vals.between(0, 1).all()

    @pytest.mark.parametrize("estimator", ["decisiontree", "xgb", "lgb"])
    def test_estimators(self, sample_df, estimator):
        try:
            result = quality(sample_df, target="target", estimator=estimator)
            assert isinstance(result, pd.DataFrame)
        except (ImportError, ValueError) as e:
            pytest.skip(str(e))


class TestStability:
    def test_returns_dataframe(self, full_df):
        result = stability(
            full_df, target="bad_flag", time_col="apply_date",
            features=["income", "debt_ratio"],
        )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_has_psi_column(self, full_df):
        result = stability(
            full_df, target="bad_flag", time_col="apply_date",
            features=["income"],
        )
        # psi column may be "psi" or "psi_vs_first" depending on version
        has_psi_col = "psi" in result.columns or "psi_vs_first" in result.columns
        assert has_psi_col

    def test_has_separate_stability_flags(self, full_df):
        result = stability(
            full_df, target="bad_flag", time_col="apply_date",
            features=["income"],
        )
        assert "psi_flag" in result.columns
        assert "bad_rate_flag" in result.columns
        assert "stability" not in result.columns

    def test_psi_and_bad_rate_flags_independent(self, full_df):
        """PSI unstable does not force bad_rate trending, and vice versa."""
        result = stability(
            full_df, target="bad_flag", time_col="apply_date",
            features=["income", "debt_ratio"],
        )
        non_base = result[result["time_period"] != result["time_period"].min()]
        if len(non_base) == 0:
            return
        # Columns are evaluated separately — no merged label
        assert set(non_base["psi_flag"].unique()).issubset({"stable", "unstable"})
        assert set(non_base["bad_rate_flag"].unique()).issubset(
            {"stable", "trending_up", "trending_down"}
        )

    def test_multiple_periods(self, full_df):
        result = stability(
            full_df, target="bad_flag", time_col="apply_date",
            features=["income"],
        )
        n_periods = full_df["apply_date"].dt.year.nunique()
        time_col = "time_period" if "time_period" in result.columns else "period"
        assert result[time_col].nunique() >= n_periods - 1


class TestVIF:
    def test_returns_dataframe(self, sample_df):
        num_cols = ["x1", "x2", "x4"]
        result = vif(sample_df[num_cols].dropna())
        # vif() returns a DataFrame with 'variable' and 'vif' columns
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "vif" in result.columns

    def test_vif_positive(self, sample_df):
        num_cols = ["x1", "x2", "x4"]
        result = vif(sample_df[num_cols].dropna())
        vif_vals = result["vif"]
        assert (vif_vals >= 1).all()


class TestCorrelation:
    def test_returns_dataframe(self, sample_df):
        num_cols = ["x1", "x2", "x4"]
        # correlation() returns high-correlation pairs, not a matrix
        result = correlation(sample_df[num_cols], method="pearson", threshold=0.01)
        assert isinstance(result, pd.DataFrame)
        assert "var1" in result.columns
        assert "var2" in result.columns
        assert "corr" in result.columns

    def test_threshold_filters(self, sample_df):
        num_cols = ["x1", "x2"]
        result = correlation(sample_df[num_cols], method="pearson", threshold=0.999)
        assert len(result) == 0
