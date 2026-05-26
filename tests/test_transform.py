"""Tests for proscore.transform — WOETransformer."""

from __future__ import annotations

import pandas as pd
import pytest

from proscore.transform import WOETransformer


class TestWOETransformer:
    def test_fit_sets_bintable(self, binning_result):
        wt = WOETransformer()
        wt.fit(binning_result.bin_table_)
        assert wt._bin_tables is not None

    def test_transform_shape(self, woe_transformer, full_df, num_features):
        X = full_df[num_features].fillna(0)
        result = woe_transformer.transform(X)
        assert result.shape[0] == full_df.shape[0]
        assert result.shape[1] == len(num_features)

    def test_transform_returns_dataframe(self, woe_transformer, full_df, num_features):
        X = full_df[num_features].fillna(0)
        result = woe_transformer.transform(X)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == num_features

    def test_transform_all_numeric(self, woe_transformer, full_df, num_features):
        X = full_df[num_features].fillna(0)
        result = woe_transformer.transform(X)
        assert result.dtypes.apply(pd.api.types.is_numeric_dtype).all()

    def test_with_nans(self, woe_transformer, full_df, num_features):
        """NaN values should not crash transform."""
        X = full_df[num_features].copy()
        result = woe_transformer.transform(X)
        assert result.shape[0] == full_df.shape[0]

    @pytest.mark.parametrize("strategy", ["worst", "most_common", "missing", "zero"])
    def test_unseen_strategies(self, binning_result, strategy):
        wt = WOETransformer(unseen_strategy=strategy)
        wt.fit(binning_result.bin_table_)
        # strategies should be set without error
        assert wt.unseen_strategy == strategy

    def test_fit_transform_shortcut(self, binning_result, full_df, num_features):
        """fit_transform convenience."""
        wt = WOETransformer(unseen_strategy="worst")
        result = wt.fit_transform(
            binning_result.bin_table_,
            full_df[num_features].fillna(0),
        )
        assert result.shape[0] == full_df.shape[0]

    def test_unseen_categories_handled(self, binning_result, full_df):
        """WOE transform with categorical column that has unseen values in test."""
        cat_cols = ["education", "employment_type", "loan_purpose"]
        train = full_df[full_df["apply_date"].dt.year <= 2021]
        test = full_df[full_df["apply_date"].dt.year == 2023]

        from proscore.binning import Binning
        b_cat = Binning(method="chi", n_bins=5)
        b_cat.fit(train[cat_cols + ["bad_flag"]], y="bad_flag")

        wt = WOETransformer(unseen_strategy="worst")
        wt.fit(b_cat.bin_table_)
        result = wt.transform(test[cat_cols])
        assert result.shape[0] == test.shape[0]
        assert not result.isna().all().any()
