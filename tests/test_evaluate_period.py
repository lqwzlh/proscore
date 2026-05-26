"""Tests for evaluate_by_period."""

from __future__ import annotations

import pandas as pd

from proscore.evaluate import evaluate_by_period


class TestEvaluateByPeriod:
    def test_by_year_from_dataframe(self, full_df, woe_train_test, binning_result):
        from proscore.modeling import ScoreCard
        from proscore.selection import StepwiseSelector
        from proscore.transform import WOETransformer

        train, _test, _ = (
            full_df[full_df["apply_date"].dt.year <= 2021],
            full_df[full_df["apply_date"].dt.year == 2022],
            full_df[full_df["apply_date"].dt.year == 2023],
        )
        oot = full_df[full_df["apply_date"].dt.year >= 2022].copy()

        wt = WOETransformer().fit(binning_result.bin_table_)
        num = [c for c in full_df.columns if c not in ("bad_flag", "apply_date")
               and pd.api.types.is_numeric_dtype(full_df[c])]
        train_woe = wt.transform(train[num])
        train_woe["bad_flag"] = train["bad_flag"].values

        ss = StepwiseSelector(n_min=3, n_max=6, force_fill=True, max_iter_round=5)
        ss.fit(train_woe, train_woe["bad_flag"], candidates=num)

        sc = ScoreCard(odds=20, pdo=20, base_score=600)
        sc.fit(train_woe, y="bad_flag", features=ss.support_)

        oot_woe = wt.transform(oot[num])
        periods = {
            str(int(yr)): (oot_woe.loc[g.index, ss.support_], g["bad_flag"])
            for yr, g in oot.groupby(oot["apply_date"].dt.year)
        }

        result = evaluate_by_period(
            sc.model_,
            train_woe[ss.support_],
            train_woe["bad_flag"],
            periods=periods,
            features=ss.support_,
        )
        assert len(result) == 2
        assert set(result["period"]) == {"2022", "2023"}
        assert (result["ks"] >= 0).all()
