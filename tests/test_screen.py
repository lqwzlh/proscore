"""Tests for feature screening outcomes."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from proscore.report import ReportBuilder
from proscore.selection import Filter, assess_screen
from proscore.selection._screen import FeatureScreenWarning


class TestAssessScreen:
    def test_ok_when_features_remain(self):
        out = assess_screen(["a", "b"], stage="test", n_candidates=5)
        assert out.ok is True
        assert out.n_selected == 2

    def test_warn_when_empty(self):
        with pytest.warns(FeatureScreenWarning):
            out = assess_screen([], stage="粗筛", n_candidates=6)
        assert out.ok is False
        assert "6" in out.message


class TestFilterExhausted:
    def test_empty_fit_no_raise(self):
        df = pd.DataFrame({"bad_flag": [0, 1, 0, 1]})
        f = Filter()
        with pytest.warns(FeatureScreenWarning):
            f.fit(df.iloc[:, :0], df["bad_flag"])
        assert f.exhausted_ is True
        assert "dropped" in f.quality_.columns


class TestReportExhausted:
    def test_report_builds_when_prefilter_drops_all(self, full_df):
        num = [c for c in full_df.columns if c != "bad_flag" and pd.api.types.is_numeric_dtype(full_df[c])]
        f = Filter(min_auc=0.52, max_corr=0.75, max_vif=10, iv_range=None, max_psi=None)
        f.fit(full_df[num], full_df["bad_flag"])
        assert f.exhausted_ is True
        rb = ReportBuilder()
        rb.with_filter(f, stage="粗筛")
        text = rb.build()
        assert "无保留变量" in text or "均未保留" in text
        assert "流水线提示" in text
