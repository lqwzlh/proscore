"""Tests for proscore.utils — load_presets."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from proscore.utils import PresetResult, load_presets


@pytest.fixture(scope="module")
def presets_path() -> str:
    return f"{__file__.rsplit('/', 1)[0] if '/' in __file__ else '.'}/variable_presets.xlsx"


class TestLoadPresets:
    def test_returns_preset_result(self, presets_path):
        result = load_presets(presets_path)
        assert isinstance(result, PresetResult)

    def test_feature_config_is_dict(self, presets_path):
        result = load_presets(presets_path)
        assert isinstance(result.feature_config, dict)
        assert len(result.feature_config) > 0

    def test_feature_belong_is_dict(self, presets_path):
        result = load_presets(presets_path)
        assert isinstance(result.feature_belong, dict)
        assert len(result.feature_belong) > 0

    def test_income_has_monotonic(self, presets_path):
        result = load_presets(presets_path)
        cfg = result.feature_config["income"]
        assert cfg["monotonic"] == "decreasing"

    def test_income_has_special_values(self, presets_path):
        result = load_presets(presets_path)
        cfg = result.feature_config["income"]
        assert "special_values" in cfg
        assert -999 in cfg["special_values"]

    def test_debt_level_belong(self, presets_path):
        result = load_presets(presets_path)
        # "负债水平" dimension should have debt_ratio and utilization
        assert "负债水平" in result.feature_belong
        assert "debt_ratio" in result.feature_belong["负债水平"]
        assert "utilization" in result.feature_belong["负债水平"]

    def test_categorical_vars_no_monotonic(self, presets_path):
        result = load_presets(presets_path)
        # education has empty monotonic → should NOT appear in feature_config
        assert "education" not in result.feature_config

    def test_empty_dimension_not_in_belong(self, presets_path):
        result = load_presets(presets_path)
        # Variables with empty dimension should not be in feature_belong
        all_belong_vars = {v for vals in result.feature_belong.values() for v in vals}
        assert "bad_flag" not in all_belong_vars


class TestMonotonicValidation:
    def test_invalid_monotonic_warns(self, tmp_path):
        p = tmp_path / "bad_mono.xlsx"
        df = pd.DataFrame([
            {"variable": "x", "monotonic": "inc"},
            {"variable": "y", "monotonic": "increasing"},
        ])
        df.to_excel(p, index=False, engine="openpyxl", sheet_name="variables")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = load_presets(p)
            assert len(w) >= 1
            assert "inc" in str(w[0].message)

        # y with valid monotonic should still be included
        assert "y" in r.feature_config
        # x with invalid monotonic should NOT be included
        assert "x" not in r.feature_config


class TestEmptyAndMissing:
    def test_missing_monotonic_column(self, tmp_path):
        p = tmp_path / "missing_col.xlsx"
        df = pd.DataFrame([{"variable": "x"}])
        df.to_excel(p, index=False, engine="openpyxl", sheet_name="variables")

        result = load_presets(p)
        assert result.feature_config == {}

    def test_empty_special_values(self, tmp_path):
        p = tmp_path / "empty_spec.xlsx"
        df = pd.DataFrame([{
            "variable": "x",
            "monotonic": "decreasing",
            "special_values": "",
        }])
        df.to_excel(p, index=False, engine="openpyxl", sheet_name="variables")

        result = load_presets(p)
        cfg = result.feature_config["x"]
        assert "special_values" not in cfg
