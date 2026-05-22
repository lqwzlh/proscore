"""Tests for WOE extreme-value handling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from proscore.binning._woe import calc_iv_woe, normalize_woe


class TestNormalizeWoe:
    def test_neg_inf_maps_to_minus_two(self):
        assert normalize_woe(float("-inf")) == -2.0

    def test_large_negative_caps_at_minus_two(self):
        assert normalize_woe(-20.0) == -2.0

    def test_pos_inf_raises(self):
        with pytest.raises(ValueError, match="不正确分箱"):
            normalize_woe(float("inf"))

    def test_finite_unchanged(self):
        assert normalize_woe(0.35) == pytest.approx(0.35)


class TestCalcIvWoe:
    def test_all_good_bin_woe_capped(self):
        regroup = np.array([
            [0, 10, 50],
            [1, 0, 30],
        ], dtype=np.float64)
        _, _, woe = calc_iv_woe(regroup)
        assert woe[1] == -2.0

    def test_all_bad_bin_positive_finite(self):
        regroup = np.array([
            [0, 40, 0],
            [1, 10, 5],
        ], dtype=np.float64)
        _, _, woe = calc_iv_woe(regroup)
        assert woe[0] == 20.0
        assert not math.isinf(woe[0])
