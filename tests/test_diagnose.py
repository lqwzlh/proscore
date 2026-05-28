"""Comprehensive tests for proscore.evaluate.diagnose (new in v0.2)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import proscore as ps
from proscore.evaluate import DiagnosisIssue, DiagnosisReport, diagnose

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_synthetic(
    n: int = 1200,
    bad_rate: float = 0.09,
    seed: int = 42,
    add_leak: bool = False,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "x1": rng.randn(n),
            "x2": rng.randn(n),
            "x3": rng.beta(2, 5, n),
            "bad_flag": (rng.rand(n) < bad_rate).astype(int),
        }
    )
    if add_leak:
        # Inject a near-perfect predictor to trigger high KS/AUC info
        df["leak"] = (df["bad_flag"] * 0.9 + rng.randn(n) * 0.1).clip(0, 1)
    return df


def _train_test_split(df: pd.DataFrame, target: str = "bad_flag"):
    return train_test_split(
        df, test_size=0.3, stratify=df[target], random_state=0
    )


# ── basic contract ───────────────────────────────────────────────────────────


def test_diagnose_returns_report():
    rep = diagnose({})
    assert isinstance(rep, DiagnosisReport)
    assert not rep  # empty is falsy


def test_diagnosis_report_properties_and_to_dataframe():
    issues = [
        DiagnosisIssue("critical", "discrimination", "t1", "e1", "s1"),
        DiagnosisIssue("warning", "overfitting", "t2", "e2", "s2"),
        DiagnosisIssue("info", "variable", "t3", "e3", "s3", culprit_vars=["x"]),
    ]
    rep = DiagnosisReport(issues=issues)
    assert len(rep.critical) == 1
    assert len(rep.warnings) == 1
    assert len(rep.infos) == 1
    assert bool(rep) is True

    df = rep.to_dataframe()
    assert list(df.columns) == ["level", "category", "title", "evidence", "suggestion"]
    assert len(df) == 3
    assert "x" in str(rep)  # __str__ contains culprit


def test_empty_report_formatting():
    rep = DiagnosisReport()
    s = str(rep)
    assert "未发现异常" in s
    assert "KS≥0.20" in s


# ── discrimination layer ─────────────────────────────────────────────────────


def test_discrimination_critical_and_high_end_info():
    ev = {"test_ks": 0.12, "test_auc": 0.55, "trn_ks": 0.15, "trn_auc": 0.58}
    rep = diagnose(ev)
    titles = [i.title for i in rep.issues]
    assert any("KS 不可用" in t for t in titles)
    assert any("AUC 不可用" in t for t in titles)

    # high-end info triggers
    ev2 = {"test_ks": 0.65, "test_auc": 0.97, "trn_ks": 0.68, "trn_auc": 0.96}
    rep2 = diagnose(ev2)
    titles2 = [i.title for i in rep2.issues]
    assert any("KS 异常高" in t for t in titles2)
    assert any("AUC 异常高" in t for t in titles2)


# ── overfitting layer (relative ratio) ───────────────────────────────────────


def test_overfitting_relative_ratio():
    ev = {
        "trn_ks": 0.40,
        "test_ks": 0.25,  # ratio = 0.375 > 0.30 → critical
        "model_vars": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
    }
    rep = diagnose(ev)
    assert any("KS 衰退严重" in i.title for i in rep.issues)

    ev2 = {"trn_ks": 0.30, "test_ks": 0.24, "model_vars": ["x1"]}  # ~20% → warning
    rep2 = diagnose(ev2)
    assert any("中度过拟合" in i.title for i in rep2.issues)


# ── stability layer ──────────────────────────────────────────────────────────


def test_stability_psi_and_oot_decay():
    ev = {"psi": 0.28, "psi_oot": 0.31, "oot_ks": 0.10, "test_ks": 0.35}
    rep = diagnose(ev)
    titles = [i.title for i in rep.issues]
    assert any("显著漂移" in t for t in titles)
    assert any("OOT KS 衰减严重" in t for t in titles)


def test_auc_decay_warning():
    ev = {"trn_auc": 0.82, "test_auc": 0.61}  # relative loss ~ 65% > 20%
    rep = diagnose(ev)
    assert any("AUC 衰退" in i.title for i in rep.issues)


# ── variable quality + coef sign ─────────────────────────────────────────────


def test_variable_quality_and_trend_match(monkeypatch):
    # Fake bin_table_ structure
    class FakeBin:
        def __init__(self, iv, miss_rate=0.0, trend_match=True):
            self.iv_total = iv
            self.has_missing = miss_rate > 0
            self.trend_match = trend_match
            self.bins = []
            if self.has_missing:
                self.bins.append(type("B", (), {"bin_label": "missing", "count": int(1000 * miss_rate)})())
            # add a dummy normal bin so total > 0
            self.bins.append(type("B", (), {"bin_label": "normal", "count": int(1000 * (1 - miss_rate))})())

    class FakeBinning:
        bin_table_ = {
            "v1": FakeBin(iv=0.03, miss_rate=0.55),   # critical missing
            "v2": FakeBin(iv=0.62),                   # high IV warning
            "v3": FakeBin(iv=0.005, trend_match=False),  # low IV + trend mismatch
        }

    ev = {"model_vars": ["v1", "v2", "v3"]}
    rep = diagnose(ev, binning=FakeBinning())
    titles = [i.title for i in rep.issues]
    assert any("缺失严重" in t for t in titles)
    assert any("IV 异常高" in t for t in titles)
    assert any("IV 极低" in t for t in titles)
    assert any("分箱趋势与预设不符" in t for t in titles)


def test_coefficient_sign_contradiction():
    class FakeSelector:
        coef_sign = "positive"
        support_ = ["good_var", "bad_var"]
        model_ = type("M", (), {"params": {"good_var": 0.12, "bad_var": -0.07, "const": 0.0}})()

    rep = diagnose({"model_vars": ["good_var", "bad_var"]}, selector=FakeSelector())
    assert any("系数符号矛盾" in i.title and "bad_var" in i.culprit_vars for i in rep.issues)


# ── pre-model branch ─────────────────────────────────────────────────────────


def test_pre_model_diagnosis_branch():
    class FakeBin:
        def __init__(self, iv):
            self.iv_total = iv
            self.has_missing = False
            self.bins = [type("B", (), {"count": 100})()]

    class FakeBinner:
        bin_table_ = {"w1": FakeBin(0.03), "w2": FakeBin(0.008), "w3": FakeBin(0.015)}

    rep = diagnose(train_columns=["w1", "w2", "w3"], binning=FakeBinner())
    assert any("IV 不足" in i.title for i in rep.issues)


# ── bad rate info ────────────────────────────────────────────────────────────


def test_bad_rate_low_info():
    y = pd.Series([0] * 98 + [1] * 2)  # exactly 2% → does not trigger (< 0.02 strict)
    rep = diagnose(y_train=y)
    assert not any("坏样本率偏低" in i.title for i in rep.issues)

    y2 = pd.Series([0] * 99 + [1] * 1)  # 1% < 2%
    rep2 = diagnose(y_train=y2)
    assert any("坏样本率偏低" in i.title for i in rep2.issues)


# ── ProScore integration + print_report ──────────────────────────────────────


def test_proscore_diagnose_integration_and_print_report_flag():
    df = _make_synthetic(n=800, bad_rate=0.08)
    train, test = _train_test_split(df)
    p = (
        ps.ProScore()
        .read(train=train, test=test, target="bad_flag")
        .detect()
        .prefilter(iv_range=None, max_psi=None)
        .bin(method="frequency", n_bins=4)
        .refine()
        .transform()
        .select(n_min=1, n_max=2, force_fill=True)
        .fit()
        .scorecard()
        .evaluate()
    )

    # silent mode
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p2 = p.diagnose(print_report=False)
    assert p2.diagnosis_ is not None
    assert isinstance(p2.diagnosis_, DiagnosisReport)

    # default (prints) still works and returns self
    p3 = p.diagnose()
    assert p3 is p


# ── end-to-end with real-ish low performance data ────────────────────────────


def test_full_chain_low_perf_triggers_multiple_layers():
    df = _make_synthetic(n=600, bad_rate=0.05)  # deliberately weak signal
    train, test = _train_test_split(df)
    p = (
        ps.ProScore()
        .read(train=train, test=test, target="bad_flag")
        .detect()
        .prefilter()
        .bin(method="frequency")
        .refine()
        .transform()
        .select(n_min=1, n_max=2, force_fill=True, pvalue_threshold=0.5)
        .fit()
        .scorecard()
        .evaluate()
        .diagnose(print_report=False)
    )
    # Expect at least discrimination critical (very low KS on this data)
    assert len(p.diagnosis_.critical) >= 1 or len(p.diagnosis_.warnings) >= 1


# ── custom thresholds ────────────────────────────────────────────────────────


def test_custom_thresholds_override():
    from proscore.evaluate import DEFAULT_THRESHOLDS, diagnose

    # Very weak model
    ev = {"test_ks": 0.17, "test_auc": 0.62, "trn_ks": 0.20, "trn_auc": 0.65}

    # Default: 0.17 is between 0.15 and 0.20 → should be "warning" (KS 偏低), not critical
    rep_default = diagnose(ev)
    assert any("KS 偏低" in i.title for i in rep_default.issues)
    assert not any("KS 不可用" in i.title for i in rep_default.issues)

    # Raise the critical line to 0.18 → now 0.17 becomes critical
    custom = {"discrimination": {"ks_critical": 0.18}}
    rep_custom = diagnose(ev, thresholds=custom)
    assert any("KS 不可用" in i.title for i in rep_custom.issues)

    # Verify DEFAULT_THRESHOLDS is exported and has the expected shape
    assert "discrimination" in DEFAULT_THRESHOLDS
    assert DEFAULT_THRESHOLDS["discrimination"]["ks_critical"] == 0.15  # original default
