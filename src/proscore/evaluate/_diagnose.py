"""Model diagnosis: rule-based health check for scorecard quality."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ── reference thresholds ─────────────────────────────────────────────────────

_DISCRIMINATION_THRESHOLDS = {
    "ks_critical": 0.15,
    "ks_warning": 0.20,
    "ks_info": 0.30,
    "auc_critical": 0.60,
    "auc_warning": 0.70,
}

_OVERFITTING_THRESHOLDS = {
    "ks_reduce_ratio_critical": 0.30,
    "ks_reduce_ratio_warning": 0.15,
    "ks_reduce_ratio_info": 0.10,
}

_STABILITY_THRESHOLDS = {
    "psi_critical": 0.25,
    "psi_warning": 0.10,
    "ks_decay_critical": 0.30,
    "ks_decay_warning": 0.15,
    "auc_reduce_ratio_warning": 0.20,  # fraction of discriminatory power lost
}

_VARIABLE_THRESHOLDS = {
    "missing_critical": 0.50,
    "missing_warning": 0.30,
    "iv_suspicious": 0.50,
    "iv_weak": 0.02,
}

# Public: default thresholds (可被 diagnose(thresholds=...) 部分覆盖)
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "discrimination": _DISCRIMINATION_THRESHOLDS.copy(),
    "overfitting": _OVERFITTING_THRESHOLDS.copy(),
    "stability": _STABILITY_THRESHOLDS.copy(),
    "variable": _VARIABLE_THRESHOLDS.copy(),
}


@dataclass
class DiagnosisIssue:
    """A single diagnosed problem with actionable advice."""

    level: str  # "critical" | "warning" | "info"
    category: str  # "discrimination" | "overfitting" | "stability" | "variable"
    title: str
    evidence: str
    suggestion: str
    culprit_vars: list[str] = field(default_factory=list)


@dataclass
class DiagnosisReport:
    """Structured diagnosis output."""

    issues: list[DiagnosisIssue] = field(default_factory=list)

    @property
    def critical(self) -> list[DiagnosisIssue]:
        return [i for i in self.issues if i.level == "critical"]

    @property
    def warnings(self) -> list[DiagnosisIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def infos(self) -> list[DiagnosisIssue]:
        return [i for i in self.issues if i.level == "info"]

    def to_dataframe(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["level", "category", "title", "evidence", "suggestion"])
        return pd.DataFrame([
            {"level": i.level, "category": i.category, "title": i.title,
             "evidence": i.evidence, "suggestion": i.suggestion}
            for i in self.issues
        ])

    def __str__(self) -> str:
        return _format_report(self)

    def __bool__(self) -> bool:
        return len(self.issues) > 0


def diagnose(
    eval_result: dict | None = None,
    *,
    binning: Any = None,
    selector: Any = None,
    stability: pd.DataFrame | None = None,
    period_eval: pd.DataFrame | None = None,
    y_train: pd.Series | np.ndarray | None = None,
    train_columns: list[str] | None = None,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> DiagnosisReport:
    """Diagnose model health from evaluation results and upstream artefacts.

    Parameters
    ----------
    eval_result : dict or None
        Output of :func:`~proscore.evaluate.evaluate`.
    binning : Binning or BinningProcess or None
        Fitted binning (for IV / trend_match / missing info).
    selector : StepwiseSelector or None
        Fitted selector (for coefficient sign / variable count).
    stability : DataFrame or None
        Output of :func:`~proscore.inspect.stability` (for PSI per variable).
    period_eval : DataFrame or None
        Output of :func:`~proscore.evaluate.evaluate_by_period` (for time trend).
    y_train : Series or ndarray or None
        Training target (for bad-rate check).
    train_columns : list of str or None
        All candidate columns for pre-model diagnosis (when no eval_result).
    thresholds : dict or None
        自定义阈值，用于覆盖默认诊断标准。格式示例::

            {
                "discrimination": {"ks_critical": 0.18, "auc_warning": 0.72},
                "stability": {"psi_warning": 0.08},
            }

        未指定的类别/键会使用 DEFAULT_THRESHOLDS 中的默认值。
        详见 :data:`~proscore.evaluate.DEFAULT_THRESHOLDS`。
    """
    issues: list[DiagnosisIssue] = []

    th = _resolve_thresholds(thresholds)
    disc_th = th["discrimination"]
    over_th = th["overfitting"]
    stab_th = th["stability"]
    var_th = th["variable"]

    if eval_result:
        issues.extend(_check_discrimination(eval_result, disc_th))
        issues.extend(_check_overfitting(eval_result, selector, over_th))
        issues.extend(_check_stability(eval_result, stability, period_eval, stab_th))
        issues.extend(_check_variable_quality(eval_result, binning, selector, var_th))
        issues.extend(_check_coefficient_signs(selector))
        issues.extend(_check_binning_trends(binning))
    elif train_columns and binning:
        issues.extend(_check_pre_model(binning, train_columns, var_th))

    if y_train is not None:
        issues.extend(_check_bad_rate(y_train))

    return DiagnosisReport(issues=issues)


def _resolve_thresholds(user: dict[str, dict[str, float]] | None) -> dict[str, dict[str, float]]:
    """Merge user-provided overrides on top of DEFAULT_THRESHOLDS (per category)."""
    if not user:
        return DEFAULT_THRESHOLDS
    resolved = {cat: th.copy() for cat, th in DEFAULT_THRESHOLDS.items()}
    for cat, overrides in user.items():
        if cat in resolved and isinstance(overrides, dict):
            resolved[cat].update(overrides)
    return resolved


# ── check functions ──────────────────────────────────────────────────────────


def _check_discrimination(ev: dict, th: dict | None = None) -> list[DiagnosisIssue]:
    th = th or _DISCRIMINATION_THRESHOLDS
    issues: list[DiagnosisIssue] = []
    ks_key = "test_ks" if "test_ks" in ev else "trn_ks"
    ks = ev.get(ks_key, 0.0)
    auc_key = ks_key.replace("ks", "auc")
    auc = ev.get(auc_key, 0.5)

    if ks < th["ks_critical"]:
        issues.append(DiagnosisIssue(
            "critical", "discrimination", "KS 不可用",
            f"当前 {ks_key}={ks:.3f}，低于可用线 {th['ks_critical']}",
            _ks_suggestion(ev, ks),
        ))
    elif ks < th["ks_warning"]:
        issues.append(DiagnosisIssue(
            "warning", "discrimination", "KS 偏低",
            f"当前 {ks_key}={ks:.3f}，勉强可用",
            _ks_suggestion(ev, ks),
        ))
    elif ks < th["ks_info"]:
        issues.append(DiagnosisIssue(
            "info", "discrimination", "KS 一般",
            f"当前 {ks_key}={ks:.3f}，可接受但优化空间大",
            "可尝试放宽 pvalue_threshold 或更换分箱算法",
        ))

    if ks > 0.60:
        issues.append(DiagnosisIssue(
            "info", "discrimination", "KS 异常高",
            f"当前 {ks_key}={ks:.3f}，> 0.60 偏离常见范围",
            "可能存在信息泄漏；检查入模变量 IV 是否异常 > 0.5",
        ))

    if auc < th["auc_critical"]:
        issues.append(DiagnosisIssue(
            "critical", "discrimination", "AUC 不可用",
            f"当前 {auc_key}={auc:.3f}，接近随机",
            "① 检查目标定义是否正确（1=坏是否正确标注）；"
            "② 尝试放宽 pvalue_threshold 让更多变量入模（注意监管对显著性的要求）；"
            "③ 如果候选变量 IV 普遍偏低，需回到变量池补充有效变量",
        ))
    elif auc < th["auc_warning"]:
        issues.append(DiagnosisIssue(
            "warning", "discrimination", "AUC 偏低",
            f"当前 {auc_key}={auc:.3f}，区分力偏弱",
            "可尝试放宽 pvalue_threshold 或更换分箱算法（tree），同时关注 coef_sign 约束是否过严",
        ))

    if auc > 0.95:
        issues.append(DiagnosisIssue(
            "info", "discrimination", "AUC 异常高",
            f"当前 {auc_key}={auc:.3f}，> 0.95 偏离行业常见范围",
            "可能存在信息泄漏（如使用了未来信息、目标列泄漏到特征中）；"
            "建议逐一检查入模变量的 IV 和业务含义",
        ))

    return issues


def _ks_suggestion(ev: dict, ks: float) -> str:
    trn_ks = ev.get("trn_ks", 0)
    test_ks = ev.get("test_ks", trn_ks)
    parts = []
    n_vars = len(ev.get("model_vars", []))
    if n_vars > 0:
        parts.append(f"入模仅 {n_vars} 个变量" if n_vars < 3 else f"入模 {n_vars} 个变量")
    if ks < 0.20:
        parts.append("放宽 pvalue_threshold; 检查 prefilter 是否误杀有效变量")
    else:
        parts.append("尝试 tree 分箱或开启扰动")
    if trn_ks - test_ks > 0.10 and test_ks > 0:
        parts.append("Train/Test KS 差距大，n_max 降至 5-6")
    return "。".join(parts) + "。"


def _check_overfitting(ev: dict, selector: Any, th: dict | None = None) -> list[DiagnosisIssue]:
    th = th or _OVERFITTING_THRESHOLDS
    issues: list[DiagnosisIssue] = []
    trn_ks = ev.get("trn_ks", 0.0)
    test_ks = ev.get("test_ks", trn_ks)
    ratio = ((trn_ks - test_ks) / trn_ks) if trn_ks > 0 else 0.0

    if ratio > th["ks_reduce_ratio_critical"]:
        level, desc = "critical", "KS 衰退严重"
    elif ratio > th["ks_reduce_ratio_warning"]:
        level, desc = "warning", "KS 中度过拟合"
    elif ratio > th["ks_reduce_ratio_info"]:
        level, desc = "info", "KS 轻微衰退"
    else:
        return issues

    suggestion = ""
    n_vars = len(ev.get("model_vars", []))
    if selector is not None:
        if n_vars > 8:
            suggestion = f"入模 {n_vars} 个变量偏高，建议 n_max 降至 6-8"
        if getattr(selector, "perturbation", True) is False:
            suggestion += "；开启 perturbation"
    else:
        suggestion = f"入模 {n_vars} 个变量" + ("，偏高" if n_vars > 8 else "")

    if test_ks == trn_ks and test_ks == ev.get("trn_ks", 0):
        level, desc = "info", "仅有 Train KS（未传 Test 数据）"

    if n_vars > 12:
        issues.append(DiagnosisIssue(
            "info", "overfitting",
            "入模变量偏多", f"当前入模 {n_vars} 个变量",
            "变量 > 12 个有温和过拟合风险；如 KS 衰退也在告警线以上，建议降低 n_max",
        ))

    issues.append(DiagnosisIssue(
        level, "overfitting", desc,
        f"trn KS={trn_ks:.3f}, test KS={test_ks:.3f}, 衰退={ratio:.0%}",
        suggestion or "检查是否数据切分有问题",
    ))
    return issues


def _check_stability(
    ev: dict,
    stability: pd.DataFrame | None,
    period_eval: pd.DataFrame | None,
    th: dict | None = None,
) -> list[DiagnosisIssue]:
    th = th or _STABILITY_THRESHOLDS
    issues: list[DiagnosisIssue] = []

    # Score PSI (train vs test)
    psi = ev.get("psi", 0.0)
    if psi > th["psi_critical"]:
        issues.append(DiagnosisIssue(
            "critical", "stability", "评分分布显著漂移（vs test）",
            f"PSI={psi:.3f}，> {th['psi_critical']}",
            _psi_suggestion(stability, psi),
        ))
    elif psi > th["psi_warning"]:
        issues.append(DiagnosisIssue(
            "warning", "stability", "评分分布轻微漂移（vs test）",
            f"PSI={psi:.3f}",
            _psi_suggestion(stability, psi),
        ))

    # Score PSI (train vs OOT)
    psi_oot = ev.get("psi_oot", 0.0)
    if psi_oot > th["psi_critical"]:
        issues.append(DiagnosisIssue(
            "critical", "stability", "评分分布显著漂移（vs OOT）",
            f"OOT PSI={psi_oot:.3f}，> {th['psi_critical']}",
            _psi_suggestion(stability, psi_oot),
        ))
    elif psi_oot > th["psi_warning"]:
        issues.append(DiagnosisIssue(
            "warning", "stability", "评分分布轻微漂移（vs OOT）",
            f"OOT PSI={psi_oot:.3f}",
            _psi_suggestion(stability, psi_oot),
        ))

    # OOT KS decay
    oot_ks = ev.get("oot_ks", 0.0)
    if oot_ks > 0:
        test_ks = ev.get("test_ks", ev.get("trn_ks", oot_ks))
        decay = (test_ks - oot_ks) / test_ks if test_ks > 0 else 0.0
        if decay > _STABILITY_THRESHOLDS["ks_decay_critical"]:
            issues.append(DiagnosisIssue(
                "critical", "stability", "OOT KS 衰减严重",
                f"OOT KS={oot_ks:.3f}，相对 test 衰减 {decay:.0%}",
                _oot_decay_suggestion(stability, period_eval),
            ))
        elif decay > _STABILITY_THRESHOLDS["ks_decay_warning"]:
            issues.append(DiagnosisIssue(
                "warning", "stability", "OOT KS 衰减",
                f"OOT KS={oot_ks:.3f}，相对 test 衰减 {decay:.0%}",
                _oot_decay_suggestion(stability, period_eval),
            ))

    # AUC decay
    trn_auc = ev.get("trn_auc", 0.5)
    test_auc = ev.get("test_auc", trn_auc)
    auc_ratio = ((trn_auc - test_auc) / (trn_auc - 0.5)) if trn_auc > 0.5 else 0.0
    if auc_ratio > th["auc_reduce_ratio_warning"]:
        issues.append(DiagnosisIssue(
            "warning", "stability", "AUC 衰退",
            f"trn AUC={trn_auc:.3f}, test AUC={test_auc:.3f}, 相对衰减={auc_ratio:.0%}",
            "检查逐变量 PSI；关注 train 到 test 的区分力变化趋势",
        ))

    return issues


def _psi_suggestion(stability: pd.DataFrame | None, psi: float) -> str:
    if stability is not None and "psi_vs_first" in stability.columns:
        top_psi = (
            stability.groupby("variable")["psi_vs_first"]
            .max().sort_values(ascending=False)
        )
        culprits = top_psi[top_psi > 0.15]
        if len(culprits) > 0:
            names = ", ".join(f"{v}({culprits[v]:.3f})" for v in culprits.index[:3])
            return f"主因变量: {names}。检查对应变量分布变化原因；如无法修复，从模型移除"
    return "检查逐期 PSI 确认漂移来源；考虑缩短模型更新周期"


def _oot_decay_suggestion(
    stability: pd.DataFrame | None,
    period_eval: pd.DataFrame | None,
) -> str:
    parts = []
    if stability is not None and "stability" in stability.columns:
        unstable = stability[stability["stability"].isin(["unstable", "trending_down"])]
        if len(unstable) > 0:
            u_vars = unstable["variable"].unique()[:3]
            parts.append(f"不稳定变量: {', '.join(u_vars)}")
    if period_eval is not None and "bad_rate" in period_eval.columns:
        parts.append("检查各时期坏样本率是否一致")
    parts.append("建议缩短模型更新周期")
    return "；".join(parts)


def _check_variable_quality(
    ev: dict, binning: Any, selector: Any, th: dict | None = None
) -> list[DiagnosisIssue]:
    th = th or _VARIABLE_THRESHOLDS
    issues: list[DiagnosisIssue] = []
    model_vars = ev.get("model_vars", [])
    if not model_vars or binning is None:
        return issues

    bt = getattr(binning, "bin_table_", {})
    if not bt:
        return issues

    for v in model_vars:
        if v not in bt:
            continue
        tbl = bt[v]
        iv_val = getattr(tbl, "iv_total", 0.0)
        has_miss = getattr(tbl, "has_missing", False)

        if has_miss:
            miss_bin = next((b for b in tbl.bins if b.bin_label == "missing"), None)
            if miss_bin:
                total_count = sum(b.count for b in tbl.bins)
                miss_rate = miss_bin.count / total_count if total_count > 0 else 0
                if miss_rate > th["missing_critical"]:
                    issues.append(DiagnosisIssue(
                        "critical", "variable",
                        f"变量 {v} 缺失严重",
                        f"{v} 缺失率 {miss_rate:.0%}，> {th['missing_critical']:.0%}",
                        "缺失率 > 50% 严重影响稳定性；建议用中位数填充或删除该变量",
                        culprit_vars=[v],
                    ))
                elif miss_rate > th["missing_warning"]:
                    issues.append(DiagnosisIssue(
                        "warning", "variable",
                        f"变量 {v} 缺失偏高",
                        f"{v} 缺失率 {miss_rate:.0%}",
                        "建议用中位数填充或删除后重跑对比",
                        culprit_vars=[v],
                    ))

        if iv_val > th["iv_suspicious"]:
            issues.append(DiagnosisIssue(
                "warning", "variable",
                f"变量 {v} IV 异常高",
                f"{v} IV={iv_val:.3f} > {th['iv_suspicious']}，可能存在信息泄漏",
                "检查该变量是否包含未来信息或目标泄漏；如确认安全可保留",
                culprit_vars=[v],
            ))

        if iv_val < th["iv_weak"] and iv_val > 0:
            issues.append(DiagnosisIssue(
                "info", "variable",
                f"变量 {v} IV 极低",
                f"{v} IV={iv_val:.4f} < {th['iv_weak']}",
                "该变量区分力极弱，可能是强填入模；考虑移除后重跑",
                culprit_vars=[v],
            ))

    return issues


def _check_coefficient_signs(selector: Any) -> list[DiagnosisIssue]:
    if selector is None or getattr(selector, "model_", None) is None:
        return []
    issues: list[DiagnosisIssue] = []
    model = selector.model_
    params = getattr(model, "params", None)
    if params is None:
        return issues
    coef_sign = getattr(selector, "coef_sign", None)
    if coef_sign not in ("positive", "negative"):
        return issues
    support = getattr(selector, "support_", []) or []
    for var in support:
        try:
            # Support both dict-like (pandas Series) and .get() (some statsmodels wrappers)
            if hasattr(params, "__getitem__"):
                coef = params[var]
            else:
                coef = params.get(var) if hasattr(params, "get") else None
            if coef is None:
                continue
            coef = float(coef)
            if coef_sign == "positive" and coef < 0:
                issues.append(DiagnosisIssue(
                    "critical", "variable",
                    f"变量 {var} 系数符号矛盾",
                    f"{var} 系数={coef:.4f} < 0，与 {coef_sign} 约束矛盾",
                    "检查该变量分箱 WOE 方向；可能分箱趋势反了；或删除该变量",
                    culprit_vars=[var],
                ))
            elif coef_sign == "negative" and coef > 0:
                issues.append(DiagnosisIssue(
                    "critical", "variable",
                    f"变量 {var} 系数符号矛盾",
                    f"{var} 系数={coef:.4f} > 0，与 {coef_sign} 约束矛盾",
                    "检查该变量分箱 WOE 方向",
                    culprit_vars=[var],
                ))
        except (KeyError, TypeError, ValueError) as e:
            # Per-variable defensive only — never swallow the whole diagnosis.
            # Surface during development per AGENTS.md Fail Fast.
            import warnings
            warnings.warn(f"诊断时读取变量 {var} 系数失败: {e}", UserWarning, stacklevel=2)
            continue
    return issues


def _check_binning_trends(binning: Any) -> list[DiagnosisIssue]:
    if binning is None:
        return []
    bt = getattr(binning, "bin_table_", {})
    mismatches = []
    for v, tbl in bt.items():
        if getattr(tbl, "trend_match", True) is False:
            mismatches.append(v)
    if mismatches:
        return [DiagnosisIssue(
            "warning", "variable",
            "分箱趋势与预设不符",
            f"变量: {', '.join(mismatches[:5])}{'...' if len(mismatches) > 5 else ''}",
            "检查对应分箱图；如数据确实不支持预设趋势，修改 monotonic 或接受当前趋势",
            culprit_vars=mismatches,
        )]
    return []


def _check_pre_model(binning: Any, columns: list[str], th: dict | None = None) -> list[DiagnosisIssue]:
    """Pre-model: check IV spread and missing rates across all candidates."""
    th = th or _VARIABLE_THRESHOLDS
    issues: list[DiagnosisIssue] = []
    bt = getattr(binning, "bin_table_", {})
    iv_vals = {}
    missing_rates = {}
    for c in columns:
        if c in bt:
            tbl = bt[c]
            iv_vals[c] = getattr(tbl, "iv_total", 0.0)
            if getattr(tbl, "has_missing", False):
                total = sum(b.count for b in tbl.bins)
                miss = next((b for b in tbl.bins if b.bin_label == "missing"), None)
                if miss and total > 0:
                    missing_rates[c] = miss.count / total

    if iv_vals:
        strong = sum(1 for v in iv_vals.values() if v >= 0.02)
        weak = sum(1 for v in iv_vals.values() if 0.01 <= v < 0.02)
        none_ = sum(1 for v in iv_vals.values() if v < 0.01)
        plural = "" if len(columns) == 1 else "量"
        if strong < 3:
            issues.append(DiagnosisIssue(
                "warning", "discrimination",
                "候选变" + plural + " IV 不足",
                (f"IV≥0.02 仅 {strong} 个，"
                 + (f"IV 0.01-0.02 有 {weak} 个，" if weak else "")
                 + f"IV<0.01 有 {none_} 个。最高 IV={max(iv_vals.values()):.4f}"),
                ("预判最终 KS 可能偏低（< 0.25）。建议丰富变量池。"
                 if strong < 5 else "变量池有信息量，可以继续建模。"),
            ))

    if missing_rates:
        high_miss = {k: v for k, v in missing_rates.items() if v > 0.30}
        if high_miss:
            names = ", ".join(f"{k}({v:.0%})" for k, v in list(high_miss.items())[:5])
            issues.append(DiagnosisIssue(
                "warning", "variable",
                "多个变量缺失率偏高",
                f"{names} 等 {len(high_miss)} 个变量缺失 > 30%",
                "建议用中位数填充或删除高缺失变量",
            ))

    return issues


def _check_bad_rate(y_train) -> list[DiagnosisIssue]:
    y = np.asarray(y_train).ravel()
    br = float(y.mean())
    if br < 0.02:
        return [DiagnosisIssue(
            "info", "discrimination",
            "坏样本率偏低",
            f"目标坏样本率仅 {br:.1%}（< 2%）",
            "样本不平衡严重。建议考虑 SMOTE 或加权 LR",
        )]
    return []


# ── report formatting ────────────────────────────────────────────────────────


def _format_report(report: DiagnosisReport) -> str:
    lines = [
        "═══════════════════════════════════════════",
        "  ProScore 模型诊断报告",
        "═══════════════════════════════════════════",
    ]

    if not report.issues:
        lines.extend(["", "  ✅ 未发现异常 — 模型各项指标在可接受范围内。", ""])
        lines.append("  参考基准: KS≥0.20 | AUC≥0.65 | PSI≤0.10 | KS衰退<0.05 | IV≥0.02")
        lines.append("═══════════════════════════════════════════")
        return "\n".join(lines)

    for level, label, emoji in [
        ("critical", "严重 — 建议修复后再投产", "🔴"),
        ("warning", "警告 — 建议优化", "🟡"),
        ("info", "提示 — 仅供参考", "ℹ️ "),
    ]:
        items = [i for i in report.issues if i.level == level]
        if not items:
            continue
        lines.append(f"\n{emoji} {label} ({len(items)} 项)")
        lines.append("─" * 42)
        for item in items:
            lines.append(f"\n【{item.category}】{item.title}")
            lines.append(f"  {item.evidence}")
            if item.culprit_vars:
                lines.append(f"  涉及变量: {', '.join(item.culprit_vars[:5])}"
                             f"{'...' if len(item.culprit_vars) > 5 else ''}")
            lines.append(f"  → {item.suggestion}")

    lines.append("\n  参考基准: KS≥0.20 | AUC≥0.65 | PSI≤0.10 | KS衰退<0.05 | IV≥0.02")
    lines.append("═══════════════════════════════════════════")
    return "\n".join(lines)
