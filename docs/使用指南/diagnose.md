# ProScore · 模型诊断

建模后自动诊断模型健康状态，输出结构化问题 + 根因 + 操作建议。

---

## 目录

- [快速开始](#快速开始)
- [诊断规则详解](#诊断规则详解)
- [链式 API](#链式-api)
- [独立使用](#独立使用)
- [输出结构](#输出结构)
- [常见问题](#常见问题)

---

## 快速开始

```python
import proscore as ps

p = (
    ps.ProScore()
    .read(train=df_train, test=df_test, target="bad_flag")
    .prefilter().bin().refine().transform().select().fit().scorecard()
    .evaluate()
    .diagnose()  # ← 自动打印诊断报告
)

# 结构化访问
print(p.diagnosis_.critical)  # 严重问题列表
print(p.diagnosis_.to_dataframe())  # 导出为表格
```

---

## 诊断规则详解

全部规则按 4 层组织，基准参考行业指标：
- Siddiqi, *Credit Risk Scorecards* (2006)（评分卡开发与验证行业标准）
- SR 11-7, *Model Risk Management Guidance*（美联储模型风险管理指引）
- 银保监《商业银行资本计量高级方法验证指引》

### 第一层：区分力

| # | 触发条件 | 级别 | 说明 |
|---|---------|------|------|
| 1 | test_KS < 0.15 | **critical** | KS 不可用，模型区分力无法通过监管 |
| 2 | 0.15 ≤ test_KS < 0.20 | **warning** | KS 偏低，勉强可用 |
| 3 | 0.20 ≤ test_KS < 0.30 | **info** | KS 可接受但优化空间大 |
| 4 | test_KS > 0.60 | **info** | KS 异常高（偏离常见消费信贷范围），警惕信息泄漏 |
| 5 | test_auc < 0.60 | **critical** | AUC 接近随机 |
| 6 | 0.60 ≤ test_auc < 0.70 | **warning** | AUC 区分力偏弱 |
| 7 | test_auc > 0.95 | **info** | AUC 异常高（>0.95），极大概率存在目标泄漏或未来信息 |

### 第二层：过拟合（相对比例衰减）

| # | 触发条件 | 级别 | 说明 |
|---|---------|------|------|
| 8 | (trn_KS - test_KS) / trn_KS > 30% | **critical** | 严重过拟合 |
| 9 | (trn_KS - test_KS) / trn_KS > 15% | **warning** | 中度过拟合 |
| 10 | (trn_KS - test_KS) / trn_KS > 10% | **info** | 轻微衰减，可能正常 |

### 第三层：稳定性

| # | 触发条件 | 级别 | 说明 |
|---|---------|------|------|
| 11 | psi > 0.25 | **critical** | 评分分布显著漂移，可能需重训 |
| 12 | 0.10 < psi ≤ 0.25 | **warning** | 评分分布轻微漂移 |
| 13 | OOT KS 相对 test 衰减 > 30% | **critical** | OOT 区分力急剧下降 |
| 14 | OOT KS 相对 test 衰减 15%-30% | **warning** | OOT 区分力下降 |
| 15 | (trn_auc - test_auc) / (trn_auc - 0.5) > 20% | **warning** | AUC 衰退（train→test 区分力占比损失） |

### 第四层：变量质量

| # | 触发条件 | 级别 | 说明 |
|---|---------|------|------|
| 16 | 系数符号与 coef_sign 矛盾 | **critical** | WOE 方向可能有误 |
| 17 | 分箱趋势与预设不符（trend_match=False） | **warning** | 检查分箱图 |
| 18 | 入模变量缺失率 > 50% | **critical** | 严重影响稳定性 |
| 19 | 入模变量缺失率 > 30% | **warning** | 建议填充 |
| 20 | 某变量 IV > 0.50 | **warning** | 疑似信息泄漏 |

### 参考级提示（不触发告警，仅提示）

| 条件 | 级别 |
|------|------|
| 入模变量 IV < 0.02 且 >0 | info |
| 入模变量 > 12 个 | info |
| 坏样本率 < 2% | info |

> **注意**：编号仅为文档阅读方便，实际触发顺序以 `_check_*` 函数执行顺序为准。

### 阈值说明（现已支持自定义）

所有默认阈值定义在 `src/proscore/evaluate/_diagnose.py` 的 `DEFAULT_THRESHOLDS` 中（按类别组织）：

```python
from proscore.evaluate import DEFAULT_THRESHOLDS, diagnose

print(DEFAULT_THRESHOLDS["discrimination"]["ks_critical"])  # 0.15
```

**使用方式**（v0.2+ 已支持）：

```python
# 仅覆盖部分阈值，其余保持默认
report = diagnose(
    eval_result,
    binning=p.binner_,
    thresholds={
        "discrimination": {"ks_critical": 0.18, "auc_warning": 0.72},
        "stability": {"psi_warning": 0.08},
    },
)
```

- 未指定的类别或键会自动使用默认值（深层合并）。
- 这对不同机构、不同监管辖区、或不同产品线（信用卡 vs 消费贷）的偏好适配非常有用。
- `ProScore` 链式 API 同样支持：`p.diagnose(thresholds=..., print_report=False)`

文档中的数值以 `DEFAULT_THRESHOLDS` 为最终准绳（随代码实时同步）。

### 根因定位

诊断不只报问题，还尽量定位根因：

| 症状 | 自动定位方式 |
|------|------------|
| PSI 漂移 | 从 `stability()` 输出中找出 PSI 最高的变量 |
| OOT 衰减 | 从 `stability()` 找出标记为 unstable/trending_down 的变量 |
| IV 偏高 | 列出 IV > 0.5 的入模变量 |
| 缺失偏高 | 列出缺失率 > 30% 的入模变量 |

---

## 链式 API

```python
# 基础：打印格式化报告（默认行为，适合 notebook）
p.evaluate().diagnose()

# 静默模式：仅生成 DiagnosisReport，不打印（推荐在脚本/生产流程中使用）
p.evaluate().diagnose(print_report=False)
report = p.diagnosis_
print(len(report.critical), "个严重问题")

# 完整：传 binning/selector/stability 做根因定位
p.evaluate().diagnose(
    binning=p.binner_,
    selector=p.selector_,
    stability=stability_result,    # inspect.stability() 输出
    period_eval=period_result,     # evaluate_by_period() 输出
    print_report=True,             # 显式要求打印
)
```

诊断结果始终可通过 `p.diagnosis_` 属性访问结构化数据（`critical` / `warnings` / `infos` / `to_dataframe()`）。

---

## 独立使用

```python
from proscore.evaluate import evaluate, diagnose

metrics = evaluate(model, X_trn, y_trn, X_test=X_tst, y_test=y_tst)
report = diagnose(metrics, binning=binner, selector=selector)

# 结构化访问
print(report)                    # 打印格式化报告
print(report.critical)           # 严重问题列表
print(report.warnings)           # 警告列表
df = report.to_dataframe()       # 导出为 DataFrame
```

### 建模前诊断

```python
from proscore.evaluate import diagnose

# 在分箱后、建模前评估候选变量池质量
report = diagnose(binning=binner, train_columns=feature_list)
```

检查候选变量 IV 分布和缺失率，预判建模后 KS 上限。

---

## 输出结构

```python
@dataclass
class DiagnosisIssue:
    level: str          # "critical" | "warning" | "info"
    category: str       # "discrimination" | "overfitting" | "stability" | "variable"
    title: str          # 一句话概括
    evidence: str       # 数据证据（含实际值 vs 阈值）
    suggestion: str     # 操作建议
    culprit_vars: list  # 涉事变量列表

@dataclass
class DiagnosisReport:
    issues: list[DiagnosisIssue]
    
    @property
    def critical(self) -> list[DiagnosisIssue]: ...
    @property
    def warnings(self) -> list[DiagnosisIssue]: ...
    @property
    def infos(self) -> list[DiagnosisIssue]: ...
    def to_dataframe(self) -> pd.DataFrame: ...
```

---

## 常见问题

**Q: 和 evaluate() 的区别？**
evaluate() 输出原始指标（KS/AUC/PSI）。diagnose() 拿这些指标对照行业基准给出可操作建议。

**Q: 传了 binning 和没传有什么区别？**
没传 binning：只告诉你 KS 不达标。传了 binning：告诉你 KS 不达标是因为哪些变量 IV 太低。

**Q: 参考基准可以自定义吗？**
可以（v0.2+ 已支持）。通过 `diagnose(thresholds=...)` 或 `p.diagnose(thresholds=...)` 按类别部分覆盖，详见上文「阈值说明」小节。未指定的键保持默认值。

**Q: 诊断结果会写进报告吗？**
会。
- 手动：`ReportBuilder().with_evaluate(ev).with_diagnosis(report).build()`
- 自动：`ReportBuilder.from_proscore(p)` 在 `p` 已调用过 `diagnose()` 或存在 `eval_result` 时会自动附带诊断章节（见 C3 增强）。

---

## 相关文档

- 模型评估：[`evaluate.md`](evaluate.md)
- 分箱：[`binning.md`](binning.md)
- 逐步回归：[`selection.md`](selection.md)
