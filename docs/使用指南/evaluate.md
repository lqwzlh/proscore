# ProScore · `evaluate` 模块使用指南

面向评分卡建模的**模型评估**：KS、AUC、PSI、准确率、评分分布表。

---

## 目录

- [导入](#导入)
- [快速开始](#快速开始)
- [API：`evaluate`](#apievaluate)
  - [参数](#参数)
  - [返回值](#返回值)
- [KS 衰退说明](#ks-衰退说明)
- [常见问题与报错](#常见问题与报错)
- [相关文档](#相关文档)

---

## 导入

```python
from proscore.evaluate import evaluate
```

---

## 快速开始

```python
from proscore.evaluate import evaluate

perf = evaluate(
    sc.model_,                    # 已拟合的模型
    df_trn_woe[features],         # 训练特征
    y_train,                      # 训练目标
    df_tst_woe[features],         # 测试特征
    y_test,                       # 测试目标
    n_bins=10,                    # 评分分布表的分箱数
)

print(f"Train KS: {perf['trn_ks']:.4f}")
print(f"Test KS:  {perf['test_ks']:.4f}")
print(f"AUC:      {perf['test_auc']:.4f}")
print(f"PSI:      {perf['psi']:.4f}")
print(perf["score_table"])        # 评分分布表
```

---

## API：`evaluate`

```python
evaluate(model, X_train, y_train, X_test, y_test, features=None, n_bins=10, threshold=0.5)
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | statsmodels Logit 或 sklearn classifier | — | 需有 `predict` 或 `predict_proba` |
| `X_train` / `y_train` | `pd.DataFrame` / `pd.Series` | — | 训练集 |
| `X_test` / `y_test` | `pd.DataFrame` / `pd.Series` | — | 测试集 |
| `features` | `list[str]` 或 `None` | `None` | 特征列（默认全部） |
| `n_bins` | `int` | `10` | 评分分布等频分箱数 |
| `threshold` | `float` | `0.5` | 分类阈值（用于准确率） |

> 前提条件：目标是**二分类**，且**训练集和测试集均需包含两个类别**，否则 `roc_auc_score` 会报错。

### 返回值

`dict[str, Any]`，包含以下键：

| 键 | 类型 | 说明 |
|----|------|------|
| `trn_ks` | `float` | 训练集 KS |
| `test_ks` | `float` | 测试集 KS |
| `trn_auc` | `float` | 训练集 AUC |
| `test_auc` | `float` | 测试集 AUC |
| `trn_acc` | `float` | 训练集准确率 |
| `test_acc` | `float` | 测试集准确率 |
| `ks_reduce` | `float` | KS 衰退（带符号）：`trn_ks - test_ks` |
| `ks_rel_gap` | `float` | KS 相对衰退：`abs(ks_reduce) / trn_ks` |
| `psi` | `float` | 模型分数 PSI |
| `score_table` | `pd.DataFrame` | 评分分布表 |
| `model_vars` | `list[str]` | 入模特征 |

`score_table` 列：`bin | count | good | bad | bad_rate | pct | cum_good_pct | cum_bad_pct | ks`。

---

## KS 衰退说明

- **`ks_reduce`**（带符号）→ `trn_ks - test_ks`。正值表示过拟合（train 高于 test），负值表示 test 反而更高。
- **`ks_rel_gap`**（相对）→ `abs(ks_reduce) / trn_ks`。用于跨模型对比衰退程度。

这两个字段与 `StepwiseSelector` 的 perf 字典命名和公式一致。

---

## API：`evaluate_by_period`（分年 OOT）

当 OOT 覆盖多个日历年（如 2022、2023）时，用本函数输出**逐年** KS/AUC/PSI，便于观察时间外衰退趋势。

```python
from proscore.evaluate import evaluate_by_period

oot_periods = {
    "2022": (oot_woe_22[features], y_22),
    "2023": (oot_woe_23[features], y_23),
}
oot_period_eval = evaluate_by_period(
    model, train_woe[features], y_train,
    periods=oot_periods,
    features=features,
)
```

也可传入带 `apply_date` 的 DataFrame，按 `time_col.dt.year` 自动分组。

返回 `DataFrame`：`period | n | bad_rate | ks | auc | acc | psi_score | ks_decay | auc_decay`（相对训练集）。

合并 OOT 指标仍用 `evaluate(..., X_oot=..., y_oot=...)` 的 `oot_ks` / `oot_auc`。

---

## 常见问题与报错

| 现象 | 可能原因 | 处理方向 |
|------|----------|----------|
| `ValueError` from `roc_auc_score` | 某侧数据只有一个类别 | 确保 train/test 都包含正负样本 |
| `ValueError: Bin edges must be unique` | 模型预测概率过于集中 | 已内置 `np.unique` 处理，不应出现 |
| 模型 predict 报 shape 错误 | statsmodels 需要 constant 列 | `evaluate` 内部自动加 constant |

---

## 相关文档

- 手册总览：[`index.md`](index.md)
- 建模模块：[`modeling.md`](modeling.md)
- 筛选模块：[`selection.md`](selection.md)
