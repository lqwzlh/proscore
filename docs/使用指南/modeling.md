# ProScore · `modeling` 模块使用指南

面向评分卡建模的**逻辑回归 + 评分卡转换**：在 WOE 数据上拟合 LR，将系数转为标准评分卡格式。

---

## 目录

- [导入](#导入)
- [快速开始](#快速开始)
- [API：`ScoreCard`](#apiscorecard)
  - [构造参数](#构造参数)
  - [方法](#方法)
  - [属性](#属性)
- [评分卡公式](#评分卡公式)
- [常见问题与报错](#常见问题与报错)
- [相关文档](#相关文档)

---

## 导入

```python
from proscore.modeling import ScoreCard
```

---

## 快速开始

```python
from proscore.modeling import ScoreCard

sc = ScoreCard(odds=50, pdo=10, base_score=600)  # odds=50:1, PDO=10, 基准分600

# y 可以是列名（在 X 中）或 Series
sc.fit(df_woe, y="target")                  # 自动排除 target 列
# 或指定入模特征
sc.fit(df_woe, y="target", features=["x1", "x2"])

# 生成评分卡
score_table = sc.scorecard(b.bin_table_)    # 需要 binning 结果
print(score_table[["variable", "bin_label", "points"]])

# 对新数据评分
scores = sc.predict(df_woe_new)
```

---

## API：`ScoreCard`

```python
ScoreCard(odds=50, pdo=10, base_score=600)
```

### 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `odds` | `float` | `50` | 基准分对应的好坏比（good:bad = odds:1） |
| `pdo` | `float` | `10` | 好坏比翻倍所需分数（Points to Double the Odds） |
| `base_score` | `float` | `600` | 基准分 |

### 方法

| 方法 | 说明 |
|------|------|
| `fit(X, y, features=None)` | 拟合 statsmodels Logit。`y` 可以是列名或 Series。`features` 默认全部列（排除 target 列） |
| `scorecard(bin_tables)` | 生成评分卡 DataFrame。`bin_tables` 为 `Binning.bin_table_` |
| `predict(X)` | 对新 WOE 数据评分（分数越高 = 风险越低） |

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `model_` | statsmodels Logit | 拟合后的 LR 模型 |
| `intercept_` | `float` | LR 截距 |
| `coef_` | `dict[str, float]` | 每特征的 LR 系数 |
| `score_table_` | `pd.DataFrame` | 评分卡表（调用 `scorecard()` 后可用） |

`score_table_` 的列：`variable | bin_label | bin_no | count | bad_rate | woe | coef | points | is_extra`。其中 `is_extra=True` 的行对应特殊值箱和缺失箱。

---

## 评分卡公式

```
Score = base_offset + factor × Σ(coef_i × WOE_i)

where:
  factor = PDO / ln(2)
  base_offset = base_score + factor × (ln(odds) + intercept)
```

每箱分数 = `factor × coef × WOE`。用户总分 = 所有箱分数之和 + base_offset。

> 若数据接近完美分离，statsmodels 可能抛出 `LinAlgError` 或 `PerfectSeparationWarning`——这些不会被吞掉，请检查特征是否存在准完全共线性。

---

## 常见问题与报错

| 现象 | 可能原因 | 处理方向 |
|------|----------|----------|
| `RuntimeError: Call fit() before...` | 未调用 `fit()` | 先 fit |
| `KeyError` 在 scorecard | `bin_tables` 中缺少某特征 | 确认 binning 覆盖所有入模特征 |
| `ValueError: No features to fit` | `X` 中除 target 外没有其他列 | 检查数据 |
| `LinAlgError` | 完美分离或奇异矩阵 | 检查特征共线性或 remove 高度预测的特征 |
| 分数为负数 | WOE 极端 + 高 PDO | 调整 `base_score` 或 `pdo` |

---

## 相关文档

- 手册总览：[`index.md`](index.md)
- WOE 转换：[`transform.md`](transform.md)
- 分箱模块：[`binning.md`](binning.md)
