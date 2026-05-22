# ProScore · `inspect` 模块使用指南

面向建模与数据分析的**数据探查**：变量质量总览、区分力（IV / 单变量 AUC·KS）、高相关对与 VIF。本手册描述**用户侧用法**；字段级行为以源码与 OpenSpec 为准。

---

## 目录

- [模块职责与边界](#模块职责与边界)
- [环境与安装](#环境与安装)
- [导入](#导入)
- [快速开始](#快速开始)
- [API：`detect`](#apidetect)
- [API：`quality`](#apiquality)
- [API：`correlation` / `vif`](#apicorrelation--vif)
- [API：`list_supported_estimators`](#apilistsupportedestimators)
- [API：`stability`](#apistability)
- [推荐分析顺序](#推荐分析顺序)
- [数据与 API 约束（必读）](#数据与-api约束必读)
- [常见问题与报错](#常见问题与报错)
- [相关文档](#相关文档)

---

## 模块职责与边界

| 会做 | 不做 |
|------|------|
| 描述性统计、缺失/唯一值/特殊值、与目标的粗关联 | 分箱、WOE、建模、正式特征筛选规则 |
| IV、单变量 AUC/KS（粗筛）、可选 PSI | 保证 AUC/KS 为无偏 OOS 指标 |
| 数值列相关与 VIF | 自动修改你的 `DataFrame` 或列名 |

---

## 环境与安装

- **Python**：与项目 `pyproject.toml` 中 `requires-python` 一致（当前为 `>=3.9`）。
- **核心依赖**：`numpy`、`pandas`、`scikit-learn`、`statsmodels`（随主包安装）。
- **可选**：单变量 AUC/KS 若使用树以外的估计器，需安装 **XGBoost** / **LightGBM**（参见项目 `pyproject.toml` 中的可选组，如 `inspect-ml`）。

从源码目录安装示例：

```bash
pip install -e ".[dev]"
# 若需 xgb/lgb：
# pip install -e ".[inspect-ml]"
```

---

## 导入

```python
from proscore.inspect import (
    detect,
    quality,
    correlation,
    vif,
    list_supported_estimators,
)
```

---

## 快速开始

以下使用**合成小表**（可直接复制运行）：

```python
import numpy as np
import pandas as pd
from proscore.inspect import detect, quality

rng = np.random.default_rng(42)
n = 300
df = pd.DataFrame(
    {
        "age": rng.integers(22, 60, n),
        "income_k": rng.integers(10, 80, n),
        "channel": rng.choice(["web", "app", "branch"], n),
        "bad_flag": rng.integers(0, 2, n),
    }
)

# 1) 变量质量总览（含与 bad_flag 的粗关联）
overview = detect(df, target="bad_flag")
print(overview.head())

# 2) 区分力：IV / AUC / KS（数值列）
scores = quality(df, target="bad_flag")
print(scores.head())
```

---

## API：`detect`

对 `df` 中每一列（可选排除 `target`）生成一行质量信息。

### 参数摘要

| 参数 | 说明 |
|------|------|
| `df` | 输入表；**列名必须全局唯一**。 |
| `target` | 可选；若给出，增加 `target_pearson`（数值列）与 `target_cramers_v`（分类列），目标列本身不出现在结果中。 |
| `special_values` | 可选列表；为 `None`（默认）时不统计特殊值，`special_pct` / `special_pct_valid` 为 0 与 NaN。若传入列表，则按该列表匹配特殊值占比。 |
| `max_categories` | 分类判定阈值：唯一值较少或 object/category 类型会判为 `categorical`。 |

### 返回列（节选）

`variable`、`dtype`、`count`、`n_unique`、`missing`、`missing_pct`、`one_value_pct`、`special_pct`、`special_pct_valid`、数值描述列；若提供 `target`，另有 `target_pearson`、`target_cramers_v`。

### 示例：特殊值列表

```python
import pandas as pd
from proscore.inspect import detect

df = pd.DataFrame({"x": [1, 2, -999, 4], "y": [0, 1, 0, 1]})
out = detect(df, target="y", special_values=[-999])
assert out.loc[out["variable"] == "x", "special_pct"].iloc[0] > 0
```

---

## API：`quality`

按列输出 **IV**、数值列的**单变量 AUC / KS**（可选）、**缺失率**、**唯一值数**；若提供 `df_test`，增加与训练边际分布对比的 **PSI**。

### 重要说明（单变量 AUC / KS）

在**与拟合相同的样本**上打分再算 AUC/KS，属于**有偏、偏乐观**的粗筛指标，**不能**当作真实泛化能力。需要提醒协作者时，可设 `warn_insample_bias=True` 触发一次 `UserWarning`；默认不告警，请以文档与注释为准。

### 参数摘要

| 参数 | 说明 |
|------|------|
| `df` | 训练数据；**列名唯一**。 |
| `target` | 目标列名；需在 `df` 中。IV 实现上假定坏=1、好=0，用 `y.sum()` 计坏样本——代码**不会自动校验**编码，请调用方确保目标列符合此约定。 |
| `df_test` | 可选；用于 PSI。须包含 `df` 中除 `target` 与 `skip_columns` 外所有待分析列，且**列名全局唯一**（否则 `ValueError`）。 |
| `skip_columns` | 不参与计算的列（如主键、账期）。 |
| `max_categories` | 与 `detect` 一致的分类判定阈值。 |
| `estimator` | `"dt"`（默认）、`"xgb"`、`"lgb"` 等；**在函数入口即校验**，即使全是分类列也会报错。 |
| `compute_ks` | 是否计算单变量 KS。 |
| `errors` | `"raise"`：拟合/指标失败向上抛出；`"ignore"`：失败时 AUC/KS 为 `None`。 |
| `warn_insample_bias` | 默认 `False`；`True` 时对同样本 AUC/KS 发 `UserWarning`。 |

### 返回列

`variable`、`dtype`、`iv`、`auc`、`ks`、`psi`、`missing_pct`、`n_unique`。未传 `df_test` 时 `psi` 列为 `None`。

### 示例：PSI（训练 / 测试）

```python
import numpy as np
import pandas as pd
from proscore.inspect import quality

rng = np.random.default_rng(0)
n = 500
train = pd.DataFrame(
    {"score": rng.normal(0, 1, n), "bad_flag": rng.integers(0, 2, n)}
)
test = pd.DataFrame(
    {"score": rng.normal(0.5, 1, n), "bad_flag": rng.integers(0, 2, n)}
)
q = quality(train, target="bad_flag", df_test=test)
print(q[["variable", "iv", "psi"]])
```

---

## API：`correlation` / `vif`

### `correlation`

在数值列（或你指定的 `features`）上计算相关系数，返回绝对值 **≥ threshold** 的变量对。

| 参数 | 说明 |
|------|------|
| `df` | 列名唯一。 |
| `features` | 可选；默认所有数值列。列表内**不得重复**列名。 |
| `threshold` | 默认 `0.7`。 |
| `method` | `pearson` / `spearman` / `kendall`（交给 `pandas.DataFrame.corr`）。 |

返回列：`var1`、`var2`、`corr`。有效非空样本少于 2 的列会被排除；`NaN` 相关对会跳过。

### `vif`

对数值列（或指定 `features`）计算方差膨胀因子；常数列、零方差列会先剔除；奇异时该特征 VIF 记为 `inf`。

| 参数 | 说明 |
|------|------|
| `df` | 列名唯一。 |
| `features` | 可选；默认数值列；列表内不得重复。 |
| `threshold` | 超过则 `flag` 为 `"high"`（默认 `10.0`）。 |

返回列：`variable`、`vif`、`flag`。

### 示例

```python
import numpy as np
import pandas as pd
from proscore.inspect import correlation, vif

rng = np.random.default_rng(1)
n = 200
x = rng.normal(0, 1, n)
df = pd.DataFrame({"a": x, "b": x * 0.95 + rng.normal(0, 0.05, n)})

pairs = correlation(df, features=["a", "b"], threshold=0.5)
print(pairs)

mult = vif(df, features=["a", "b"], threshold=5.0)
print(mult)
```

仅独立数值列时的 `vif` 示例：

```python
import numpy as np
import pandas as pd
from proscore.inspect import vif

rng = np.random.default_rng(2)
n = 150
df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
print(vif(df))
```

---

## API：`list_supported_estimators`

返回**当前环境**下 `quality(..., estimator=...)` 可用的估计器名称列表（随是否安装 xgb/lgb 变化）。用于 Notebook 里展示或写配置前的自检。

```python
from proscore.inspect import list_supported_estimators

print(list_supported_estimators())
```

---

## API：`stability`

对带时间字段的 `DataFrame` 做**时序稳定性分析**，输出每个变量在各时期的样本量、bad_rate、分布 PSI（vs 首期 / vs 上一期）及稳定性标签。适合监控特征漂移与 bad_rate 趋势。

### 参数摘要

| 参数 | 说明 |
|------|------|
| `df` | 必须含 `target` 与 `time_col`。 |
| `target` | 二分类目标列（1=bad）；NaN 行会被排除。 |
| `time_col` | 时间周期列（如 `acctbegindate`、`month`）；函数会自动排序。 |
| `features` | 可选；默认所有非 target/time 列。 |
| `n_bins` | 连续变量 PSI 分箱数（默认 5）。 |
| `bad_rate_trend_threshold` | 末期 bad_rate 相对首期变化超过此值（默认 0.5）则标 `trending_up/down`。 |
| `psi_warn_threshold` | 任意时期 PSI（vs 首期）超过此值（默认 0.1）则标 `unstable`。 |

### 返回列

`variable`、`time_period`、`n`、`bad_rate`、`bad_rate_change`、`psi_vs_first`、`psi_vs_prev`、`mean`、`std`、**`psi_flag`**、**`bad_rate_flag`**。

两列**独立评价**，不要混在一列里：

| 列 | 取值 | 含义 |
|----|------|------|
| `psi_flag` | `baseline` / `stable` / `unstable` | 分布 PSI（vs 首期）是否超阈 |
| `bad_rate_flag` | `baseline` / `stable` / `trending_up` / `trending_down` | 坏账率相对首期变化是否超阈 |

### 示例

```python
from proscore.inspect import stability

res = stability(df, target="bad_flag", time_col="month")
# 仅看分布漂移
print(res[res["psi_flag"] == "unstable"])
# 仅看坏账率趋势
print(res[res["bad_rate_flag"].isin(["trending_up", "trending_down"])])
```

---

## 推荐分析顺序

1. **`detect`**：整体质量、缺失、特殊值、与目标的粗关联。  
2. **`quality`**：IV 排序与单变量区分力粗筛。`target` 列自动排除；主键、账期等不需分析的列放入 `skip_columns`。
3. 对入模候选数值子集：**`correlation`** → **`vif`**，控制共线性。
4. 之后先做 **`Filter`**（轻量）剔除不合格变量，再做 **`Binning`** 分箱——优先把计算资源花在有价值的变量上。

---

## 数据与 API 约束（必读）

1. **列名全局唯一**  
   `detect`、`quality`（及 `df_test`）、`correlation`、`vif` 均要求 `DataFrame` 列名不重复；否则 **`ValueError: duplicate column labels`**。请先在外部清洗列名。

2. **`correlation` / `vif` 的 `features` 列表**  
   同一列名不得出现两次，否则 **`ValueError: duplicate entries`**。

3. **`quality` 与 `df_test`**  
   除 `target` 与 `skip_columns` 外，`df` 中参与扫描的列必须在 `df_test` 中存在，否则 **`KeyError`**。

4. **`estimator`**  
   非法名称或未安装的可选库会在调用早期抛出 **`ValueError`** / **`ImportError`**。

5. **IV 与目标编码**  
   IV 使用 `target` 的聚合方式与「坏=1」的常见约定一致；若业务标签含义相反，需在业务层统一编码后再调用。

---

## 常见问题与报错

| 现象 | 可能原因 | 处理方向 |
|------|-----------|----------|
| `ValueError` … duplicate column labels (df) | `df` 中有重复列名 | 重命名列后再调用 |
| `ValueError` … duplicate column labels (df_test) | `quality` 的 `df_test` 有重复列名 | 重命名测试集列 |
| `ValueError` … duplicate entries | `features` 里写了两次同一列 | 去重列表 |
| `KeyError` … df_test is missing | 测试集缺训练里有的特征列 | 对齐列或调整 `skip_columns` |
| `KeyError` … target | `target` 不在 `df` 中 | 检查列名拼写 |
| 全分类 + 仍报 estimator 错 | 入口即校验估计器 | 使用 `list_supported_estimators()` 核对 |
| AUC/KS 为 `None` | 样本过少、常数特征、目标非两类等 | 检查数据；或 `errors="ignore"` 仅用于排障 |

---

## 相关文档

- OpenSpec（字段级规格）：[`docs/spec/04-inspect.md`](../spec/04-inspect.md)  
- 项目约定：仓库根目录 `AGENTS.md`  

若本手册与实现不一致，**以当前版本源码与 OpenSpec 为准**；欢迎在开 issue 时附上 `pandas` / `sklearn` 版本与最小复现表。
