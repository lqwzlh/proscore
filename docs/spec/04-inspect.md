# OpenSpec: Inspect 数据探查模块

## 标识
- **模块名**: proscore.inspect
- **优先级**: P0
- **依赖**: numpy, pandas, scikit-learn
- **可选依赖**: xgboost、lightgbm（用于区分力指标计算）；在 `pyproject.toml` 中声明为 `inspect-ml` 可选组：`pip install proscore[inspect-ml]`
- **参考**: toad.detect (设计思路，非代码)

## 功能规格

### F1：变量质量报告 `detect()`
输入 ``DataFrame`` 的**列名必须全局唯一**（否则 ``ValueError``），与 ``correlation`` / ``vif`` / ``quality`` 一致。

一行代码输出 DataFrame 所有特征的**质量总览表**，包含：
- **数据类型**：连续/离散（自动推断）
- **样本量**：总数 / 非空数
- **缺失率**：缺失占比(百分比)
- **唯一值数**：unique values count
- **单值率**：占比最高的值占总数的比例(百分比)
- **特殊值率**：匹配预设特殊值列表的比例(百分比)
- **均值 / 标准差 / 分位数**：连续变量统计

当提供 `target` 时，额外输出（目标列本身不出现在表中）：
- **`target_pearson`**：数值特征与目标的 Pearson 相关（成对非空样本）
- **`target_cramers_v`**：分类特征与目标的 Cramér's V（0~1，与 Pearson 量纲不同）

### F2：区分力报告 `quality()`
训练集与可选的 ``df_test`` 的**列名均须全局唯一**（否则 ``ValueError``）。

输出每个变量的**建模区分力指标**：
- **IV**：信息量（自动分箱后计算）
- **单变量 AUC**：用可配置的 estimator 在**同一样本**上拟合并打分后计算（粗筛用，非无偏 OOS 估计）
- **单变量 KS**：同上（默认开启）
- **PSI**：当传入 `df_test` 时，按与 IV 一致的分箱规则在训练集上定箱，比较该特征在训练集与测试集上的边际分布稳定性；未传 `df_test` 时该列为 `None`
- **缺失率**：缺失占比
- **唯一值数**：unique values count

参数 **`errors`**：`"raise"`（默认）时 AUC/KS 拟合或指标计算失败会向上抛出；`"ignore"` 时失败记为 `None`。

函数入口即校验 **`estimator`**（即使全部为分类特征、不会触发单变量拟合也会报错）。

参数 **`warn_insample_bias`**：默认 ``False``；为 ``True`` 时对同样本 AUC/KS 发出一次 ``UserWarning``（文档仍说明有偏性）。

#### 支持的 estimator

| 名称 | 描述 | 依赖 |
| :--- | :--- | :--- |
| `"dt"` (默认) | 决策树（max_depth=3, min_samples_leaf=5） | sklearn（内置） |
| `"xgb"` | XGBoost（n_estimators=50, max_depth=3） | xgboost（可选） |
| `"lgb"` | LightGBM（n_estimators=50, max_depth=3） | lightgbm（可选） |

### F3：相关性报告 `correlation()`
- **相关系数矩阵**：pearson / spearman / kendall，标注 > threshold 的高相关对
- 自动忽略有效样本数 < 2 的列；跳过相关系数为 NaN 的配对
- **约束**：`DataFrame` 列名必须全局唯一；`features` 参数中同一列名不得重复出现，否则 `ValueError`

### F4：VIF 报告 `vif()`
- **VIF**：方差膨胀系数，自动剔除常数列和零方差列
- 仅在矩阵奇异等数值错误时将该项记为 `inf`，不再吞掉任意异常
- **约束**：同上，列名全局唯一且 `features` 不得含重复项，否则 `ValueError`

### F5：时序稳定性分析 `stability()`
输入 ``DataFrame`` 必须包含目标列与时间字段（`time_col`），时间字段须可排序。

对每个变量按时间周期输出：
- 样本量、bad_rate
- bad_rate 相对首期的变化率（`bad_rate_change`）
- PSI（vs 首期、vs 上一期）
- 数值型变量的均值/标准差
- 稳定性标签（**分列**，互不覆盖）：
  - `psi_flag`：`"baseline"` / `"stable"` / `"unstable"`（PSI vs 首期超阈）
  - `bad_rate_flag`：`"baseline"` / `"stable"` / `"trending_up"` / `"trending_down"`（坏账率相对首期变化超阈）

分类变量自动收集全时期取值，避免新类别丢失。PSI 计算复用项目统一实现 `psi_from_distributions`。

## API 签名

```python
def detect(df: pd.DataFrame,
           target: str | None = None,
           special_values: list | None = None,
           max_categories: int = 20) -> pd.DataFrame:
    """返回变量质量总览表"""
    ...

def quality(df: pd.DataFrame,
            target: str,
            df_test: pd.DataFrame | None = None,
            skip_columns: list[str] | None = None,
            max_categories: int = 20,
            estimator: str = "dt",
            compute_ks: bool = True,
            errors: Literal["raise", "ignore"] = "raise",
            warn_insample_bias: bool = False) -> pd.DataFrame:
    """返回变量区分力指标（IV / AUC / KS / PSI）"""
    ...

def correlation(df: pd.DataFrame,
                features: list[str] | None = None,
                threshold: float = 0.7,
                method: str = "pearson") -> pd.DataFrame:
    """返回高相关变量对"""
    ...

def vif(df: pd.DataFrame,
        features: list[str] | None = None,
        threshold: float = 10.0) -> pd.DataFrame:
    """返回 VIF"""
    ...

def list_supported_estimators() -> list[str]:
    """返回当前环境可用的 estimator 列表"""
    ...

def stability(
    df: pd.DataFrame,
    target: str,
    time_col: str,
    features: list[str] | None = None,
    n_bins: int = 5,
    bad_rate_trend_threshold: float = 0.5,
    psi_warn_threshold: float = 0.1,
) -> pd.DataFrame:
    """时序稳定性分析（bad_rate 趋势 + 分布 PSI）"""
    ...
```
