# ProScore · `selection` 模块使用指南

面向评分卡建模的**特征筛选**：Filter 初筛（缺失率/IV/AUC/PSI/相关性/VIF）+ StepwiseSelector 双向迭代筛选（LR 全约束 + 来源控制 + 扰动）。

---

## 目录

- [导入](#导入)
- [API：`Filter`](#apifilter)
  - [构造参数](#filter-构造参数)
  - [方法](#filter-方法)
  - [属性](#filter-属性)
- [API：`StepwiseSelector`](#apistepwiseselector)
  - [构造参数](#stepwise-构造参数)
  - [方法](#stepwise-方法)
  - [属性](#stepwise-属性)
  - [目标函数](#目标函数)
- [推荐工作流](#推荐工作流)
- [常见问题与报错](#常见问题与报错)
- [相关文档](#相关文档)

---

## 导入

```python
from proscore.selection import Filter, StepwiseSelector
```

---

## API：`Filter`

初筛器：按顺序执行缺失率 → 单值率 → IV → AUC → PSI → 相关系数 → VIF，最后按 IV 截断 top-N。

**推荐两阶段**（与链式 `prefilter` / `refine` 一致）：

1. **粗筛**（`iv_range=None`, `max_psi=None`）：缺失、单值、AUC、相关、VIF  
2. **分箱**（`BinningProcess` / `Binning`）  
3. **精筛**（传入 `bin_table`）：IV、PSI（Train vs Test）

### Filter 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_missing_rate` | `float` | `0.8` | 缺失率上限 |
| `max_one_value_rate` | `float` | `0.95` | 单值率上限 |
| `iv_range` | `tuple` | `(0.02, None)` | IV 下界；`None` 表示不设上界 |
| `min_auc` | `float` 或 `None` | `None` | 单变量 AUC 下限（in-sample，偏乐观） |
| `max_psi` | `float` 或 `None` | `None` | PSI 上限 |
| `max_corr` | `float` | `0.7` | 相关系数上限（超阈值保留 IV 高者） |
| `max_vif` | `float` 或 `None` | `None` | VIF 上限 |
| `n_selected` | `int` 或 `None` | `None` | 最终保留 top-N（按 IV 排名） |

> IV/PSI 优先使用 `bin_table`；无 `bin_table` 时 IV 回退为与 `quality()` 相同的等频/类别粗算。`quality_` 的 `reason` 会列出**所有**未通过的检查项（分号分隔）。

### Filter 方法

| 方法 | 说明 |
|------|------|
| `fit(X, y, X_test=None, bin_table=None)` | 执行筛选。`bin_table` 为 `Binning.bin_table_`，提供后 IV 和 PSI 使用真实分箱值 |

### Filter 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `support_` | `list[str]` | 筛选后保留的特征 |
| `iv_` | `pd.DataFrame` | IV 详情（`feature` / `iv` / `source`） |
| `quality_` | `pd.DataFrame` | 变量质量总览（含 `selected` / `dropped` / `reason`） |

---

## API：`StepwiseSelector`

双向迭代选择器：前向添加 + 后向剔除，LR 条件检查全参数化（`None` 即关闭）。

### Stepwise 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pvalue_threshold` | `float` 或 `None` | `0.05` | P 值阈值，`None` 关闭 |
| `coef_sign` | `str` 或 `None` | `"positive"` | 系数符号 `"positive"` / `"negative"` / `None` |
| `vif_threshold` | `float` 或 `None` | `10` | VIF 阈值，`None` 关闭 |
| `corr_threshold` | `float` 或 `None` | `0.8` | 相关系数阈值，`None` 关闭 |
| `feature_belong` | `dict` 或 `None` | `None` | 来源归属 `{来源: [特征列表]}` |
| `belong_max_pct` | `float` 或 `None` | `None` | 单来源最大占比 |
| `perturbation` | `bool` | `True` | 是否启用扰动（随机删+加跳出局部最优） |
| `perturbation_pct` | `float` | `0.1` | 扰动删除比例 |
| `perturbation_add` | `int` | `2` | 每次扰动最多随机补回的候选特征数（`0` 表示不补回） |
| `max_iter_round` | `int` | `100` | 最大迭代轮次 |
| `max_iter_time` | `int` | `600` | 最大秒数 |
| `same_round_exit` | `int` | `4` | 连续相同变量集退出轮次 |
| `r` | `float` | `0.8` | 过拟合惩罚权重 |
| `objective` | `str` 或 `Callable` | `"ks_reduce"` | 目标函数（见下方） |
| `goal_threshold` | `float` | `0.01` | 最小得分提升 |
| `n_min` / `n_max` | `int` | `5` / `15` | 变量数约束 |
| `force_fill` | `bool` | `True` | 不足 `n_min` 时强制补入 |

### Stepwise 方法

| 方法 | 说明 |
|------|------|
| `fit(X, y, candidates=None, force_in=None, sort_df=None, X_test=None, y_test=None)` | 执行迭代。`X_test`/`y_test` 提供后计算真实 KS/AUC 衰退 |

### Stepwise 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `support_` | `list[str]` | 最终选中特征 |
| `record_` | `dict[int, dict]` | 每轮迭代的详细记录（统一 schema） |
| `best_performance_` | `dict` | 最佳解的 perf 字典 |
| `model_` | statsmodels Logit | 最佳 LR 模型 |

### 目标函数

| `objective` | 公式 | 说明 |
|-------------|------|------|
| `"ks"` / `"ks_reduce"` | `test_ks + (1-r)×trn_ks - r×ks_reduce` | 最大化 KS，惩罚过拟合 |
| `"auc"` / `"auc_reduce"` | `test_auc + (1-r)×trn_auc - r×auc_reduce` | 最大化 AUC，惩罚过拟合 |
| 自定义 | `callable(perf_dict) -> float` | 用户自定义 |

`goal_threshold` 为绝对提升阈值：KS 目标建议 0.001–0.02，AUC 目标建议 0.0005–0.01。

---

## 从 Excel 导入来源归属

`feature_belong` 通常与 `BinningProcess` 的 `feature_config` 共享同一份 Excel 预设文件。使用 `proscore.utils.load_presets` 一次性加载两者：

```python
from proscore.utils import load_presets
from proscore.binning import BinningProcess
from proscore.selection import StepwiseSelector

presets = load_presets("variable_presets.xlsx")

# presets.feature_config  → 给 BinningProcess
# presets.feature_belong  → 给 StepwiseSelector

bp = BinningProcess(feature_config=presets.feature_config)
ss = StepwiseSelector(
    feature_belong=presets.feature_belong,
    belong_max_pct=0.5,    # 单维度变量不超过入模总数的 50%
)
```

> Excel 中 `dimension` 列为空的行不会参与同源竞争。示例文件见 `tests/variable_presets.xlsx`。

---

## 推荐工作流

1. **`prefilter`**（粗筛，不传 `bin_table`） → 剔除缺失率高/单值率高/高相关的变量。这些检查不需要分箱结果，先筛掉垃圾变量
2. **`Binning` + `WOETransformer`** → 对剩余变量分箱 + WOE 转换
3. **`refine`**（精筛，传入 `bin_table`） → 用分箱后的真实 IV/PSI/AUC/VIF 做第二轮筛选
4. **`StepwiseSelector`** → 在 WOE 数据上双向迭代，选出最终入模变量

---

## 常见问题与报错

| 现象 | 可能原因 | 处理方向 |
|------|----------|----------|
| `KeyError: feature_belong contains features not in X` | 来源字典中的特征名不在 `X.columns` | 检查拼写 |
| 迭代未找到可行解 | `n_min` 过高或约束过严 | 放宽 `coef_sign=None` 或降低 `n_min` |
| 目标函数不变 | `goal_threshold` 太大 | 降低到 0.001 |
| `ValueError: y_test must be provided` | 传了 `X_test` 但未传 `y_test` | 同传或不传 |

---

## 相关文档

- 手册总览：[`index.md`](index.md)
- OpenSpec：[`docs/spec/03-selection.md`](../spec/03-selection.md)
