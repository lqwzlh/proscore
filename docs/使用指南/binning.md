# ProScore · `binning` 模块使用指南

面向评分卡建模的**特征分箱**：连续变量的自动分箱、分类变量的合并策略、趋势调整与质量检查。本手册描述业务原则与用户侧用法；字段级行为以源码与 OpenSpec 为准。

---

## 目录

- [分箱业务原则与趋势调整](#分箱业务原则与趋势调整)
  - [分箱质量准则](#一分箱质量准则)
  - [趋势调整算法](#二趋势调整算法)
- [模块职责与边界](#模块职责与边界)
- [环境与安装](#环境与安装)
- [导入](#导入)
- [快速开始](#快速开始)
- [API：`Binning`](#apibinning)
  - [构造参数](#构造参数)
  - [分箱算法](#分箱算法)
  - [分类变量策略](#分类变量策略)
  - [趋势约束](#趋势约束)
  - [特殊值与缺失值](#特殊值与缺失值)
  - [专家指定切点](#专家指定切点)
  - [属性](#属性)
- [API：`BinningProcess`](#apibinningprocess)
- [数据结构：`BinTable`](#数据结构bintable)
- [API：`apply_bin_index`](#apiapply_bin_index)
- [推荐工作流](#推荐工作流)
- [数据与 API 约束（必读）](#数据与-api-约束必读)
- [常见问题与报错](#常见问题与报错)
- [相关文档](#相关文档)

---

## 分箱业务原则与趋势调整

### 一、分箱质量准则

**分组数量与样本分布**

- 连续变量最终分箱数通常控制在 3–8 组。过多分组难以解释，过少则损失区分力。
- 单箱样本占比不应低于 5%（或坏样本数不少于 50），也不宜超过 25%。极端集中的分箱通常意味着切点选择不当或变量本身信息量薄弱。
- 例外：业务上明确需要单独隔离的特殊值（如 -999、-1）不受此限制。

**WOE 合理范围**

- 相邻分箱 WOE 差值小于 0.1（等价于 GBI 指数小于 1.1 倍），说明两箱区分力接近，建议合并以减少冗余。
- 用于切分客群的变量，切点两侧 WOE 宜一正一负——切点应落在"好坏反转"的位置，分段才有业务含义。
- WOE 绝对值通常落在 -2 到 +2 之间。若超过此范围，表明该区间好坏比极端偏离整体，该变量可能更适合作为策略规则单独使用，而非放入模型。

**特殊值与缺失值处理**

- 特殊值（如空值、行业惯用的缺失标记）和缺失值各自独立成箱，不参与分箱算法的迭代合并。
- 若特殊值箱或缺失箱占比极小，可将其合并到坏账率最接近的普通箱中，避免产生无统计意义的碎片箱。

**数据变化趋势**

- 分箱后坏账率随变量取值变化的总体方向应与业务认知一致。例如收入越高坏账率越低、历史逾期次数越多坏账率越高。
- 趋势形态优先选择单调（递增或递减），单调且接近线性最优。
- 少数变量在业务逻辑上确实存在"中间低、两端高"（U 型）或"中间高、两端低"（倒 U 型），可在确认业务合理性后保留该形态。

### 二、趋势调整算法

**原理**

逻辑回归要求特征与目标的对数几率呈线性关系，WOE 转换后变量若保持单调，能更好地满足这一前提。本模块的趋势调整在分箱后执行，通过迭代合并相邻箱来逼近目标趋势形态。

**预设趋势（专家模式）**

若用户通过 `monotonic` 参数明确指定了变量的趋势方向（递增、递减、U 型、倒 U 型），分箱调整将严格以此为目标。调整过程中反复合并破坏该趋势的邻箱对，直到满足约束或仅剩两箱。最终结果若仍与预设不符，该变量将被标记为不合规（`trend_match=False`），建模阶段可据此排除。

**自动识别趋势（无预设模式）**

若用户未指定趋势，系统按以下步骤自动判断：

1. **UV 型检测**。分箱后检查是否满足以下全部条件，满足则标记为 U 型或倒 U 型：
   - 坏账率的最低点（或最高点）落在分箱序列的中间区域（排除首尾各约 20% 的箱数）；
   - 核心点单侧样本量占比不低于阈值（如 25%）；
   - 两侧端点坏账率的差值不超过全振幅的 50%，即"两端差不多高"；
   - 调整后分箱数不少于 5 箱。

2. **单调型确认**。若未被确认为 UV 型，默认尝试调整为严格单调。对于调整后仍保持 3 箱及以上且满足单调的变量，标记为单调型。单调型变量在后续特征筛选中优先使用。

3. **未知型处理**。剩余无法在 3 箱及以上达成严格单调的变量，标记为未知型，降低其入模优先级——这些变量通常信息量较弱或噪声较大。

---

## 模块职责与边界

| 会做 | 不做 |
|------|------|
| 连续变量自动分箱（chi / tree / frequency / distance） | optimal (optbinning) 分箱（暂未实现） |
| 分类变量合并（WOE 按值 / 坏账率合并 / 高频合并） | 自定义分类映射（`custom` 模式暂未实现） |
| 趋势调整（单调 / U 型 / 倒 U 型）与质量检查 | 修改用户指定的专家切点 |
| 特殊值与缺失值单独成箱 | 自动决定特殊值列表 |
| IV / WOE 计算并按变量输出 | 建模或特征筛选 |

---

## 环境与安装

- **Python**：≥ 3.9
- **核心依赖**：`numpy`、`pandas`、`scikit-learn`（随主包安装）
- **可选**：`optbinning`（`method='optimal'` 暂未实现，后续支持）

从源码安装：

```bash
pip install -e ".[dev]"
```

---

## 导入

```python
from proscore.binning import Binning, BinningProcess, apply_bin_index
from proscore.binning._base import BinRecord, BinTable
```

---

## 快速开始

```python
import numpy as np
import pandas as pd
from proscore.binning import Binning

rng = np.random.default_rng(42)
n = 500

# 构造一个单调递增坏账率的变量
x = np.concatenate([
    rng.normal(0, 1, 100), rng.normal(2, 1, 100),
    rng.normal(4, 1, 100), rng.normal(6, 1, 100),
    rng.normal(8, 1, 100),
])
y = np.concatenate([
    rng.binomial(1, 0.05, 100), rng.binomial(1, 0.15, 100),
    rng.binomial(1, 0.35, 100), rng.binomial(1, 0.65, 100),
    rng.binomial(1, 0.9, 100),
])

df = pd.DataFrame({"score": x, "target": y})

# 卡方分箱，5 箱，要求递增
b = Binning(method="chi", n_bins=5, monotonic="increasing")
b.fit(df, y="target")

# 查看结果
print(b.iv_)
print(b.bin_table_["score"])
```

---

## API：`Binning`

```python
class Binning(
    method: str = "chi",
    n_bins: int = 10,
    min_bin_pct: float = 0.05,
    min_woe_diff: float = 0.1,
    monotonic: bool | int | str | None = None,
    confidence_val: float = 3.841,
    categorical_mode: str = "woe_per_value",
    special_values: dict | None = None,
    skip_values: dict | None = None,
    manual_cutoffs: dict | None = None,
    adjust_shape: bool = True,
    max_categories: int = 20,
)
```

### 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `method` | `str` | `"chi"` | 分箱算法，见下方表格 |
| `n_bins` | `int` | `10` | 目标分箱数（实际箱数受数据与阈值约束） |
| `min_bin_pct` | `float` | `0.05` | 最小箱占比。低于此阈值的箱在调整阶段被合并 |
| `min_woe_diff` | `float` | `0.1` | 最小 WOE 差。邻箱 WOE 差小于此值则合并 |
| `monotonic` | 多类型 | `None` | 趋势约束，详见 [趋势约束](#趋势约束) |
| `confidence_val` | `float` | `3.841` | 卡方置信度阈值（仅 `method="chi"` 时生效）。默认值对应自由度 1、95% 置信水平 |
| `categorical_mode` | `str` | `"woe_per_value"` | 分类变量策略，详见 [分类变量策略](#分类变量策略) |
| `special_values` | `dict` | `None` | 特殊值列表，`{列名: [值列表]}`。这些值单独成箱，不参与算法分箱 |
| `skip_values` | `dict` | `None` | 排除值列表，`{列名: [值列表]}`。这些值在分箱前直接丢弃 |
| `manual_cutoffs` | `dict` | `None` | 专家指定切点，`{列名: [切点列表]}`。指定后跳过算法分箱与趋势调整，仅做趋势检测 |
| `adjust_shape` | `bool` | `True` | 是否执行趋势调整与箱占比/WOE 差检查。设为 `False` 时仅用算法切点，不做任何合并 |
| `max_categories` | `int` | `20` | 分类变量判定阈值 |

### 分箱算法

| `method` | 算法 | 需要 `target` | 说明 |
|----------|------|--------------|------|
| `"chi"` | 卡方合并 | 是 | 先等频预分箱至 50 箱，再按卡方值迭代合并，卡方值低于 `confidence_val` 时停止 |
| `"tree"` | 决策树 | 是 | 单变量 CART，`max_leaf_nodes=n_bins`。切点取自树的分裂阈值 |
| `"frequency"` | 等频 | 否 | `pd.qcut` 等频切分，自动处理重复分位点 |
| `"distance"` | 等距 | 否 | 值域等间距切分 |
| `"optimal"` | MIP 优化 | — | 需 `optbinning` 包，暂未实现 |

### 分类变量策略

| `categorical_mode` | 说明 |
|-------------------|------|
| `"woe_per_value"` | 每个 distinct 值独立一箱 |
| `"badrate_merge"` | 按坏账率排序后，合并相邻坏账率差 < 0.05 或占比不足的类别 |
| `"freq_merge"` | 保留 top-10 高频类别各一箱，其余归入 "Other" |
| `"custom"` | 用户传入映射表，暂未实现 |

> 分类变量不参与趋势调整，`monotonic` 约束仅对连续变量生效。

### 趋势约束

`monotonic` 接受以下值：

| 输入 | 含义 | 内部码 |
|------|------|--------|
| `None` / `False` / `0` | 无约束（默认） | 0 |
| `True` / `1` / `"increasing"` / `"ascending"` | 坏账率严格递增 | 1 |
| `-1` / `2` / `"decreasing"` / `"descending"` | 坏账率严格递减 | 2 |
| `3` / `"u"` / `"valley"` | U 型（中间低两端高） | 3 |
| `4` / `"inverted_u"` / `"peak"` | 倒 U 型（中间高两端低） | 4 |

当最终分箱结果与预设趋势不符时，该变量会被记录为 `trend_match=False` 并发出 `UserWarning`，建模阶段可据此排除。

### 特殊值与缺失值

特殊值和缺失值**不参与分箱算法**，在主分箱完成后追加为独立箱：

- 特殊值：每类值单独一箱，箱号为 `n_main + i`
- 缺失值：一箱，箱号为 `n_main + len(special_values)`

### 专家指定切点

通过 `manual_cutoffs` 传入切点后：

- 跳过算法分箱，直接使用专家切点
- 跳过趋势调整（不会修改切点）
- **仍会**检测实际趋势并与 `monotonic` 预设对比，不匹配时发出警告

```python
b = Binning(manual_cutoffs={"income": [3000, 8000, 15000]}, monotonic="decreasing")
b.fit(df, y="target")
bt = b.bin_table_["income"]
if not bt.trend_match:
    print(f"实际趋势 {bt.monotonic} 与预设不符，请检查切点")
```

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `bin_table_` | `dict[str, BinTable]` | 每列的分箱结果 |
| `iv_` | `pd.DataFrame` | 各列 IV，按 IV 降序 |
| `cutoffs_` | `dict[str, list[float]]` | 各列的切点 |
| `woe_` | `dict[str, dict[int, float]]` | 各列的 `{箱号: WOE}` |

### 方法

| 方法 | 说明 |
|------|------|
| `fit(X, y)` | 对 `X` 中每列执行分箱。`y` 可以是列名（在 `X` 中）或 `pd.Series` |
| `transform(X)` | 将 `X` 每列转为 0-based 箱号 |
| `fit_transform(X, y)` | `fit` + `transform` |

---

## API：`BinningProcess`

批量分箱，支持按特征覆盖参数。

```python
bp = BinningProcess(
    feature_config={
        "income": {"method": "chi", "n_bins": 5, "monotonic": "decreasing"},
        "age":    {"method": "tree", "n_bins": 4},
    },
    default_method="chi",
    default_n_bins=8,
)
bp.fit(df, y="target")
df_binned = bp.transform(df)
```

| 参数 | 说明 |
|------|------|
| `feature_config` | `{列名: {参数覆盖}}`。支持 `Binning.__init__` 的所有参数 |
| `default_method` | 未在 `feature_config` 中指定时的算法 |
| `default_n_bins` | 默认目标箱数 |
| `**default_kwargs` | 其他默认参数，透传给每列的 `Binning` 实例 |

`BinningProcess` 同样提供 `bin_table_` 和 `iv_` 属性。

---

## 从 Excel 导入变量预设

当变量较多时，通过 Excel 统一管理每个变量的分箱参数比在代码中逐个写 `feature_config` 更直观且不易出错。

Excel 文件需包含名为 `variables` 的 sheet，字段如下：

| 字段 | 必填 | 说明 |
|------|------|------|
| `variable` | 是 | DataFrame 中的列名 |
| `monotonic` | 否 | `"increasing"` / `"decreasing"` / `"u"` / `"inverted_u"` |
| `special_values` | 否 | 逗号分隔的特殊值，如 `"-999, -998"` |
| `dimension` | 否 | 业务维度标签，自动转为 `StepwiseSelector` 的 `feature_belong` |

加载并传入 `BinningProcess`：

```python
from proscore.utils import load_presets
from proscore.binning import BinningProcess
from proscore.selection import StepwiseSelector

presets = load_presets("variable_presets.xlsx")

bp = BinningProcess(
    feature_config=presets.feature_config,
    default_method="chi",
    default_n_bins=5,
)
bp.fit(df, y="target")

# feature_belong 给逐步回归用
ss = StepwiseSelector(
    feature_belong=presets.feature_belong,
    belong_max_pct=0.5,
)
```

可选地，Excel 中可含第二个 sheet `dimensions`（纯文档用途，加载时忽略）。

示例文件参考 `tests/variable_presets.xlsx`。

---

## 数据结构：`BinTable`

```python
@dataclass
class BinTable:
    var: str                   # 变量名
    bins: list[BinRecord]      # 各箱详情
    cutoffs: list[float]       # 切点（分类变量为空）
    iv_total: float            # 总 IV
    method: str                # 分箱方法
    n_bins: int                # 主分箱数（不含特殊值和缺失）
    monotonic: int             # 实际趋势码 (0-4)
    trend_preset: int          # 用户预设趋势码
    trend_match: bool          # 实际趋势是否匹配预设
    dtype: str                 # "continuous" 或 "categorical"
    special_values: list       # 特殊值列表
    has_missing: bool          # 是否有缺失箱
    missing_merged: bool       # 缺失箱是否曾合并；当前实现恒 False（预留）
    cat_mapping: dict          # 分类变量：{原始值 → 箱号}
```

```python
@dataclass
class BinRecord:
    bin_no: int                # 箱号 (0-based)
    min_val: float | None      # 左边界
    max_val: float | None      # 右边界
    count: int                 # 样本数
    count_bad: int             # 坏样本数
    count_good: int            # 好样本数
    bad_rate: float            # 坏账率
    woe: float                 # WOE 值
    iv: float                  # 该箱 IV
    bin_label: str             # 标签（如 "(-inf, 3.5]"、"A, B, C"）
```

---

## API：`apply_bin_index`

模块级纯函数，用于将单列按已拟合的 `BinTable` 转换为箱号，无需 `Binning` 实例。

```python
from proscore.binning import apply_bin_index

bin_indices = apply_bin_index(df["income"], bt)
```

适用于只需要逐个 apply 分箱结果的场景。

---

## 推荐工作流

1. **`inspect.detect`** / **`inspect.quality`** → 了解变量质量和单变量区分力
2. **`prefilter`**（粗筛） → 剔除缺失率高/单值率高/高相关的垃圾变量。这些检查不需要分箱
3. **`Binning(manual_cutoffs=...)`** → 对业务上有明确切分逻辑的变量（如年龄段）使用专家切点
4. **`Binning(method="chi", monotonic=...)`** → 对剩余连续变量卡方分箱，指定趋势约束
5. **检查 `trend_match`** → 对不匹配的变量确认是数据问题还是预设错误
6. **`transform`** → 生成 WOE 值供后续建模

---

## 数据与 API 约束（必读）

1. **目标编码**：`y` 必须为 `{0, 1}`，且 `1` 表示坏样本。`fit()` 入口会校验目标唯一值为 2，值不在 {0,1} 内则抛出 `ValueError`。

2. **列名全局唯一**：`fit()` 入口调用 `require_unique_column_labels(X)`，与 `inspect` 模块一致。重复列名会抛出 `ValueError`。

3. **参数名严格校验**：`Binning.__init__` 不接受未声明的关键字参数。拼写错误会直接抛出 `TypeError`，避免静默忽略。

4. **`categorical_mode` / `method` 在构造时校验**：非法值立即报错，不会等到 `fit()` 时才暴露。

5. **专家切点不被修改**：指定 `manual_cutoffs` 后，分箱算法和趋势调整均不执行，切点原样使用。

6. **分类变量不做趋势调整**：`monotonic` 约束仅对连续变量生效。分类变量通过 `categorical_mode` 中的策略定义分组。

---

## 常见问题与报错

| 现象 | 可能原因 | 处理方向 |
|------|----------|----------|
| `TypeError: Binning got unexpected keyword argument` | 参数名拼写错误 | 检查参数名是否与文档一致 |
| `ValueError: Unknown binning method` | `method` 不在合法列表中 | 使用 `"chi"` / `"tree"` / `"frequency"` / `"distance"` |
| `ValueError: Unknown categorical_mode` | `categorical_mode` 不在合法列表中 | 使用 `"woe_per_value"` / `"badrate_merge"` / `"freq_merge"` |
| `NotImplementedError: optimal binning` | 使用了 `method="optimal"` | 等待后续版本，或安装 `optbinning` |
| `ValueError: Target must be binary` | 目标列有 3 个以上唯一值 | 确认目标为二分类，且编码为 {0, 1} |
| `ValueError: Target values must be in {0, 1}` | 目标值为 {1, 2} 或其他非标准编码 | 将目标重新编码为 {0, 1}，1=坏 |
| `UserWarning: final bins trend is ...` | 最终趋势与 `monotonic` 预设不符 | 检查 `bt.trend_match`；考虑放宽约束或使用其他趋势方向 |
| 分类变量分箱结果中 `trend_match=False` | 分类变量 `monotonic` 恒为 0 | 分类变量不参与趋势约束，忽略此字段 |
| 某列未出现在 `bin_table_` 中 | 该列无有效数据（全为缺失/特殊值） | 检查列的缺失率 |
| 分箱后箱数远少于 `n_bins` | 数据不足以支撑更多箱，或趋势调整合并过多 | 检查 `adjust_shape` 或降低 `min_bin_pct` |

---

## 相关文档

- OpenSpec（字段级规格）：[`docs/spec/02-binning.md`](../spec/02-binning.md)
- 架构设计：[`docs/spec/01-architecture.md`](../spec/01-architecture.md)
- 前置模块（数据探查）：[`inspect.md`](inspect.md)

若本手册与实现不一致，**以当前版本源码与 OpenSpec 为准**。
