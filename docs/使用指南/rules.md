# ProScore · `rules` 模块使用指南

从候选变量中挖掘决策规则，输出可用于风控策略引擎的规则列表。

---

## 目录

- [核心概念](#核心概念)
- [导入](#导入)
- [快速开始](#快速开始)
- [API：`RuleMiner`](#apiruleminer)
  - [构造参数](#构造参数)
  - [方法](#方法)
  - [属性](#属性)
  - [数据结构：`RuleRecord`](#数据结构rulerecord)
- [三种搜索方法](#三种搜索方法)
- [规则评估指标](#规则评估指标)
- [链式集成](#链式集成)
- [常见问题](#常见问题)

---

## 核心概念

规则负责拦截极端风险（用原始值 + 阈值），评分卡负责综合排序（用 WOE + LR）。**规则和模型应该使用不同的变量**。两者变量重叠会导致规则调整反向冲击模型稳定性。

推荐流程：

```
prefilter → bin → refine → mine_rules → transform → select → fit → scorecard
                               │                       │
                          规则先挑              select 自动排除规则变量
                          原始值搜索          （WOE 表保留，供后续分析）          
```

> **互斥范围**：`select()` 从候选池中排除规则变量，但 `transform()` 仍会为它们生成 WOE 值。这样不影响模型，同时保留了规则变量供后续分析使用。

---

## 导入

```python
from proscore.rules import RuleMiner
```

---

## 快速开始

```python
import pandas as pd
from proscore.binning import Binning
from proscore.rules import RuleMiner

# 准备数据（原始特征 + 目标）
df = pd.read_csv("your_data.csv")
X = df[["income", "debt_ratio", "age", "num_inquiries"]]
y = df["bad_flag"]

# 先分箱
b = Binning(method="chi", n_bins=5)
b.fit(pd.concat([X, y], axis=1), y="bad_flag")

# 规则挖掘
rm = RuleMiner(method="exhaustive", min_lift=3.0, max_rules=10)
rm.fit(X, y, bin_table=b.bin_table_)
print(rm.rules_table_)
print(rm.used_features_)  # → 排除列表，传给 select()
```

---

## API：`RuleMiner`

### 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `method` | `str` | `"exhaustive"` | 搜索方法：`exhaustive` / `tree` / `apriori` |
| `max_depth` | `int` | `3` | 最多几变量交叉（exhaustive / apriori 模式） |
| `max_tree_depth` | `int` | `4` | 决策树最大深度（tree 模式） |
| `min_lift` | `float` | `3.0` | 最小 Lift，规则的坏账率须至少是大盘的 3 倍 |
| `min_hit_rate` | `float` | `0.01` | 最小命中率（太低没有业务意义） |
| `max_hit_rate` | `float` | `0.20` | 最大命中率（太高误杀过多好客户） |
| `max_rules` | `int` | `20` | 最多输出规则数（按 Lift 降序截取） |

### 方法

| 方法 | 说明 |
|------|------|
| `fit(X, y, bin_table=None)` | 执行规则挖掘。`bin_table` 为 `Binning.bin_table_`，传了就用分箱切点生成规则条件；不传则自动等频分 5 箱 |

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `rules_table_` | `pd.DataFrame` | 规则评估表，按 Lift 降序 |
| `used_features_` | `list[str]` | 规则中引用的变量名，用于 `select()` 排除 |

---

## 数据结构：`RuleRecord`

```python
@dataclass
class RuleRecord:
    rule: str                # 规则条件表达式
    hit_count: int           # 命中样本数
    bad_count: int           # 命中坏样本数
    good_count: int          # 命中好样本数
    hit_rate: float          # 命中率（命中数 / 总样本）
    precision: float         # 精确率（命中坏样本数 / 命中数）
    recall: float            # 召回率（命中坏样本数 / 全局坏样本数）
    lift: float              # 提升度（precision / 整体坏账率）
    single_hit_count: int    # 独立命中数（不被其他规则覆盖的）
    single_hit_rate: float   # 独立命中率
```

---

## 三种搜索方法

### exhaustive（默认）

对每个变量的每个分箱区间尝试 1-3 层交叉组合。适合候选变量不多（20-50 个）的场景，保证不遗漏。

```python
rm = RuleMiner(method="exhaustive", max_depth=2, min_lift=3.0)
```

### tree

每个变量单独训练一棵浅层决策树，每片叶子输出一条规则。速度快、可解释性强。

```python
rm = RuleMiner(method="tree", max_tree_depth=4, min_lift=3.0)
```

### apriori

两阶段：**Phase 1** 枚举所有通过 `min_lift` / `hit_rate` 阈值的单变量规则并全部保留；**Phase 2** 仅按 Lift 取 top-10 单变量规则做两两交叉（`max_depth >= 2` 时）。最终仍按 Lift 排序并由 `max_rules` 截断。适合在变量较多时减少交叉组合量。

```python
rm = RuleMiner(method="apriori", min_lift=4.0)
```

---

## 规则评估指标

| 指标 | 含义 | 判断标准 |
|------|------|---------|
| `lift` | 规则 Precision / 整体坏账率 | > 3 有效，> 5 优秀 |
| `precision` | 命中样本中坏样本比例 | 越高越精准 |
| `recall` | 捕获了多少全局坏样本 | > 5% 有业务意义 |
| `hit_rate` | 覆盖了多少总样本 | 0.01–0.20 合适 |
| `single_hit_rate` | 不被其他规则覆盖的独立命中 | > 0 则规则有独立价值 |

> `single_hit_rate = 0` 表示这条规则命中的样本全部被其他规则覆盖了，可以删除。

---

## 链式集成

```python
import proscore as ps

p = (
    ps.ProScore()
    .read(train=df_train, test=df_test, target="bad_flag")
    .prefilter()
    .bin(method="chi", n_bins=5)
    .refine(iv_range=(0.02, None))
    .mine_rules(method="exhaustive", min_lift=3.0)   # ← 规则挖掘
    .transform()
    .select(method="stepwise")                         # ← 自动排除规则变量
    .fit(odds=20, pdo=20)
    .scorecard()
    .evaluate()
)

print(p.rules_table_)    # 规则评估表
print(p.rulemine_)       # RuleMiner 实例
print(p.support_)        # 入模变量（不含规则变量）
```

> `mine_rules()` 是可选的。不调它则 select 照常。

---

## 常见问题

**Q: 规则挖掘必须在分箱之后吗？**
A: 不必须。`fit()` 的 `bin_table` 是可选的——不传则自动等频分 5 箱生成规则条件。但传了分箱结果（`binner.bin_table_`），规则条件会使用建模用的相同切点，条件更精准。此外，传了 `bin_table` 还能**自动挖掘缺失值规则**（如 `debt_ratio is missing`），因为 `BinTable` 记录了每个变量是否有缺失箱。

**Q: 哪些变量类型支持？**
A: 当前仅支持**数值型**变量。类别变量（如 `education`、`employment_type`）暂不支持，后续版本将基于分箱的 `cat_mapping` 生成 `feat in {A, B, C}` 规则。

**Q: 挖出来的规则能直接上线吗？**
A: **不能。** 规则挖掘基于训练集，未做 OOT 交叉验证。任何规则上线前必须在独立 OOT 数据上复验 Lift / Precision / Recall 是否稳定。`RuleMiner` 的输出是"候选规则"，不是"最终策略"。

**Q: `used_features_` 为什么是整变量名？**
A: 出于保守策略，只要某变量出现在任一条规则中，该变量就被整体从 `select()` 候选池中排除。后续可能支持更细粒度的排除（如仅排除命中规则的分箱区间）。

**Q: `used_features_` 为空怎么办？**
A: 说明没有找到满足阈值的规则。`select()` 不会排除任何变量，正常建模。

**Q: 为什么规则不用 WOE 值？**
A: 规则是给策略引擎用的，条件必须是业务人员能理解的原始值。`income <= 30000` 比 `income_WOE <= -0.5` 可读得多。

---

## Excel 配置驱动

零代码用户可在 pipeline 模板中开启规则挖掘：

1. **Steps** 表将 `mine_rules` 设为 `on`（默认 `off`）。
2. 在 **Rules** 表调整 `method`、`min_lift`、`max_rules` 等（参数名为裸名，与 `RuleMiner` 一致）。
3. 运行 `proscore run pipeline.xlsx`；若 `export_csv=on`，得到 `{project}_report/rules_table.csv`。

完整说明见 [pipeline-config.md](pipeline-config.md) 的 Sheet 7: Rules。

---

## 相关文档

- OpenSpec：[`docs/spec/06-rules.md`](../spec/06-rules.md)
- Excel 配置：[`pipeline-config.md`](pipeline-config.md)
- 参数集中配置：[`pipeline-spec.md`](pipeline-spec.md)
- 分箱模块：[`binning.md`](binning.md)
- 特征筛选：[`selection.md`](selection.md)
