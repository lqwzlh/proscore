# OpenSpec: Rules 策略规则挖掘模块

## 标识
- **模块名**: `proscore.rules`
- **优先级**: P1
- **依赖**: numpy, pandas, sklearn (DecisionTreeClassifier)

## 功能规格

### F1：三种搜索方法

| 方法 | 参数 | 说明 |
|------|------|------|
| `exhaustive` | `max_depth=3` | 穷举 1-3 变量组合的每个分箱区间 |
| `tree` | `max_tree_depth=4` | 单变量决策树，每片叶节点输出一条规则 |
| `apriori` | `max_depth=3` | Apriori 剪枝：先搜单变量，top-N 胜者交叉 |

### F2：规则评估指标

每条规则输出以下指标：

| 指标 | 说明 |
|------|------|
| `lift` | 规则 Precision / 整体坏账率 |
| `precision` | 命中样本中坏样本占比 |
| `recall` | 捕获的坏样本 / 全局坏样本 |
| `hit_rate` | 命中样本数 / 总样本数 |
| `single_hit_rate` | 仅被本规则命中的样本占比（=0 表示冗余） |

### F3：分箱集成

- 传 `bin_table`：按 `BinTable.cutoffs` 生成规则条件 `feat in (lo, hi]`
- 不传：自动等频分 5 箱

### F4：候选池互斥

- `mine_rules()` 后的 `select()` 自动排除规则已占用的变量
- 符合行业规范：策略规则和评分卡变量不重叠

## API 签名

```python
@dataclass
class RuleRecord:
    rule: str              # "debt_ratio in (0.60, 1.00] AND age in (20.00, 35.00]"
    hit_count: int
    bad_count: int
    good_count: int
    hit_rate: float
    precision: float
    recall: float
    lift: float
    single_hit_count: int
    single_hit_rate: float

class RuleMiner:
    def __init__(
        self,
        method: str = "exhaustive",     # exhaustive | tree | apriori
        max_depth: int = 3,
        max_tree_depth: int = 4,
        min_lift: float = 3.0,
        min_hit_rate: float = 0.01,
        max_hit_rate: float = 0.20,
        max_rules: int = 20,
    ):
        ...

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        bin_table: dict | None = None,
    ) -> RuleMiner:
        ...

    @property
    def rules_table_(self) -> pd.DataFrame:
        ...

    @property
    def used_features_(self) -> list[str]:
        ...

# ProScore 链式集成
class ProScore:
    def mine_rules(self, **kwargs) -> ProScore:
        """Mine rules before transform / select.  Rules auto-excluded from model."""
        ...

    @property
    def rulemine_(self) -> RuleMiner | None:
        ...

    @property
    def rules_table_(self):
        ...

    # select() 自动排除 _rule_features
```

### PipelineSpec 集成

``PipelineSpec.rules`` 字段接受所有 ``RuleMiner`` 构造参数，通过 ``ProScore.apply(spec)`` 注入链式 API。

### Excel / PipelineConfig 集成

- Steps 开关 ``mine_rules``（默认 **off**）；``mine_rules=on`` 须 ``refine=on``。
- 模板 **Rules** Sheet 参数名为裸名（``method``、``min_lift`` 等），解析后写入 ``rules_cfg``，再传给 ``RuleMiner``。
- 流水线位置：``refine → mine_rules → transform → select``；``export_csv=on`` 时输出 ``{project}_report/rules_table.csv``。
- 详见 [`docs/使用指南/pipeline-config.md`](../使用指南/pipeline-config.md)。
