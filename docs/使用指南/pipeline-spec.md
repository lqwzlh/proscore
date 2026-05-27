# ProScore · `PipelineSpec` 参数集中配置

在链式 API 中，把各阶段默认参数收进一个 `PipelineSpec`，通过 `ProScore.apply(spec)` 注入；各步骤方法上的**显式 kwargs 仍优先**于 spec。

> 与 [pipeline-config.md](pipeline-config.md) 的 **PipelineConfig**（Excel 零代码）不同：`PipelineSpec` 面向 Python 脚本 / Notebook，不读 Excel。

---

## 适用场景

- 同一套参数要在多次实验里复用（改一处即可）
- Notebook 里链式调用太长，希望 `prefilter` / `bin` / `refine` 等保持无参或少量覆盖
- 规则挖掘与建模参数需要版本化管理（如 Git 里维护一个 spec 字典）

---

## 快速开始

```python
from proscore import ProScore, PipelineSpec

spec = PipelineSpec(
    prefilter={"max_corr": 0.75, "max_vif": 10, "iv_range": None, "max_psi": None},
    binning={"method": "chi", "n_bins": 5},
    refine={"iv_range": (0.02, None), "max_psi": 0.25},
    rules={"method": "exhaustive", "min_lift": 3.0},
    select={"n_min": 5, "n_max": 12, "pvalue_threshold": 0.05},
    model={"odds": 20, "pdo": 20, "base_score": 600},
)

p = (
    ProScore()
    .read(train=df_train, test=df_test, target="bad_flag")
    .apply(spec)
    .detect()
    .prefilter()
    .bin()              # 使用 spec.binning 的 method / n_bins
    .refine()
    .mine_rules()       # 可选；无参时使用 spec.rules → RuleMiner
    .transform()
    .select()           # 使用 spec.select → StepwiseSelector
    .fit()              # 使用 spec.model 的 odds / pdo / base_score
    .scorecard()
    .evaluate()
)
```

---

## 字段说明

| 字段 | 对应链方法 | 传给 |
|------|------------|------|
| `prefilter` | `prefilter()` | `Filter` |
| `binning` | `bin()` | `Binning` / `BinningProcess`（`method`、`n_bins` 等） |
| `refine` | `refine()` | `Filter` |
| `rules` | `mine_rules()` | `RuleMiner`（**须显式调用** `mine_rules()`） |
| `select` | `select()` | `StepwiseSelector` |
| `model` | `fit()` | `ScoreCard`（`odds`、`pdo`、`base_score` 等） |

未接入 spec 的步骤：`detect`、`quality`、`transform`、`evaluate`（参数仍写在链上）。

---

## 合并规则

1. **显式优先**：`bin(method="dt", n_bins=8)` 会覆盖 `spec.binning` 里同名字段。
2. **`bin()` / `fit()` 位置参数**：签名上的 `method`、`n_bins`、`odds` 等若未出现在 `**kwargs` 中，也会从 spec 读取（与 `_merge_kw` 互补）。
3. **`merge()`**：按段浅合并字典，返回新 spec，不修改原对象。

```python
base = PipelineSpec(binning={"method": "chi", "n_bins": 5})
exp = base.merge(binning={"n_bins": 8})  # method 仍为 chi，n_bins=8
```

---

## 与规则挖掘

- `rules={...}` 只在调用 **`mine_rules()`** 时生效；`apply(spec)` **不会**自动挖规则。
- 规则变量由 `select()` 从建模候选中排除；`transform()` 仍可为规则变量生成 WOE（见 [rules.md](rules.md)）。

---

## 常见问题

**Q: 和 PipelineConfig 能互转吗？**  
A: 当前无自动转换。Excel 流程用 `PipelineConfig.from_excel()` + `run_pipeline()`；代码流程用 `PipelineSpec` + 链式 API。

**Q: `select(method="stepwise")` 还有用吗？**  
A: 链上仅支持逐步回归；若传入 `method=` 会被忽略并发出 `UserWarning`。请把 `StepwiseSelector` 支持的参数写在 `spec.select` 或 `select(...)` 的 kwargs 里。

---

## 相关文档

- OpenSpec：[`docs/spec/06-rules.md`](../spec/06-rules.md)（规则段 `rules`）
- 规则模块：[rules.md](rules.md)
- Excel 配置：[pipeline-config.md](pipeline-config.md)
