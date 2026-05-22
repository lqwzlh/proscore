# ProScore · `report` 模块使用指南

面向建模人员的**Markdown 报告生成**工具。支持从完整 `ProScore` 链式对象一键生成建模报告，或按模块逐步注入数据后输出 Markdown 文件。

---

## 目录

- [模块职责与边界](#模块职责与边界)
- [快速开始](#快速开始)
- [API：`ReportBuilder`](#apireportbuilder)
- [推荐用法](#推荐用法)
- [报告内容说明](#报告内容说明)
- [相关文档](#相关文档)

---

## 模块职责与边界

| 会做 | 不做 |
|------|------|
| 从 ProScore 各模块结果自动组装 Markdown 报告 | 生成 Word / PDF |
| `save(..., write_html=True)` 同时输出 HTML 并内嵌图片 | 依赖 Jinja2 / 模板引擎（当前硬编码渲染） |
| 支持自定义标题、项目、人员、目的；图表支持相对路径或 base64 内嵌 | 自动美化复杂图表布局 |
| `build()` 返回字符串、`save()` 直接写文件 | - |

---

## 快速开始

```python
from proscore import ProScore
from proscore.report import ReportBuilder
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n, p = 200, 4
X = rng.normal(size=(n, p))
cols = [f"f{i}" for i in range(p)]
df_train = pd.DataFrame(X, columns=cols)
z = X[:, 0] + 0.3 * X[:, 1] + rng.normal(scale=0.5, size=n)
df_train["bad_flag"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

df_oot = df_train.sample(60, random_state=1)

ps = (
    ProScore()
    .read(train=df_train, test=df_test, oot=df_oot, target="bad_flag")
    .detect()
    .prefilter()
    .bin(method="chi", n_bins=5)
    .refine()
    .transform()
    .select(n_min=2, n_max=5)
    .fit(odds=50, pdo=10)
    .scorecard()
    .evaluate()
)

report = ReportBuilder.from_proscore(
    ps,
    title="信用卡评分卡开发报告",
    project="Retail Credit 2026Q2",
    modeler="张三",
    purpose="新客准入评分卡迭代",
)
report.save("./reports/scorecard_report.md")
```

---

## API：`ReportBuilder`

```python
class ReportBuilder:
    def __init__(
        self,
        title: str = "评分卡建模报告",
        project: str = "",
        modeler: str = "",
        purpose: str = "",
    )
```

### 工厂方法

- `from_proscore(ps, **kwargs) -> ReportBuilder`  
  最推荐的用法。自动读取 `ps` 的公开属性（`detect_result`、`quality_result`、`filter_`、`binner_`、`selector_`、`scorecard_`、`eval_result` 等）。

### 数据注入方法（链式调用）

- `with_inspect(detect, quality, corr=None, vif=None)`
- `with_target_distribution(df, target, time_col=None)`
- `with_binning(bin_tables, iv_table=None)`
- `with_filter(f)` — 传入 `Filter` 实例
- `with_stepwise(selector)` — 传入 `StepwiseSelector` 实例
- `with_model(scorecard)` — 传入 `ScoreCard` 实例
- `with_evaluate(eval_result: dict)`
- `with_stability(stability_result: pd.DataFrame)`

### 输出方法

- `build() -> str`：返回完整 Markdown 字符串
- `save(path: str) -> str`：写入文件并返回绝对路径（自动创建上级目录）

---

## 推荐用法

1. **完整流程推荐**：使用 `from_proscore(ps)`，最省事。
2. **模块化报告**：只想展示部分内容时，先实例化 `ReportBuilder()`，再调用需要的 `with_*` 方法。
3. **自定义元信息**：通过构造函数传入 `title`、`project`、`modeler`、`purpose`。

---

## 报告内容说明

报告自动包含以下章节（若对应数据存在）：

1. 报告头（标题、项目、日期、人员、目的）
2. 建模概览（样本量、坏账率、分箱方法、入模变量数、KS/AUC/PSI）
3. 数据探查结果
4. 分箱结果摘要
5. 特征筛选过程（Filter + Stepwise 迭代记录）
6. 评分卡
7. 模型评估指标与分布
8. 时序稳定性（若提供）
9. 建模结论

---

## 相关文档

- 技术规格：[`docs/spec/08-report.md`](../spec/08-report.md)
- 架构总览：[`docs/spec/01-architecture.md`](../spec/01-architecture.md)
- ProScore 主流程：用户手册首页
