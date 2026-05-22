# ProScore 用户手册

ProScore 提供三种递进的使用方式：**模块独立使用**（自定义）、**链式 API**（标准流程）、**Excel 配置驱动**（零代码）。

---

## 三种使用方式

| 方式 | 文档 | 适合谁 | 门槛 |
|------|------|--------|------|
| 模块独立使用 | 下方各模块手册 | 需要自定义逻辑、部分环节手动干预 | 熟悉 Python |
| 链式 API | [README](../../README.md#b-链式-api) + [Notebook](../../notebooks/ProScore完整建模流程.ipynb) | 标准建模流程，一行到底 | 会写 Python |
| Excel 配置驱动 | [**pipeline-config.md**](pipeline-config.md) | 业务人员零代码，填 Excel 跑全流程 | 会填 Excel |

---

## 模块手册

| 模块 | 指南 | 功能 |
|------|------|------|
| 数据接入 | [data.md](data.md) | `DataReader` — 加载 DataFrame/CSV，校验列结构 |
| 数据探查 | [inspect.md](inspect.md) | `detect` / `quality` / `correlation` / `vif` / `stability` — 变量质量总览、区分力、共线性、时序稳定性 |
| 分箱 | [binning.md](binning.md) | `Binning` / `BinningProcess` — 4 种算法分箱 + 趋势调整 + 专家切点 + Excel 预设 |
| WOE 转换 | [transform.md](transform.md) | `WOETransformer` — 原始值→WOE，unseen 策略 |
| 特征筛选 | [selection.md](selection.md) | `Filter` / `StepwiseSelector` — 初筛 + 双向迭代 + 来源归属 |
| 评分卡建模 | [modeling.md](modeling.md) | `ScoreCard` — LR 拟合 + 评分卡转换 |
| 模型评估 | [evaluate.md](evaluate.md) | `evaluate` / `evaluate_by_period` — KS / AUC / PSI / 分布表 |
| 报告生成 | [report.md](report.md) | `ReportBuilder` — Markdown 建模报告自动生成 |
| 可视化 | [viz.md](viz.md) | `plot_binning` / `plot_ks` / `plot_roc` / `plot_score_distribution` |
| **Excel 配置** | [**pipeline-config.md**](pipeline-config.md) | **PipelineConfig** — 零代码 Excel 驱动全流程 |

## 推荐工作流

```
prefilter（粗筛）→ binning → refine（精筛）→ transform → select → fit → scorecard → evaluate
```

> 先 prefilter（缺失率/单值率，不需要分箱的检查）剔除垃圾变量，再分箱，再 refine（IV/PSI/AUC/VIF，用真实分箱值）精筛。train 用于拟合，test 监控过拟合，OOT 仅用于最终评估。

> **建模范式说明**  
> 上述工作流是 **逻辑回归评分卡（LR + WOE）** 的推荐路径。在该路径下，`bin`（分箱）必须先于 `refine` 和 `transform`，因为 IV、PSI、WOE 均依赖分箱后的统计结果。  
> 
> 如果使用树模型（LightGBM、XGBoost 等）、深度学习或其他非线性算法，通常不需要传统的最优分箱 + WOE 流程，可直接使用 ProScore 的底层模块（`Binning`、`Filter`、`StepwiseSelector` 等）按需组合，而不必走完整 `ProScore` 链式流程或 `PipelineConfig`。  
> 
> 当前版本以 LR 评分卡为核心，后续将逐步支持树模型等其他建模范式。

## 安装

```bash
pip install proscore[excel]        # 含 Excel 配置驱动功能
```

## 完整建模示例（Notebook）

端到端演示（含专家预设、Train/Test/OOT 切分、监控与报告）：[`notebooks/ProScore完整建模流程.ipynb`](../../notebooks/ProScore完整建模流程.ipynb)

## 相关文档

- 架构设计：[`docs/spec/01-architecture.md`](../spec/01-architecture.md)
- 分箱 OpenSpec：[`docs/spec/02-binning.md`](../spec/02-binning.md)
- 筛选 OpenSpec：[`docs/spec/03-selection.md`](../spec/03-selection.md)
- 探查 OpenSpec：[`docs/spec/04-inspect.md`](../spec/04-inspect.md)
- 监控 OpenSpec：[`docs/spec/05-monitor.md`](../spec/05-monitor.md)
- 报告 OpenSpec：[`docs/spec/08-report.md`](../spec/08-report.md)

本手册以当前版本源码为准。
