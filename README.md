# ProScore

[![PyPI version](https://img.shields.io/pypi/v/proscore.svg)](https://pypi.org/project/proscore/)
[![Python](https://img.shields.io/pypi/pyversions/proscore.svg)](https://pypi.org/project/proscore/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**生产级评分卡开发工具包**  
端到端的确定性评分卡建模管线，为银行和金融机构的信用评分卡建模场景设计, 满足对可解释性、合规性和稳定性的要求。

---

## 目录

- [三种使用方式](#三种使用方式)
- [核心功能概览](#核心功能概览)
- [安装](#安装)
- [依赖](#依赖)
- [License](#license)

---

## 三种使用方式

ProScore 提供三种递进的使用方式，从零代码到完全自定义，按需选择。

| 方式 | 适合 | 门槛 |
|------|------|------|
| [A. 模块独立使用](#a-模块独立使用) | 需要自定义逻辑、部分环节手动干预 | 熟悉 Python |
| [B. 链式 API](#b-链式-api) | 标准建模流程，一行到底 | 会写 Python |
| [C. Excel 配置驱动](#c-excel-配置驱动) | 业务人员零代码，一套 Excel 跑到底 | 会填 Excel |

### A. 模块独立使用

每个模块可单独 `import`，适合在任意环节插入自定义逻辑。

```python
from proscore.inspect import detect, quality
from proscore.selection import Filter
from proscore.binning import Binning
from proscore.transform import WOETransformer
# ... 按需组合
```

详见 [使用指南](https://github.com/lqwzlh/proscore/tree/main/docs/%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97) 各模块手册。

### B. 链式 API

标准建模流程，数据切分与模型边界严格分离（Train/Test/OOT）。

```python
import proscore as ps

p = (
    ps.ProScore()
    .read(train=df_train, test=df_test, oot=df_oot, target="bad_flag")
    .detect()
    .prefilter()
    .bin(method="chi", n_bins=5)
    .refine(iv_range=(0.02, None))
    .mine_rules(method="exhaustive", min_lift=3.0)  # 可选：规则挖掘
    .transform()
    .select()
    .fit(odds=20, pdo=20, base_score=600)
    .scorecard()
    .evaluate()  # 自动汇报 train / test / oot 三列指标
)
```

> `train` 必传，`test` 和 `oot` 可选。分箱/WOE 只在 train 上拟合；逐步回归用 test 监控过拟合；OOT 仅用于最终评估。
>
> 完整教程见 [notebooks/ProScore完整建模流程.ipynb](notebooks/ProScore完整建模流程.ipynb)
>
> **诊断增强**（v0.2+）：`.evaluate().diagnose()` 生成 4 层结构化健康报告（含根因变量），支持 `thresholds=...` 自定义阈值，适配不同机构/产品风控偏好。

### C. Excel 配置驱动

拿模板填参数，一行命令跑通全流程。**不需要写一行代码。**

```bash
# 1. 获取空白配置模板（二选一）
cp examples/pipeline_template.xlsx ./my_project/   # 克隆仓库后可直接复制
# 或: proscore template ./my_project/

# 2. 打开 Excel，填 data_file、target、time_col 等参数

# 3. 运行
proscore run my_project/pipeline_template.xlsx

# 可选：导出等效 Python 脚本
proscore run my_project/pipeline_template.xlsx --output-script run.py
```

模板含 8 个 Sheet（Global / Data / Steps / Binning / Screening / Modeling / Rules / Variables），每个参数带中文说明、可选范围和默认值。留空 = 使用默认值。无 OOT 时最少只需填 `data_file`、`target`、`time_col` 3 个格子；有 OOT 时再补充时间切分参数。

详细参数说明见 [pipeline-config.md](https://github.com/lqwzlh/proscore/blob/main/docs/%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97/pipeline-config.md)

---

## 核心功能概览

| 模块       | 核心能力                                      | 业务价值                              |
|------------|-----------------------------------------------|---------------------------------------|
| 数据探查   | IV/AUC/KS 三指标 + PSI 时序稳定性 + 相关性/VIF | 快速筛选优质变量，识别分布漂移风险    |
| 分箱       | 4 种单调趋势 + 5 种分箱方法 + 两阶段趋势校验   | 确保 WOE 趋势符合业务逻辑，满足监管   |
| 逐步回归   | 双向选择 + 五重约束（p值/符号/VIF/相关/来源） | 严谨的多重共线性控制与维度归属管理    |
| 模型监控   | Score/Feature PSI + 规则引擎告警 + JSON 持久化 | 投产后持续验证，自动风险预警          |
| 报告生成   | 7 章自动 Markdown 报告（含图表）              | 银保监合规文档一键生成                |
| 模型诊断   | 4 层健康检查 + 根因定位 + 可自定义阈值        | 投产前自动风险识别，支持策略微调      |

### 设计原则

- **确定性**：相同输入 → 相同输出，不依赖随机优化器。
- **sklearn 风格**：统一 `fit()` / `transform()` 接口。
- **生产就绪**：内置 unseen 处理、inf 容错、分箱序列化。
- **轻量核心**：仅 numpy/pandas/scikit-learn/statsmodels，XGBoost/LightGBM 为可选依赖。

---

## 安装

核心依赖仅需 numpy、pandas、scikit-learn、statsmodels，无重依赖：

```bash
pip install proscore
```

如需使用 XGBoost 或 LightGBM 作为变量质量评估的备选估计器（在 `inspect.quality()` 中启用 `estimator="xgb"` 或 `estimator="lgb"`），安装对应的可选依赖组：

```bash
pip install proscore[inspect-ml]
```

> 为什么是可选？XGBoost 和 LightGBM 体积较大，且涉及平台相关的编译依赖。绝大多数场景下，默认的逻辑回归估计器已经足够可靠。仅在需要用树模型对变量进行非线性排序时才需安装。
>
> 后续 AI / LLM 相关功能同样会以可选依赖组方式发布（如 `proscore[ai]`），不强制安装，不拖累核心包体积。

---

## 依赖

- Python >= 3.9
- numpy >= 1.20
- pandas >= 1.5
- scikit-learn >= 1.2
- statsmodels >= 0.13

**可选依赖**：

| 依赖组 | 安装命令 | 用途 |
|--------|---------|------|
| `inspect-ml` | `pip install proscore[inspect-ml]` | XGBoost / LightGBM 用于变量质量评估 |
| `excel` | `pip install proscore[excel]` | openpyxl，用于 `proscore run` 和 `load_presets()` |

> `proscore run` 命令由 `[project.scripts]` 注册，安装后即可使用。

## License

MIT