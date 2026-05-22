# ProScore 正式设计文档 — 一期

> 二期规划：AI 插件层（NL2Code / 单调性语义预判 / LLM 诊断）、报告生成（HTML/Word/Markdown）、模型监控、策略规则挖掘

## 1. 项目概述

ProScore 是一个**生产级的评分卡建模工具库**，定位为"评分卡自动化构建引擎"。
一期范围为**确定性评分卡建模全流程**：探查 → 粗筛（prefilter）→ 分箱 → 精筛（refine）→ WOE 转换 → 双向迭代筛选 → LR 建模 → 评分卡转换 → 评估。

> **说明**：当前以逻辑回归评分卡（LR + WOE）为核心路径；树模型、深度学习等其他建模范式可按需使用底层模块，后续版本将逐步支持端到端流程。

## 2. 设计原则

### 2.1 核心原则
- **模块独立**：每个子模块可独立使用，互不强制依赖
- **兼容 sklearn**：核心类遵循 `.fit()` / `.transform()` / `.fit_transform()` 约定
- **双模式 API**：链式调用 + 模块化调用双模式，底层统一
- **参数化设计**：所有业务逻辑（含双向迭代中的 LR 检查条件）均以参数暴露，用户可开关

### 2.2 技术栈
- Python >= 3.9（兼容 3.9 ~ 3.13）
- numpy >= 1.20, pandas >= 1.5, scikit-learn >= 1.2
- statsmodels（LR 建模、P值、VIF）
- 仅核心依赖，可选依赖optbinning,xgboost,lightgbm

## 3. 整体架构

```
proscore/
│
├── inspect/          # 数据探查（质量报告、IV/缺失率/单值率）
│
├── binning/          # 分箱引擎（4种算法 + 单调性调整）
│   └── _adjust.py    # 趋势调整（单调性检查 / UV检查 / 合并调整）
│
├── transform/        # WOE 转换
│
├── selection/        # 特征筛选
│   ├── _filter.py    # IV/PSI/Corr/VIF 等 filter 方法
│   └── _stepwise.py  # 双向迭代引擎（核心差异化）
│
├── modeling/         # LR 建模 + 评分卡转换
│
├── evaluate/         # 模型评估（KS/AUC/PSI）
│
├── _data/            # 数据接入（DataFrame/CSV）
│
└── utils/            # 工具函数
```

### 3.1 数据流向（LR 评分卡推荐路径）

```
用户数据（DataFrame / CSV）
    │
    ▼
┌──────────┐
│  inspect  │ → 变量质量总览（detect / quality）
└─────┬────┘
      ▼
┌──────────┐
│ prefilter │ → 粗筛（缺失/单值/相关/VIF，不依赖分箱）
└─────┬────┘
      ▼
┌──────────┐
│  binning  │ → 分箱（BinTable，Train 上 fit）
└─────┬────┘
      ▼
┌──────────┐
│  refine  │ → 精筛（IV/PSI/AUC/VIF，依赖 bin_table_）
└─────┬────┘
      ▼
┌───────────┐
│ transform │ → WOE 转换（依赖分箱结果）
└─────┬─────┘
      ▼
┌───────────┐
│ selection │ → 逐步回归 + 来源归属控制
└─────┬─────┘
      ▼
┌───────────┐
│ modeling  │ → LR + 评分卡转换
└─────┬─────┘
      ▼
┌───────────┐
│ evaluate  │ → KS/AUC/PSI（含 Train/Test/OOT）
└───────────┘
```

> 树模型/深度学习路径可跳过 WOE，直接使用原始特征或简单分箱 + 底层模块。

## 4. 核心模块设计

### 4.1 双模式 API

#### 链式模式（用户快速体验）
```python
from proscore import ProScore

ps = (ProScore()
    .read(train=df_train, test=df_test, oot=df_oot, target='bad_flag')
    .detect()
    .prefilter(max_corr=0.8, iv_range=None, max_psi=None)
    .bin(method='chi', n_bins=10)
    .refine(iv_range=(0.02, None), max_psi=0.25)
    .transform(unseen_strategy='worst')
    .select(method='stepwise')
    .fit(odds=50, pdo=10, base_score=600)
    .scorecard()
    .evaluate()
)
```

说明：`prefilter` 不依赖分箱；`refine` 须在 `bin()` 之后（使用 `bin_table_`）。评分卡分值
`points = -pdo/ln(2) × coef × woe`（高分 = 低风险）。

#### 模块化模式（分步控制，推荐顺序）
```python
from proscore import inspect
from proscore.binning import Binning
from proscore.transform import WOETransformer
from proscore.selection import Filter, StepwiseSelector
from proscore.modeling import ScoreCard

target = 'bad_flag'
feat_cols = [c for c in df.columns if c not in {target, 'id'}]

# 1. 探查
inspect.detect(df, target=target)

# 2. 粗筛（不依赖分箱）
preflt = Filter(max_missing_rate=0.5, max_one_value_rate=0.95)
preflt.fit(df[feat_cols], df[target])

# 3. 分箱（Train 上 fit）
binner = Binning(method='chi', n_bins=10)
binner.fit(df[preflt.support_ + [target]], y=target)

# 4. 精筛（使用 bin_table_）
refine = Filter(iv_range=(0.02, None), max_psi=0.25)
refine.fit(df[preflt.support_], df[target], bin_table=binner.bin_table_)

# 5. WOE 转换
woe_tr = WOETransformer(unseen_strategy='worst')
woe_tr.fit(binner.bin_table_)
df_woe = woe_tr.transform(df[refine.support_])

# 6. 逐步回归 + 建模
selector = StepwiseSelector(pvalue_threshold=0.05, coef_sign='positive')
selector.fit(df_woe, df[target], candidates=refine.support_)

sc = ScoreCard(odds=50, pdo=10, base_score=600)
sc.fit(df_woe, y=target, features=selector.support_)
sc.scorecard(binner.bin_table_)
```

### 4.2 Binning 分箱模块

#### 设计要点
- **4 种基础算法**：卡方(chi)、等频(frequency)、等距(distance)、决策树(tree)
- **趋势调整引擎**：monotonicity_check → trend_adjust → UV_check 链
- **分类变量模块**：支持多种编码方式（WOE per category / badrate merge / target encoding）
- **可选后端**：detect optbinning 后启用 `method='optimal'`
- **分箱结果统一格式**：`BinTable` 数据结构

```python
# 与实现一致的设计摘要（详见 docs/spec/02-binning.md）
class Binning:
    def __init__(self, method='chi', n_bins=10, min_bin_pct=0.05,
                 min_woe_diff=0.1, monotonic=None, categorical_mode='woe_per_value', ...):
        ...

    def fit(self, X: pd.DataFrame, y: str | pd.Series) -> 'Binning':
        ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """返回分箱编号（0-based）"""
        ...

    @property
    def bin_table_(self) -> dict[str, BinTable]:
        """BinTable.bins 为 list[BinRecord]；连续变量 cutoffs 为切点列表"""
        ...
```

### 4.3 Selection 双向迭代模块

#### 设计要点
- **前向 + 后向迭代**：P 值、系数符号、VIF、相关性、目标函数与扰动机制均通过 `StepwiseSelector` 暴露，可按需关闭
- **全参数化**：所有业务逻辑的开关和阈值都暴露为参数，用户可关闭
- **来源覆盖度控制**：默认关闭，开启时需提供 `feature_belong` 字典

```python
class StepwiseSelector:
    """双向迭代特征选择器"""
    def __init__(self,
                 pvalue_threshold=None,
                 coef_sign=None,
                 vif_threshold=None,
                 corr_threshold=None,
                 feature_belong=None,
                 belong_max_pct=None,
                 perturbation=True,
                 perturbation_pct=0.1,
                 perturbation_add=2,
                 max_iter_round=100,
                 max_iter_time=600,
                 same_round_exit=4,
                 objective='ks_reduce',  # 或 'auc_reduce' / Callable[[dict], float]
                 r=0.8,
                 goal_threshold=0.01,
                 n_min=5,
                 n_max=15,
                 force_fill=True):
        ...
```

> AI 插件层（单调性预判 / NL2Code / LLM 诊断）、报告生成、模型监控、策略规则挖掘统一规划至二期。
