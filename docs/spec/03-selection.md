# OpenSpec: Selection 特征筛选模块

## 标识
- **模块名**: proscore.selection
- **优先级**: P0
- **依赖**: numpy, pandas, scikit-learn, statsmodels

## 功能规格

### F1：Filter 初筛
- **缺失率筛选** `max_missing_rate`：高于阈值的特征剔除
- **单值率筛选** `max_one_value_rate`：高于阈值的特征剔除
- **IV 筛选** `iv_range=(0.02, None)`：低于下界剔除；上界为 `None` 时不设上限
- **单变量 AUC 筛选** `min_auc`：低于阈值的特征剔除
- **PSI 筛选** `max_psi`：高于阈值的特征剔除
- **相关系数筛选** `max_corr=0.7`：高度相关的特征只保留 IV 最高的
- **VIF 筛选** `max_vif=10`：高于阈值的特征剔除

### F2：双向迭代 (Stepwise)
- **前向迭代**：从基变量集开始，逐个尝试加入候选特征，用目标函数评估
- **后向迭代**：从前向结果中，逐个尝试剔除特征，评估是否提升目标
- **LR 条件检查**（全可参数化关闭）：
  - Pvalue 检查：每个特征系数显著性检验
  - 系数正负号检查：配置 `coef_sign='positive'/'negative'/None`
  - VIF 检查：入模变量集多重共线性
  - 相关系数检查：入模变量间相关性
- **目标函数**：内置 `ks_reduce` / `auc_reduce`（及别名 `ks` / `auc`）为  
  `test_metric + (1-r)*trn_metric - r*metric_reduce`；亦可传入 `Callable[[dict[str, float]], float]`
- **来源覆盖度控制**：`feature_belong` 字典配置，控制同来源变量数
- **重要性扰动**：陷入局部最优时随机剔除一部分特征再重启
- **退出条件**：
  - 指标达标且变量数满足约束
  - 多轮循环无变化
  - 超时/超轮次
  - 无可用变量

### F3：强制补入
- 迭代结束后变量数不足时，强制补入特征直至达到最小变量数要求
- 补入过程仍通过 LR 条件检查

## API 签名

```python
class Filter:
    """特征初筛"""
    def __init__(self,
                 max_missing_rate: float = 0.8,
                 max_one_value_rate: float = 0.95,
                 iv_range: tuple = (0.02, None),
                 min_auc: float | None = None,
                 max_psi: float | None = None,
                 max_corr: float = 0.7,
                 max_vif: float | None = None,
                 n_selected: int | None = None):  # 保留前N个特征
        ...

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_test: pd.DataFrame | None = None,
            bin_table: dict | None = None) -> 'Filter':
        """*y* 为二元目标（1=坏），必选。"""
        ...

    @property
    def support_(self) -> list[str]:
        ...

    @property
    def iv_(self) -> pd.DataFrame:
        ...

    @property
    def quality_(self) -> pd.DataFrame:
        """变量质量总览表"""
        ...

class StepwiseSelector:
    """双向迭代特征选择器（核心差异化）"""
    def __init__(self,
                 # LR 条件检查（None = 不启用）
                 pvalue_threshold: float | None = 0.05,
                 coef_sign: str | None = 'positive',  # 'positive'/'negative'/None
                 vif_threshold: float | None = 10.0,
                 corr_threshold: float | None = 0.8,
                 # 来源控制
                 feature_belong: dict | None = None,  # {source_name: [features]}
                 belong_max_pct: float | None = None, # 单来源最大占比
                 # 迭代控制
                 perturbation: bool = True,
                 perturbation_pct: float = 0.1,
                 perturbation_add: int = 2,
                 max_iter_round: int = 100,
                 max_iter_time: int = 600,
                 same_round_exit: int = 4,
                 # 目标函数
                 r: float = 0.8,
                 objective: str | Callable = 'ks_reduce',  # 'ks_reduce'|'auc_reduce'|callable
                 goal_threshold: float = 0.01,
                 # 变量数约束
                 n_min: int = 5,
                 n_max: int = 15,
                 force_fill: bool = True):

    def fit(self, X: pd.DataFrame, y: pd.Series,
            candidates: list[str] | None = None,
            force_in: list[str] | None = None,
            sort_df: pd.DataFrame | None = None,
            X_test: pd.DataFrame | None = None,
            y_test: pd.Series | None = None) -> 'StepwiseSelector':
        """*X* 为训练集 WOE；*X_test*/*y_test* 可选，用于计算 test KS/AUC 及衰退项。"""
        ...

    @property
    def support_(self) -> list[str]:
        ...

    @property
    def record_(self) -> dict:
        """每轮迭代的详细记录（变量集/指标/通过状态）"""
        ...

    @property
    def best_performance_(self) -> dict:
        ...

    @property
    def model_(self) -> Any:
        """最佳 LR 模型"""
        ...
```
