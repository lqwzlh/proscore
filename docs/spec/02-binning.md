# OpenSpec: Binning 分箱模块

## 标识
- **模块名**: proscore.binning
- **优先级**: P0
- **依赖**: numpy, pandas, scikit-learn
- **可选依赖**: optbinning (method='optimal')

## 功能规格

### F1：4种基础分箱算法
- **等频分箱** `method='frequency'`：按分位数等分，自动处理重复分位点
- **等距分箱** `method='distance'`：按值域等距切分
- **卡方分箱** `method='chi'`：卡方检验合并，支持置信度参数
- **决策树分箱** `method='tree'`：CART 决策树分箱，支持 `max_leaf_nodes` 控制

### F2：分箱趋势调整
- **单调性检查**：`monotonic` 可传 `bool` / `int` / `str`（如 `"increasing"` / `"decreasing"`），内部映射为趋势预设码；不满足时通过邻箱合并逼近
- **单调性调整**：不满足单调的箱自动合并（按违约率差值最小的相邻箱合并）
- **UV检查**：识别 U 型 / V 型趋势并保留
- **专家单调方向**：支持传入预设方向，不满足时 `BinTable.trend_match=False`
- **实际趋势编码**：`BinTable.monotonic` 为 `int`（0=无/未知，1=递增，2=递减，3=U，4=倒U），与布尔型不同

### F3：分箱质量检查
- **最小箱占比** `min_bin_pct=0.05`：低于阈值的箱自动合并
- **最小WOE差距** `min_woe_diff=0.1`：WOE差距小于阈值的邻箱合并
- **特殊值处理**：支持指定特殊值单独成箱，不参与分箱算法
- **缺失值处理**：缺失默认单独成箱；文档中“小占比缺失箱合并到 badrate 相近箱”的**自动合并规则当前未实现**，`BinTable.missing_merged` 字段预留，当前恒为 `False`

### F4：分类变量处理
- **按值WOE** `categorical_mode='woe_per_value'`：每个取值独立一箱计算WOE
- **Badrate合并** `categorical_mode='badrate_merge'`：按badrate相邻合并类别
- **Frequency合并** `categorical_mode='freq_merge'`：低频类别合并到"其他"类
- **自定义映射** `categorical_mode='custom'`：接口保留，当前在 `Binning.__init__` 中抛出 `NotImplementedError`

### F5：可选MIP分箱
- 当 `method='optimal'` 且已安装 `optbinning` 时启用
- 原生支持单调性约束、最大/最小箱数
- 若无 optbinning，报错提示安装

### F6：批量处理
- `BinningProcess` 类支持一次性批量处理 DataFrame 所有特征
- 支持对每个特征单独配置参数（`feature_config` 字典）

## 数据结构

```python
@dataclass
class BinRecord:
    bin_no: int
    min_val: float | None
    max_val: float | None
    count: int
    count_bad: int
    count_good: int
    bad_rate: float
    woe: float
    iv: float
    bin_label: str = ""

@dataclass
class BinTable:
    var: str
    bins: list[BinRecord]
    cutoffs: list[float]
    iv_total: float
    method: str
    n_bins: int
    monotonic: int          # 实际趋势码：0~4（见实现与手册）
    trend_preset: int
    trend_match: bool
    dtype: str               # continuous | categorical
    special_values: list
    has_missing: bool
    missing_merged: bool     # 预留：缺失箱是否曾合并（当前实现恒 False）
    cat_mapping: dict        # 分类：原始值 → bin_no（WOE 映射由 WOETransformer 消费）
```

## API 签名

```python
class Binning:
    def __init__(self,
                 method: str = 'chi',           # chi|frequency|distance|tree|optimal
                 n_bins: int = 10,
                 min_bin_pct: float = 0.05,
                 min_woe_diff: float = 0.1,
                 monotonic: bool | int | None = None,  # True/False/1/-1/None
                 confidence_val: float = 3.841,
                 categorical_mode: str = 'woe_per_value',
                 special_values: dict | None = None,  # {var: [vals]}
                 skip_values: dict | None = None,     # {var: [vals]}
                 manual_cutoffs: dict | None = None,  # {var: [cutoffs]}
                 adjust_shape: bool = True,
                 **kwargs)

    def fit(self, X: pd.DataFrame, y: str | pd.Series) -> 'Binning':
        ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """返回分箱编号"""
        ...

    def fit_transform(self, X, y) -> pd.DataFrame:
        ...

    @property
    def bin_table_(self) -> dict[str, BinTable]:
        ...

    @property
    def iv_(self) -> pd.DataFrame:
        ...

    @property
    def cutoffs_(self) -> dict[str, list[float]]:
        ...

    @property
    def woe_(self) -> dict[str, dict[int, float]]:
        ...

class BinningProcess:
    """对 DataFrame 批量分箱，支持按特征单独配置"""
    def __init__(self,
                 feature_config: dict | None = None,  # {var: {method:, n_bins:, ...}}
                 default_method: str = 'chi',
                 default_n_bins: int = 10,
                 **default_kwargs)
    ...
```
