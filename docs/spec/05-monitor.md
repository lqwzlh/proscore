# OpenSpec: Monitor 模型监控模块

## 标识
- **模块名**: proscore.monitor
- **优先级**: P1（先出接口，实现可后续）
- **依赖**: numpy, pandas

## 功能规格

### F1：PSI 追踪
- 存储模型上线时的基准分布（base distribution）
- 定期计算当前分布 vs 基准分布的 PSI
- PSI 阈值预警（>0.1 关注，>0.25 需重训）
- 支持 PSI 趋势图数据输出

### F2：KS/AUC 衰减追踪
- 记录模型在各时间窗口内的 KS/AUC
- 追踪衰减率：`1 - current_ks / base_ks`
- 衰减阈值预警

### F3：规则引擎
```python
# 预置规则示例
MONITOR_RULES = [
    {"metric": "psi", "level": "warn", "condition": "> 0.1", "action": "review"},
    {"metric": "psi", "level": "fail", "condition": "> 0.25", "action": "retrain"},
    {"metric": "ks_decay", "level": "warn", "condition": "> 0.15", "action": "review"},
    {"metric": "ks_decay", "level": "fail", "condition": "> 0.3", "action": "retrain"},
]
```
- 用户自定义规则：`condition` 支持简单表达式
- 规则动作：`review`（标记关注）/ `retrain`（建议重训）/ `alarm`（立即告警）

### F4：LLM 诊断接口（AI 插件）
- 收集监控指标 + 特征分布变化
- 生成诊断 prompt 发送给 LLM
- 输出诊断结论 + 迭代建议（自然语言）

## API 签名

```python
@dataclass
class MonitorResult:
    psi: float
    psi_detail: pd.DataFrame
    ks_current: float
    ks_base: float
    ks_decay: float
    auc_current: float
    auc_base: float
    auc_decay: float
    alerts: list[str]
    diagnose: str | None  # LLM 诊断文本

class ModelMonitor:
    """模型监控器"""
    def __init__(self,
                 base_predictions: pd.Series,  # 基准分箱分布
                 bins: int = 10,
                 rules: list[dict] | None = None,
                 ai_provider: str | None = None):
        ...

    def track(self, current_predictions: pd.Series,
              current_ks: float, current_auc: float,
              features: pd.DataFrame | None = None) -> MonitorResult:
        ...

    def diagnose(self, result: MonitorResult) -> str:
        """LLM 诊断"""
        ...

class MonitorRules:
    """监控规则引擎"""
    @classmethod
    def default(cls) -> list[dict]: ...
    @classmethod
    def from_yaml(cls, path: str) -> list[dict]: ...
    def evaluate(self, metrics: dict) -> list[str]: ...
```
