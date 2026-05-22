# OpenSpec: Rules 策略规则挖掘模块

## 标识
- **模块名**: proscore.rules
- **优先级**: P1
- **依赖**: numpy, pandas, itertools

## 功能规格

### F1：单变量规则挖掘
- 从分箱结果中识别 badrate 异常高的箱
- 输出规则：`feature >= a and feature <= b → reject`
- 支持参数化阈值：`min_badrate`, `min_lift`, `min_support`

### F2：交叉规则挖掘
- 两两特征组合，寻找 badrate 异常高的组合区间
- Apriori 剪枝式搜索（pre-computed stats 加速）
- 支持参数化阈值：`min_badrate`, `min_support`, `max_features`

### F3：规则表达式 DSL
```python
RuleExpr("age > 60 and income < 5000 → reject", badrate=0.45, lift=3.2, support=0.02)
```

## API 签名

```python
@dataclass
class Rule:
    condition: str          # "age > 60 and income < 5000"
    action: str             # "reject" / "manual_review"
    badrate: float
    lift: float
    support: float
    feature: str | list

class RuleMiner:
    """规则挖掘器"""
    def __init__(self,
                 min_badrate: float = 0.3,
                 min_lift: float = 2.0,
                 min_support: float = 0.01):
        ...

    def univariate(self,
                   bins: dict[str, BinTable],
                   df: pd.DataFrame,
                   target: str) -> list[Rule]:
        ...

    def cross(self,
              df: pd.DataFrame,
              target: str,
              features: list[str],
              max_pairs: int = 50) -> list[Rule]:
        ...

    def to_dataframe(self, rules: list[Rule]) -> pd.DataFrame:
        ...

    def to_yaml(self, rules: list[Rule], path: str):
        ...
```
