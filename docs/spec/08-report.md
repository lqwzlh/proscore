# OpenSpec: Report 报告生成模块

## 标识
- **模块名**: proscore.report
- **优先级**: P1
- **当前实现**: ReportBuilder（Markdown 轻量报告）
- **依赖**: pandas, numpy

## 功能规格

### F1：ReportBuilder 流式构建器
- 通过 `from_proscore(ps)` 一键从完整 ProScore 链式对象注入全部数据
- 或使用 `with_*` 方法按需注入各模块结果（inspect、binning、filter、stepwise、model、evaluate、stability）
- 支持自定义报告标题、项目名、建模人员、建模目的
- `build()` 返回 Markdown 字符串，`save(path)` 直接写入文件

### F2：支持的报告章节
- 报告头（标题、项目、日期、人员、目的）
- 建模概览（样本量、坏账率、分箱方法、入模变量数、KS/AUC/PSI）
- 数据探查结果（detect + quality 表）
- 分箱结果（各变量 BinTable 摘要）
- 特征筛选过程（Filter 参数 + 支持变量、Stepwise 迭代记录 + 最优性能）
- 评分卡（ScoreCard 变量分 + 评分卡表）
- 模型评估（Train/Test KS、AUC、PSI、分布表）
- 时序稳定性（可选）
- 建模结论

### F3：与 ProScore 集成
- 推荐使用 `ReportBuilder.from_proscore(ps)`，自动读取 `ps.detect_result`、`ps.quality_result`、`ps.filter_`、`ps.binner_`、`ps.selector_`、`ps.scorecard_`、`ps.eval_result` 等公开属性
- 也支持模块级独立使用

## API 签名

```python
class ReportBuilder:
    def __init__(
        self,
        title: str = "评分卡建模报告",
        project: str = "",
        modeler: str = "",
        purpose: str = "",
    ): ...

    @classmethod
    def from_proscore(cls, ps, **kwargs) -> ReportBuilder: ...

    def with_inspect(self, detect=None, quality=None, corr=None, vif=None) -> ReportBuilder: ...
    def with_target_distribution(self, df, target, time_col=None) -> ReportBuilder: ...
    def with_binning(self, bin_tables: dict, iv_table=None) -> ReportBuilder: ...
    def with_filter(self, f) -> ReportBuilder: ...
    def with_stepwise(self, selector) -> ReportBuilder: ...
    def with_model(self, scorecard) -> ReportBuilder: ...
    def with_evaluate(self, eval_result: dict) -> ReportBuilder: ...
    def with_stability(self, stability_result: pd.DataFrame) -> ReportBuilder: ...

    def build(self) -> str: ...
    def save(self, path: str) -> str: ...
```

当前阶段仅输出 Markdown，后续可扩展 HTML/Word 模板。
