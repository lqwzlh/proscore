# ProScore · `viz` 模块使用指南

面向评分卡建模的**可视化辅助**：分箱分布图、KS 曲线、ROC 曲线、评分分布对比。所有函数返回 `matplotlib.figure.Figure`，可在 Notebook 内嵌或 `savefig` 导出。

---

## 目录

- [导入](#导入)
- [API](#api)
  - [`plot_binning`](#plot_binning)
  - [`plot_ks`](#plot_ks)
  - [`plot_roc`](#plot_roc)
  - [`plot_score_distribution`](#plot_score_distribution)
- [常见问题与报错](#常见问题与报错)
- [相关文档](#相关文档)

---

## 导入

```python
from proscore.viz import (
    plot_binning,
    plot_ks,
    plot_roc,
    plot_score_distribution,
)
```

> 需要安装 `matplotlib`：`pip install matplotlib`

---

## API

### `plot_binning`

```python
plot_binning(bt: BinTable, figsize=(9, 4)) -> Figure
```

单个变量的分箱分布：柱状图显示各箱样本量（左轴），折线显示坏账率（右轴），每箱上方标注 WOE 值。

```python
from proscore.viz import plot_binning

fig = plot_binning(binner.bin_table_["income"])
fig.savefig("income_binning.png", dpi=150)
```

### `plot_ks`

```python
plot_ks(y_true, prob, labels=("Model",), figsize=(7, 5)) -> Figure
```

KS 曲线：累积坏样本比例 + 累积好样本比例，标注 KS 最大值位置和数值。

```python
from proscore.viz import plot_ks

fig = plot_ks(y_test, prob_test)
fig.savefig("ks_curve.png")
```

### `plot_roc`

```python
plot_roc(y_true, prob, labels=("Model",), figsize=(6, 6)) -> Figure
```

ROC 曲线：TPR vs FPR，标注 AUC 值，包含随机基准线。

```python
from proscore.viz import plot_roc

fig = plot_roc(y_test, prob_test)
```

### `plot_score_distribution`

```python
plot_score_distribution(train_scores, test_scores=None, n_bins=20, figsize=(8, 4)) -> Figure
```

评分分布直方图：train + test 叠加对比，共用分箱边界。

```python
from proscore.viz import plot_score_distribution

scores_trn = scorecard.predict(df_trn_woe)
scores_tst = scorecard.predict(df_tst_woe)
fig = plot_score_distribution(scores_trn, scores_tst)
```

---

## 常见问题与报错

| 现象 | 可能原因 | 处理方向 |
|------|----------|----------|
| `ImportError: matplotlib is required` | 未安装 matplotlib | `pip install matplotlib` |
| 分箱图中文字体显示方块 | 系统缺少中文字体 | 设置 `plt.rcParams['font.sans-serif']` |
| KS/ROC 报错 `insufficient class balance` | y_true 只有一个类别 | 确认数据包含正负样本 |

---

## 相关文档

- 手册总览：[`index.md`](index.md)
- 分箱模块：[`binning.md`](binning.md)
- 报告生成：[`report.md`](report.md)
