# ProScore · `transform` 模块使用指南

面向评分卡建模的 **WOE 转换**：将原始特征值替换为 Weight-of-Evidence 值。接收分箱结果，输出 WOE DataFrame。

---

## 目录

- [导入](#导入)
- [快速开始](#快速开始)
- [API：`WOETransformer`](#apiwoetransformer)
  - [构造参数](#构造参数)
  - [unseen 策略](#unseen-策略)
  - [方法](#方法)
  - [属性](#属性)
- [常见问题与报错](#常见问题与报错)
- [相关文档](#相关文档)

---

## 导入

```python
from proscore.transform import WOETransformer
```

---

## 快速开始

```python
from proscore.transform import WOETransformer

# 假设 bin_tables 来自 Binning.bin_table_
wt = WOETransformer(unseen_strategy="worst")
wt.fit(binning.bin_table_)         # 接收 Binning 的分箱结果

df_woe = wt.transform(df)          # 原始值 → WOE
# 或一步完成
df_woe = wt.fit_transform(binning.bin_table_, df)
```

---

## API：`WOETransformer`

```python
WOETransformer(unseen_strategy="worst")
```

### 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `unseen_strategy` | `str` | `"worst"` | 分类变量未见值的 WOE 取值策略 |

---

### unseen 策略

控制 test/OOT 中出现 train 中未见过的分类值时，应赋予什么 WOE：

| 值 | 含义 | 适用场景 |
|----|------|---------|
| `"worst"` | 取坏账率最高箱的 WOE | 风控保守——新类别视为高风险 |
| `"most_common"` | 取样本最多箱的 WOE | 用大众类别作为代理 |
| `"missing"` | 取缺失箱的 WOE（若无则 0） | 将未知等同于缺失 |
| `"zero"` | WOE = 0 | 中性——无信息不偏倚 |

### 方法

| 方法 | 说明 |
|------|------|
| `fit(bin_tables)` | 从 `Binning.bin_table_` 构建 WOE 映射 |
| `transform(X)` | 将 `X` 每列转为 WOE 值 |
| `fit_transform(bin_tables, X)` | fit + transform |

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `bin_tables_` | `dict[str, BinTable]` | 原始分箱表引用 |
| `woe_map_` | `dict[str, dict]` | `{列名: {值/区间: WOE}}` |

### 转换规则

| 变量类型 | 转换方式 |
|---------|---------|
| 连续变量 | `pd.cut` → 箱号 → WOE |
| 分类变量 | `cat_mapping` 查值 → WOE |
| 特殊值 | 特殊值箱的 WOE |
| 缺失值 | 缺失箱的 WOE |
| 未见类别 | 根据 `unseen_strategy` 决定 |

---

## 常见问题与报错

| 现象 | 可能原因 | 处理方向 |
|------|----------|----------|
| `RuntimeError: Call fit() before...` | 未调用 `fit()` | 先 fit 再 transform |
| `ValueError: Unknown unseen_strategy` | 传入非法策略名 | 使用 `"worst"` / `"most_common"` / `"missing"` / `"zero"` |
| 某列在 transform 后全部 NaN | 该列不在 bin_tables 中 | 检查 bin_tables 的 keys |

---

## 相关文档

- 手册总览：[`index.md`](index.md)
- 分箱模块：[`binning.md`](binning.md)
- 建模模块：[`modeling.md`](modeling.md)
