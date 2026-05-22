# ProScore · `_data` 模块使用指南

面向评分卡建模的**数据加载与校验**：DataFrame 接入、CSV 读取、列结构验证。本手册描述用户侧用法。

---

## 目录

- [导入](#导入)
- [快速开始](#快速开始)
- [API：`DataReader`](#apidatareader)
  - [构造参数](#构造参数)
  - [属性](#属性)
  - [方法](#方法)
- [常见问题与报错](#常见问题与报错)
- [相关文档](#相关文档)

---

## 导入

```python
from proscore._data import DataReader
```

---

## 快速开始

```python
import pandas as pd
from proscore._data import DataReader

df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "age": [25, 30, 35, 40, 45],
    "income": [5000, 8000, 12000, 15000, 20000],
    "bad_flag": [0, 0, 1, 0, 1],
})

dr = DataReader(df, target="bad_flag", id_col="id")
print(dr)               # DataReader(n_rows=5, n_features=2, target='bad_flag', id='id')
print(dr.features_)      # ['age', 'income']
print(dr.summary())      # 缺失率、唯一值等概览
```

---

## API：`DataReader`

```python
DataReader(df, target, id_col=None)
```

### 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | `pd.DataFrame` | — | 输入数据 |
| `target` | `str` | — | 目标列名 |
| `id_col` | `str` | `None` | 主键列名（自动排除） |

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `df` | `pd.DataFrame` | 原始数据 |
| `target` | `str` | 目标列名 |
| `id_col` | `str` 或 `None` | 主键列名 |
| `features_` | `list[str]` | 可用特征列（排除 target 和 id） |
| `X` | `pd.DataFrame` | 仅特征列 |
| `y` | `pd.Series` | 目标列 |
| `shape` | `tuple` | `(行数, 特征列数)`，与 `len(features_)` 一致（不含 `target` 与 `id_col`） |

### 方法

| 方法 | 说明 |
|------|------|
| `summary()` | 返回每列 dtype、缺失数、缺失率、唯一值数的 DataFrame |
| `from_csv(path, target, id_col, **kwargs)` | 类方法，从 CSV 文件创建 DataReader，`**kwargs` 透传给 `pd.read_csv` |

---

## 常见问题与报错

| 现象 | 可能原因 | 处理方向 |
|------|----------|----------|
| `KeyError` … not found | `target` 或 `id_col` 不在 DataFrame 列中 | 检查列名拼写 |
| CSV 读取乱码 | 文件编码非 UTF-8 | `DataReader.from_csv(path, target, encoding='gbk')` |

---

## 相关文档

- 手册总览：[`index.md`](index.md)
- 项目约定：仓库根目录 `AGENTS.md`
