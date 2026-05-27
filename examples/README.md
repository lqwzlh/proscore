# Excel Pipeline 示例模板

本目录提供一份可直接打开的空白配置表，与 `proscore template` 生成的文件一致。

| 文件 | 说明 |
|------|------|
| `pipeline_template.xlsx` | 8 个 Sheet（Global / Data / Steps / Binning / Screening / Modeling / Rules / Variables） |

## 使用方式

**方式 A：复制本目录模板**

```bash
cp examples/pipeline_template.xlsx ./my_project/
# 在 Excel 中填写 data_file、target、time_col 等
proscore run my_project/pipeline_template.xlsx
```

**方式 B：自行生成（与仓库内文件等价）**

```bash
proscore template ./my_project/
proscore run my_project/pipeline_template.xlsx
```

参数说明见 [docs/使用指南/pipeline-config.md](../docs/使用指南/pipeline-config.md)。

## 维护说明

升级模板结构后，在仓库根目录执行：

```bash
proscore template examples
```

然后提交 `examples/pipeline_template.xlsx` 的变更。
