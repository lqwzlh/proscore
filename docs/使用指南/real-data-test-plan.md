# 真实数据全流程测试方案（ProScore）

目标：在**公开信贷数据**上跑通 **分模块 API**、**链式 API**、**Excel 配置** 三条路径，参数以模板默认为主，对比结果是否一致、流程是否可复现。

## 1. 数据与环境

| 项目 | 说明 |
|------|------|
| 推荐数据 | **Give Me Some Credit**（`gmsc_train.csv`，约 15 万行）：无需 Kaggle，一条命令可下载 |
| 可选数据 | **Lending Club**（Zenodo 授信用子集，约 134 万行）：体积大，适合性能压测 |
| 可选数据 | **Home Credit** `application_train.csv`：需 Kaggle API，行列均大 |
| Python 依赖 | 核心：`numpy/pandas/sklearn/statsmodels`；Excel 路径需 `pip install "proscore[excel]"`（openpyxl） |
| 数据目录 | `data/`（已加入 `.gitignore`，不提交仓库） |

下载与预处理：

```bash
python scripts/download_real_data.py gmsc
python scripts/prepare_real_scorecard_data.py --sample 50000
```

输出：`data/processed/real_scorecard.csv`（列：`apply_date`、`bad_flag`、数值特征）。

## 2. 切分约定（Notebook 默认）

- **目标**：`bad_flag`（GMSC 对应 `SeriousDlqin2yrs`；LC 宽表由 `loan_status` 映射）。
- **时间列**：`apply_date` 仍写入 CSV，供 Excel / 监控等使用；**Notebook 内默认不按年份切分**。
- **Train / Test / OOT**：对全量样本做 **随机分层**（`stratify=bad_flag`）：先划出约 30% 为非训练池，再将其对半分为 Test 与 OOT（比例见 `REST_FRAC`、`OOT_OF_REST`）。适合 **Lending Club 单年切片**、GMSC 伪日期等无法做真实 OOT 跨年的场景。
- **多年份真实申请日数据**：可自行改为按 `apply_date` 切开发池与 OOT；Excel 路径可在 **Data** 表填写 `dev_end` / `oot_start` 做时间过滤后再随机分 Train/Test（仍无第三份随机 OOT，除非扩展脚本）。

## 3. 三条路径与验收点

### A. 分模块 API

顺序与 OpenSpec 一致：`detect` → `Filter(prefilter)` → `Binning` → `Filter(refine)` → `WOETransformer` → `StepwiseSelector` → `ScoreCard` → `evaluate`。

**验收**：`evaluate` 含 `trn_ks` / `test_ks` / `oot_ks`（若有 OOT）；`support_` 非空；无未捕获异常。

### B. 链式 API

`ProScore().read(...).detect().prefilter().bin().refine().transform().select().fit().scorecard().evaluate()`。

**验收**：与 A 的入模变量集合、核心指标（KS/AUC）在合理范围内接近（逐步回归有随机性，允许小幅差异）。

### C. Excel 驱动

1. `python -m proscore template <目录>` 生成模板。  
2. 填写 **Data**：`data_file`（**绝对路径**）、`target`、`time_col`；**随机切分时**将 `dev_start` / `dev_end` / `oot_start` / `oot_end` **留空**（全量读入后去掉时间列，再按 `train_ratio` 随机分 Train/Test；无第三份 OOT）。若要做**时间 OOT**，再填写 `dev_end` / `oot_start` 等。  
3. 可选 **Global**：`project_name`，避免覆盖默认 `scorecard_report`。  
4. `python -m proscore run filled.xlsx`。

**注意**：模板 **Variables** 表含演示变量名（如 `income`）；若你的 CSV 无这些列，请清空该表（仅保留表头）或只填数据中存在的变量，否则会触发「变量在数据文件中不存在」校验错误。

**验收**：生成 `{project_name}_report/report.md`；控制台无校验错误。

## 4. 性能与资源（建议）

| 样本量 | 用途 |
|--------|------|
| 2 万～5 万 | Notebook 冒烟、CI 可选 |
| 15 万（GMSC 全量） | 日常真实测试 |
| 百万级（LC） | 压测；建议先 `--sample` 或仅数值列子集 |

## 5. Notebook 入口

执行全流程与对比：`notebooks/ProScore真实数据全流程测试.ipynb`。

### Lending Club 本地宽表（`loan_status` + `issue_d`）

使用 Zenodo 子集以外的 LC 导出（例如 Kaggle / 本地 `lendingclub_data2018.csv`）时：

```bash
python scripts/prepare_real_scorecard_data.py --input "/绝对路径/lendingclub_data2018.csv"
```

- 自动识别为 **lending_club_wide**：目标 `bad_flag` 由 `loan_status` 映射；`apply_date` 由 `issue_d` 解析（如 `18-Jun` → 2018-06）。
- **仅保留** `Fully Paid` 与逾期/坏账相关状态；`Current`、`Issued` 等未终态样本会丢弃，故**切片年份若多为在贷**，剩余行数可能较少。
- 剔除列名中含 `total_pymnt`、`out_prncp`、`recoveries` 等子串的字段，避免用还款后信息建模。
- **Excel 引擎**（`PipelineConfig`）：`prefilter` 与 `refine` 共用 Screening 表中的 `max_corr` / `max_vif`；粗筛阶段 **`iv_range` / `max_psi` 固定为 None**（IV/PSI 仅在 `refine` 用分箱结果计算）。Train/Test 在去掉 `time_col` 后对 `target` **尽量分层**随机划分（与纯 `np.random.permutation` 不同）。**Modeling** 表含 `max_iter_round`（默认 100，可与链式 `select(max_iter_round=…)` 对齐）。
