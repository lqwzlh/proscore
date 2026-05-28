# 真实数据目录

本目录存放公开信贷数据集（**不提交 Git**，见根目录 `.gitignore`）。

## 下载

```bash
# 推荐：无需 Kaggle，约 7 MB
python scripts/download_real_data.py gmsc

# 可选：Lending Club 授信用子集（Zenodo，~168 MB）
python scripts/download_real_data.py lending_club

# 可选：Home Credit 主表（需 Kaggle API）
python scripts/download_real_data.py home_credit
```

## 预处理为 ProScore 单表

```bash
python scripts/prepare_real_scorecard_data.py
python scripts/prepare_real_scorecard_data.py --sample 50000

# 本地 Lending Club 宽表（含 loan_status、issue_d）
python scripts/prepare_real_scorecard_data.py --input "/path/to/lendingclub_data2018.csv"
```

输出：`data/processed/real_scorecard.csv`（列：`apply_date`、`bad_flag`、数值特征）。

**Lending Club 宽表**：仅保留 `Fully Paid` 与逾期/坏账类状态；剔除 `total_pymnt`、`out_prncp` 等贷后字段；`int_rate` / `revol_util` 会去掉 `%`。若切片里多为 `Current`/`Issued`，过滤后行数会很少——可换多年份导出。

Notebook 默认对全量样本做 **随机分层** Train/Test/OOT，与 `apply_date` 年份无关；说明见 `docs/使用指南/real-data-test-plan.md`。

## 全流程 Notebook

见 `notebooks/ProScore真实数据全流程测试.ipynb`。
