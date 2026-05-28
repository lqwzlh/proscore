# ProScore · Excel 配置驱动（零代码建模）

将建模参数填入 Excel，一行命令跑通全流程。**不需要写任何 Python 代码。**

---

## 目录

- [快速体验](#快速体验)
- [CLI 命令](#cli-命令)
- [模板详解](#模板详解)
  - [Sheet 1: Global — 项目信息](#sheet-1-global--项目信息)
  - [Sheet 2: Data — 数据与切分](#sheet-2-data--数据与切分)
  - [Sheet 3: Steps — 流水线开关](#sheet-3-steps--流水线开关)
  - [Sheet 4: Binning — 分箱参数](#sheet-4-binning--分箱参数)
  - [Sheet 5: Screening — 变量筛选](#sheet-5-screening--变量筛选)
  - [Sheet 6: Modeling — 逐步回归与评分卡](#sheet-6-modeling--逐步回归与评分卡)
  - [Sheet 7: Rules — 规则挖掘（可选）](#sheet-7-rules--规则挖掘可选)
  - [Sheet 8: Variables — 逐变量预设（可选）](#sheet-8-variables--逐变量预设可选)
- [参数校验规则](#参数校验规则)
- [输出等效代码](#输出等效代码)
- [常见问题](#常见问题)

---

## 快速体验

```bash
# 0. 安装（含 Excel 支持）
pip install proscore[excel]

# 1. 获取空白模板（克隆仓库后可直接复制，或自行生成）
cp examples/pipeline_template.xlsx ./my_project/
# 或: proscore template ./my_project/

# 2. 打开 my_project/pipeline_template.xlsx
#    只填 Data sheet 里这 5 个格子：
#      data_file = tests/test_data.csv
#      target    = bad_flag
#      time_col  = apply_date
#      dev_end   = 2021
#      oot_start = 2022

# 3. 运行
proscore run my_project/pipeline_template.xlsx
```

运行后在当前目录生成 `scorecard_report/report.md`（含图表）。若 Steps 中 `mine_rules=on` 且 `export_csv=on`，另生成 `scorecard_report/rules_table.csv`。

---

## CLI 命令

### `proscore template [目录]`

生成空白配置模板到指定目录（默认当前目录）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `out_dir` | `.` | 输出目录，自动创建 |

### `proscore run [配置] [--output-script]`

读取 Excel 配置，执行完整建模流水线。

| 参数 | 必填 | 说明 |
|------|------|------|
| `config` | 是 | pipeline 配置 `.xlsx` 文件路径 |
| `--output-script` / `-o` | 否 | 同时生成等效 Python 脚本 |

---

## 模板详解

模板包含 8 个 Sheet，所有 Sheet 均采用统一列布局：

| 列名 | 含义 |
|------|------|
| `参数名` | 参数标识（不可修改） |
| `默认值` | 系统默认值（留空时使用） |
| `可选范围` | 可填的取值范围或枚举 |
| `中文说明` | 参数的业务含义解释 |
| `类型` | 参数数据类型 |
| `必填` | 是否必须填写 |
| `您的取值` | **用户在此列填写**（空 = 使用默认值） |

---

### Sheet 1: Global — 项目信息

| 参数名 | 默认值 | 说明 | 必填 |
|--------|--------|------|------|
| `project_name` | scorecard | 项目名称，用于报告标题和输出目录名 | 是 |
| `modeler` | （空） | 建模负责人/团队，写入报告署名 | 否 |
| `purpose` | （空） | 建模目的，写入报告页眉 | 否 |
| `random_seed` | 42 | 随机种子，保证结果可复现 | 否 |

---

### Sheet 2: Data — 数据与切分

**核心概念**：开发池（dev pool）内随机切分 train/test，OOT 独立于开发池之外。

```
有 time_col 且设了 oot_start：
┌───────── 开发池 ─────────┐        ┌── OOT ──┐
├─train(70%)─┤├─test(30%)─┤        ├───OOT───┤
^dev_start                ^dev_end ^oot_start ^oot_end

无 time_col：
全量数据 → 随机切 train/test，无 OOT
```

| 参数名 | 默认值 | 可选范围 | 说明 | 必填 |
|--------|--------|---------|------|------|
| `data_file` | — | 文件路径 | 数据文件，支持 csv / xlsx | **是** |
| `target` | — | 列名 | 目标列（0=好 1=坏） | **是** |
| `time_col` | — | 列名 | 日期列，用于切分 OOT（可不填） | 否 |
| `id_col` | — | 列名 | ID 列，不参与建模 | 否 |
| `dev_start` | （最早） | 年份/日期 | 开发池起始时间（含），空=最早一条 | 否 |
| `dev_end` | （最晚） | 年份/日期 | 开发池截止时间（含），空=最晚一条 | 否 |
| `train_ratio` | 0.7 | 0.5–0.9 | 开发池内 train 占比，剩余 = test | 否 |
| `oot_start` | — | 年份/日期 | OOT 起始时间（含），空=无 OOT | 否 |
| `oot_end` | （最晚） | 年份/日期 | OOT 截止时间（含），空=最晚一条 | 否 |

> **最小填写**：`data_file` + `target` + `time_col` + `dev_end` + `oot_start`（5 格）。不填 `time_col` 则不切 OOT，全量随机分 train/test。

---

### Sheet 3: Steps — 流水线开关

控制哪些步骤执行。"关闭"意味着该步骤跳过，不影响模型训练但可能减少输出的信息。

| 步骤 | 默认 | 关了会怎样 | 依赖 |
|------|------|-----------|------|
| `detect` | on | 报告无数据探查章节 | — |
| `quality` | on | 报告无 IV/AUC/KS 排序表 | — |
| `prefilter` | on | 所有数字变量进分箱，不筛缺失/单值率 | — |
| `refine` | on | 预筛结果直接进选择，不做 IV/PSI/VIF 精筛 | prefilter |
| `mine_rules` | **off** | 不挖掘决策规则；开启后输出 `rules_table.csv` | refine |
| `select` | on | 不做逐步回归，refine 结果全部入模 | refine |
| `evaluate` | on | 不输出 KS/AUC/PSI | fit |
| `report` | on | 不生成 Markdown 报告 | — |
| `monitor` | off | 不建立监控基线 | fit |

> **不可关闭的步骤**：`binning`、`fit`、`scorecard` — 这是建模的核心，不提供开关。
>
> **依赖校验**：refine=on 但 prefilter=off → 报错。select=on 但 refine=off → 报错。mine_rules=on 但 refine=off → 报错。

---

### Sheet 4: Binning — 分箱参数

| 参数名 | 默认值 | 可选范围 | 说明 |
|--------|--------|---------|------|
| `method` | chi | chi / tree / distance / frequency | 分箱算法。chi=卡方合并（推荐） |
| `n_bins` | 5 | 3–10 | 目标分箱数 |
| `min_bin_pct` | 0.05 | 0.01–0.20 | 单箱最小样本占比 |
| `adjust_shape` | on | on / off | 是否自动调整趋势（推荐 on） |
| `missing_combine` | none | none / near / worst | 缺失箱合并策略。none=不合并；near=合并到坏账率最接近的箱；worst=合并到坏账率最高的箱 |

> 逐变量的 `monotonic` 和 `special_values` 在 Variables Sheet 中设置。

---

### Sheet 5: Screening — 变量筛选

分为两个阶段，粗筛在分箱前、精筛在分箱后。参数放在同一 Sheet 中，标注适用阶段。

| 参数名 | 默认值 | 可选范围 | 说明 | 阶段 |
|--------|--------|---------|------|------|
| `max_missing_rate` | 0.8 | 0.1–0.95 | 缺失率上限，超出丢弃 | 粗筛 |
| `max_one_value_rate` | 0.95 | 0.8–0.99 | 单值率上限，超出丢弃 | 粗筛 |
| `iv_low` | 0.02 | 0.0–0.15 | IV 下限，低于此值丢弃 | 精筛 |
| `iv_high` | （空） | 空或 > iv_low | IV 上限，空=不设上限 | 精筛 |
| `max_psi` | （空） | 空或 0.05–0.25 | PSI 上限（需 test 数据），空=跳过 | 精筛 |
| `max_corr` | 0.8 | 0.6–0.95 | 相关系数上限，超阈值保留 IV 高者 | 精筛 |
| `max_vif` | 10 | 空或 3–10 | VIF 上限，空=跳过 | 精筛 |
| `min_auc` | （空） | 空或 0.50–0.70 | 单变量 AUC 下限，空=跳过 | 精筛 |

> **为什么分两阶段？** 粗筛不依赖分箱结果，先快速排除垃圾变量；精筛在分箱后做，用真实分箱 IV/PSI。

---

### Sheet 6: Modeling — 逐步回归与评分卡

| 参数名 | 默认值 | 可选范围 | 说明 |
|--------|--------|---------|------|
| `n_min` | 5 | 2–20 | 最少入模变量数 |
| `n_max` | 12 | 3–30 | 最多入模变量数（须 ≥ n_min） |
| `pvalue_threshold` | 0.05 | 0.01–0.20 | 逐步回归 P 值阈值 |
| `coef_sign` | positive | positive / negative / 空 | 系数符号约束。positive=所有变量系数>0，保证 WOE 方向与风险一致（推荐）。negative=所有<0。空=不限 |
| `force_fill` | on | on / off | 变量不足 n_min 时是否强制补齐 |
| `perturbation` | on | on / off | 是否启用扰动搜索（卡住时随机增减变量） |
| `max_iter_round` | 100 | 2–200 | 逐步回归最大迭代轮数（与链式 `select(max_iter_round=…)` 对应） |
| `odds` | 20 | 10–100 | 基准好坏比（1:20 ≈ 坏账率 4.8%） |
| `pdo` | 20 | 10–50 | odds 翻倍时增加的分数 |
| `base_score` | 600 | 400–800 | 基准 odds 对应的分数 |

> **`coef_sign` 为什么默认 positive？** WOE = ln(坏样本占比/好样本占比)，WOE 越大风险越高。系数为正意味着变量贡献方向与 WOE 方向一致，这是监管评分卡的标准要求。

---

### Sheet 7: Rules — 规则挖掘（可选）

在 **refine 之后、WOE 转换之前** 挖掘策略规则（与链式 API 的 `mine_rules()` 一致）。Steps 中 `mine_rules` 默认 **off**，需要规则时改为 on。

| 参数名 | 默认值 | 可选范围 | 说明 |
|--------|--------|---------|------|
| `method` | exhaustive | exhaustive / tree / apriori | 搜索方法 |
| `max_depth` | 3 | 1–3 | 最多几变量交叉（exhaustive / apriori） |
| `max_tree_depth` | 4 | 2–8 | 决策树深度（tree 模式） |
| `min_lift` | 3.0 | 1.0–10.0 | 最小 Lift（precision / 整体坏账率） |
| `min_hit_rate` | 0.02 | 0.001–0.5 | 最小命中率 |
| `max_hit_rate` | 0.20 | 0.01–0.8 | 最大命中率（避免过度拒绝） |
| `max_rules` | 15 | 1–100 | 最多输出规则条数 |
| `random_state` | 42 | 整数 | tree 模式随机种子 |
| `export_csv` | on | on / off | 是否导出 `{project}_report/rules_table.csv` |

> **参数名列使用裸名**（如 `method`，不是 `rm_method`），与 `RuleMiner` 构造参数一致。  
> **与评分卡互斥**：`select()` 会排除规则中出现的**整变量**（保守策略），详见 [rules.md](rules.md)。  
> **WOE 仍保留**：`transform()` 会为规则变量生成 WOE，仅入模筛选时排除。

---

### Sheet 8: Variables — 逐变量预设（可选）

不填此 Sheet = 全部自动处理。填了的变量覆盖自动行为。

模板已包含 4 条示例数据（income / debt_ratio / age / education）。第一行为列名说明。

| 参数名 | 可选值 | 说明 |
|--------|--------|------|
| `variable` | 数据中列名 | 必须与数据中的列名完全一致 |
| `name_cn` | 中文 | 变量中文名，报告展示用 |
| `dimension` | 业务维度标签 | 同维度变量在逐步回归中竞争入模名额（见下方说明） |
| `monotonic` | increasing / decreasing / u / inverted_u / 空 | 预设 WOE 单调方向。空=自动检测 |
| `special_values` | 逗号分隔，如 `-999, missing` | 需单独成箱的特殊值，支持数字和字符串 |
| `forced_in` | on / 空 | on=强制入模，不因逐步回归被剔除 |

**dimension 示例**：

| variable | dimension |
|----------|-----------|
| income | 还款能力 |
| debt_ratio | 负债水平 |
| utilization | 负债水平 |
| age | 个人信息 |
| education | 个人信息 |

此时 `负债水平` 和 `个人信息` 维度各含 2 个变量。若设 `belong_max_pct=0.5`，逐步回归中每个维度最多允许占入模总数的 50%。数据维度多的场景可防止模型被单一维度主导。

---

## 参数校验规则

模板解析时自动执行以下校验，不通过则报错并提示具体问题。

| 校验类型 | 示例 |
|---------|------|
| 类型校验 | `n_bins = "abc"` → 报错，无法解析为整数 |
| 范围校验 | `train_ratio = 0.95` → 报错，超出 0.5–0.9 |
| 白名单校验 | `method = "kmeans"` → 报错，可选 chi / tree / distance / frequency / optimal（需安装 optbinning） |
| 交叉校验 | `n_min=8, n_max=5` → 报错，n_max 须 ≥ n_min |
| 依赖校验 | refine=on 但 prefilter=off → 报错；mine_rules=on 但 refine=off → 报错 |
| 存在性校验 | `data_file` 不存在 → 报错；Variables 中变量在数据中不存在 → 报错并提示相似列名 |
| 日期校验 | `dev_end = "abc"` → 报错，无法解析为日期 |

**降级策略**：校验失败时使用默认值并发出 warning（而非中断），仅在跨参数矛盾时阻断。

---

## 输出等效代码

```bash
proscore run pipeline.xlsx --output-script my_model.py
```

生成的 `my_model.py` 是自包含脚本，内含数据加载、切分、完整链式 API 调用和报告生成。可直接 `python my_model.py` 运行，不依赖 Excel。

---

## 常见问题

**Q: 最少需要填多少个格子？**
A: 无 OOT 时最少填 `data_file`、`target`、`time_col` 3 个；有 OOT 时再补充 `dev_end` / `oot_start` 等时间切分参数。其余全部留空即使用默认值。

**Q: 没有时间列怎么办？**
A: 不填 `time_col`、`oot_start`，数据全量随机切 train/test。无 OOT。

**Q: 如何覆盖某个变量的分箱参数？**
A: 在 Variables Sheet 中填对应行。例如 `income` 行设 `monotonic=decreasing`、`special_values=-999`。

**Q: Excel 填错了会怎样？**
A: 解析阶段给出中文错误提示（哪个 Sheet、哪个参数、原因），不会静默执行。

**Q: 生成的 Python 脚本能脱离 Excel 运行吗？**
A: 能。`--output-script` 生成的脚本是自包含的，不读 Excel。

**Q: `mine_rules=on` 后规则变量会不会进评分卡？**
A: 不会。`select()` 自动排除规则中出现的变量；`transform()` 仍可为这些变量生成 WOE 供分析。

**Q: 规则能直接上线吗？**
A: 不能。Excel 路径同样只在训练集上挖掘，须 OOT 复验后再上线。详见 [rules.md](rules.md)。
