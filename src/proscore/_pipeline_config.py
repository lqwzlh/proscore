"""Excel-driven pipeline configuration and execution.

Read a 7-sheet Excel workbook and produce a ProScore chain, then execute the
full modelling pipeline and save the report / monitor baseline.

Usage (via CLI)::

    proscore run pipeline.xlsx
    proscore run pipeline.xlsx --output-script my_model.py
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── constants ────────────────────────────────────────────────────────────────

_DEFAULT_GLOBAL = {
    "project_name": "scorecard",
    "modeler": "",
    "purpose": "",
    "random_seed": 42,
}

_DEFAULT_DATA = {
    "data_file": None,
    "target": None,
    "time_col": None,
    "id_col": None,
    "dev_start": None,
    "dev_end": None,
    "train_ratio": 0.7,
    "oot_start": None,
    "oot_end": None,
}

_DEFAULT_STEPS = {
    "detect": "on",
    "quality": "on",
    "prefilter": "on",
    "refine": "on",
    "select": "on",
    "evaluate": "on",
    "report": "on",
    "monitor": "off",
}

# (param, default, choices, type, low, high, description)
_PARAM_SPEC = {
    # ── binning ──────────────────────────────────────────────────────────────
    "method": ("chi", ["chi", "tree", "distance", "frequency"],
               "str", "卡方合并（推荐）/ 决策树 / 等距 / 等频"),
    "n_bins": (5, None, "int", 3, 10,
               "目标分箱数，实际受数据约束可能略少"),
    "min_bin_pct": (0.05, None, "float", 0.01, 0.20,
                    "单箱最小样本占比"),
    "adjust_shape": ("on", ["on", "off"], "str",
                     "是否自动调整分箱趋势（推荐 on）"),
    "missing_combine": ("none", ["none", "near", "worst"], "str",
                        "缺失箱合并策略：none=不合并 / near=合并到坏账率最接近的箱 / worst=合并到坏账率最高的箱"),

    # ── screening ────────────────────────────────────────────────────────────
    "max_missing_rate": (0.8, None, "float", 0.1, 0.95,
                         "缺失率上限", "prefilter"),
    "max_one_value_rate": (0.95, None, "float", 0.8, 0.99,
                           "单值率上限", "prefilter"),
    "iv_low": (0.02, None, "float", 0.0, 0.15,
               "IV 下限，低于此值丢弃", "refine"),
    "iv_high": (None, None, "float_or_none", 0.0, 100.0,
                "IV 上限，空=不限制", "refine"),
    "max_psi": (None, None, "float_or_none", 0.05, 0.25,
                "PSI 上限（需 test 数据），空=跳过", "refine"),
    "max_corr": (0.8, None, "float", 0.6, 0.95,
                 "相关系数上限", "refine"),
    "max_vif": (10, None, "float_or_none", 3, 10,
                "VIF 上限，空=跳过", "refine"),
    "min_auc": (None, None, "float_or_none", 0.50, 0.70,
                "单变量 AUC 下限，空=跳过", "refine"),

    # ── modeling (stepwise + scorecard) ──────────────────────────────────────
    "n_min": (5, None, "int", 2, 20,
              "最少入模变量数"),
    "n_max": (12, None, "int", 3, 30,
              "最多入模变量数（须 ≥ n_min）"),
    "pvalue_threshold": (0.05, None, "float", 0.01, 0.20,
                         "逐步回归 P 值阈值"),
    "coef_sign": ("positive", ["positive", "negative", ""],
                  "str", "系数符号约束。positive=所有变量系数>0，保证WOE方向与风险一致（推荐）。negative=所有<0。空=不限"),
    "force_fill": ("on", ["on", "off"], "str",
                   "变量不足 n_min 时是否强制补齐"),
    "perturbation": ("on", ["on", "off"], "str",
                     "是否启用扰动搜索"),
    "odds": (20, None, "int", 10, 100,
             "基准好坏比（1:20 ≈ 坏账率 4.8%）"),
    "pdo": (20, None, "int", 10, 50,
            "odds 翻倍时增加的分数"),
    "base_score": (600, None, "int", 400, 800,
                   "基准 odds 对应的分数"),
}


# ── error / result types ─────────────────────────────────────────────────────


@dataclass
class ValidationError:
    sheet: str
    param: str
    message: str


@dataclass
class PipelineConfig:
    """Parsed and validated pipeline configuration."""

    global_cfg: dict[str, Any] = field(default_factory=dict)
    data_cfg: dict[str, Any] = field(default_factory=dict)
    steps: dict[str, bool] = field(default_factory=dict)
    binning_cfg: dict[str, Any] = field(default_factory=dict)
    screening_cfg: dict[str, Any] = field(default_factory=dict)
    modeling_cfg: dict[str, Any] = field(default_factory=dict)
    variable_presets: dict[str, Any] | None = None
    presets_path: str = ""
    _errors: list[ValidationError] = field(default_factory=list)

    # ── factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_excel(cls, path: str) -> PipelineConfig:
        """Parse a pipeline Excel workbook.

        Raises ``ValueError`` if validation fails.
        """
        import openpyxl  # noqa: F401

        cfg = cls()
        cfg.presets_path = path

        try:
            sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        except Exception as e:
            raise ValueError(
                f"无法读取 {path!r}。请确认文件存在且为 .xlsx 格式。\n原始错误: {e}"
            ) from e

        cfg._parse_global(sheets.get("Global"))
        cfg._parse_data(sheets.get("Data"))
        cfg._parse_steps(sheets.get("Steps"))
        cfg._parse_params(sheets.get("Binning"), "binning")
        cfg._parse_params(sheets.get("Screening"), "screening")
        cfg._parse_params(sheets.get("Modeling"), "modeling")
        cfg._parse_variables(sheets.get("Variables"))

        if cfg._errors:
            msg = "\n".join(
                f"❌ [{e.sheet}] {e.param}: {e.message}" for e in cfg._errors
            )
            raise ValueError(f"配置校验失败 ({len(cfg._errors)} 项):\n{msg}")

        cfg._cross_validate()
        return cfg

    # ── sheet parsers ────────────────────────────────────────────────────────

    def _parse_global(self, df: pd.DataFrame | None) -> None:
        self.global_cfg = dict(_DEFAULT_GLOBAL)
        if df is None:
            return
        for _, row in df.iterrows():
            key = str(row.get("参数名", "")).strip()
            if key in self.global_cfg:
                val = _cell(row, "您的取值", _DEFAULT_GLOBAL[key])
                self.global_cfg[key] = val

    def _parse_data(self, df: pd.DataFrame | None) -> None:
        self.data_cfg = dict(_DEFAULT_DATA)
        if df is None:
            self._errors.append(ValidationError(
                "Data", "(Sheet)", "Data Sheet 不存在，无法继续。"))
            return
        for _, row in df.iterrows():
            key = str(row.get("参数名", "")).strip()
            if key in self.data_cfg:
                val = _cell(row, "您的取值", _DEFAULT_DATA[key])
                self.data_cfg[key] = val

        # Validate data_file exists (non-blocking — defer to execute for full check)
        fpath = self.data_cfg.get("data_file")
        if fpath and not os.path.exists(str(fpath)):
            self._errors.append(ValidationError(
                "Data", "data_file",
                f"文件 {fpath!r} 不存在。请确认路径。"))

        # Numeric ranges
        self._check_numeric("Data", self.data_cfg, "train_ratio", 0.5, 0.9)

        # Validate date fields
        for date_field in ("dev_start", "dev_end", "oot_start", "oot_end"):
            val = self.data_cfg.get(date_field)
            if val is None or (isinstance(val, str) and val.strip() == ""):
                continue
            try:
                pd.to_datetime(str(val))
            except Exception:
                self._errors.append(ValidationError(
                    "Data", date_field,
                    f"无法解析日期 {val!r}。请使用 2021 或 2021-01-01 格式。"))

        # Validate time_col against actual data
        self._validate_time_col(fpath)

    def _validate_time_col(self, fpath: str | None) -> None:
        time_col = self.data_cfg.get("time_col")
        if not time_col or not fpath or not os.path.exists(str(fpath)):
            return
        try:
            df_sample = pd.read_csv(str(fpath), parse_dates=[time_col], nrows=100)
            if time_col not in df_sample.columns:
                cols = list(df_sample.columns)[:15]
                self._errors.append(ValidationError(
                    "Data", "time_col",
                    f"列 {time_col!r} 在数据中不存在。"
                    f"可用列: {', '.join(cols)}..."))
            else:
                if not pd.api.types.is_datetime64_any_dtype(df_sample[time_col]):
                    try:
                        pd.to_datetime(df_sample[time_col])
                    except Exception:
                        self._errors.append(ValidationError(
                            "Data", "time_col",
                            f"列 {time_col!r} 无法解析为日期。"
                            f"请确认格式（如 2021-01-01 或 2021）。"))
        except Exception:
            pass

    def _parse_steps(self, df: pd.DataFrame | None) -> None:
        self.steps = {}
        for key, default in _DEFAULT_STEPS.items():
            self.steps[key] = default == "on"
        if df is None:
            return
        for _, row in df.iterrows():
            key = str(row.get("参数名", "")).strip()
            if key in _DEFAULT_STEPS:
                val = str(_cell(row, "您的取值", _DEFAULT_STEPS[key])).strip().lower()
                if val not in ("on", "off"):
                    self._errors.append(ValidationError(
                        "Steps", key, f"取值 {val!r} 无效，仅接受 on / off。"))
                else:
                    self.steps[key] = val == "on"

        # Dependency checks
        if self.steps.get("refine") and not self.steps.get("prefilter"):
            self._errors.append(ValidationError(
                "Steps", "refine",
                "refine=on 需要 prefilter=on 先运行。请打开 prefilter 或关闭 refine。"))
        if self.steps.get("select") and not self.steps.get("refine"):
            self._errors.append(ValidationError(
                "Steps", "select",
                "select=on 需要 refine=on 提供候选变量。请打开 refine 或关闭 select。"))

    def _parse_params(self, df: pd.DataFrame | None, section: str) -> None:
        target = {"binning": self.binning_cfg, "screening": self.screening_cfg,
                  "modeling": self.modeling_cfg}[section]

        # Fill defaults
        for key, spec in _PARAM_SPEC.items():
            stage = spec[5] if len(spec) > 5 else None
            if section == "binning" and (stage is None):
                target[key] = spec[0]
            elif section == "screening" and stage is not None:
                target[key] = spec[0]
            elif section == "modeling" and (stage is None):
                if key in ("n_min", "n_max", "pvalue_threshold", "coef_sign",
                           "force_fill", "perturbation", "odds", "pdo", "base_score"):
                    target[key] = spec[0]

        if df is None:
            return

        for _, row in df.iterrows():
            key = str(row.get("参数名", "")).strip()
            if key not in _PARAM_SPEC:
                continue

            spec = _PARAM_SPEC[key]
            stage = spec[5] if len(spec) > 5 else None

            # Only apply if this param belongs to this section
            if section == "binning" and stage is not None:
                continue
            if section == "screening" and stage is None:
                continue
            if section == "modeling":
                valid = ("n_min", "n_max", "pvalue_threshold", "coef_sign",
                         "force_fill", "perturbation", "odds", "pdo", "base_score")
                if key not in valid:
                    continue

            default_val = spec[0]
            raw = _cell(row, "您的取值", default_val)
            parsed = self._validate_param(section, key, raw, spec)
            if parsed is not None:
                target[key] = parsed

    def _validate_param(
        self, sheet: str, key: str, raw: Any, spec: tuple
    ) -> Any:
        """Validate a single parameter value. Returns parsed value or defaults."""
        ptype = spec[2]
        low = spec[3] if len(spec) > 3 else None
        high = spec[4] if len(spec) > 4 else None
        choices = spec[1]

        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return spec[0]  # use default

        try:
            if ptype in ("int", "float", "float_or_none"):
                val = float(raw)
                if ptype == "int":
                    val = int(val)
                if low is not None and val < low:
                    self._errors.append(ValidationError(
                        sheet, key,
                        f"取值 {val} 小于下限 {low}。已使用默认值 {spec[0]}。"))
                    return spec[0]
                if high is not None and val > high:
                    self._errors.append(ValidationError(
                        sheet, key,
                        f"取值 {val} 大于上限 {high}。已使用默认值 {spec[0]}。"))
                    return spec[0]
                return val
            elif choices is not None:
                val = str(raw).strip()
                if val not in choices:
                    self._errors.append(ValidationError(
                        sheet, key,
                        f"取值 {val!r} 无效，可选: {', '.join(str(c) for c in choices)}。"
                        f"已使用默认值 {spec[0]}。"))
                    return spec[0]
                return val
            else:
                return str(raw).strip()
        except (ValueError, TypeError):
            self._errors.append(ValidationError(
                sheet, key, f"取值 {raw!r} 无法解析为 {ptype}。已使用默认值 {spec[0]}。"))
            return spec[0]

    def _parse_variables(self, df: pd.DataFrame | None) -> None:
        if df is None:
            return
        # Normalize column names: strip leading/trailing whitespace,
        # drop trailing " *" (used in template headers as footnote markers).
        _norm = {c: c.strip().removesuffix(" *").strip() for c in df.columns}
        _renamed = df.rename(columns=_norm)

        # Check for "variable" column
        if "variable" not in _renamed.columns:
            self._errors.append(ValidationError(
                "Variables", "(Sheet)",
                "缺少 'variable' 列。请参考模板填写。"))
            return

        presets: dict[str, dict[str, object]] = {}
        for _, row in _renamed.iterrows():
            var = row.get("variable")
            if pd.isna(var):
                continue
            var = str(var).strip()
            # Skip instruction / comment rows (contain Chinese or are not
            # valid ASCII variable names).
            if not var.isascii():
                continue
            cfg: dict[str, object] = {}
            for col in ("monotonic", "dimension", "forced_in", "special_values"):
                if col in _renamed.columns:
                    v = row.get(col)
                    if pd.notna(v):
                        cfg[col] = str(v).strip()
            if cfg:
                presets[var] = cfg

        if presets:
            from proscore.utils._presets import _ALLOWED_MONOTONIC

            for var, cfg in presets.items():
                mono = cfg.get("monotonic", "")
                if isinstance(mono, str) and mono and mono not in _ALLOWED_MONOTONIC:
                    self._errors.append(ValidationError(
                        "Variables", var,
                        f"monotonic={mono!r} 无效。可选: "
                        f"{sorted(_ALLOWED_MONOTONIC)}。"))

            self.variable_presets = presets

    # ── cross-validation ─────────────────────────────────────────────────────

    def _cross_validate(self) -> None:
        """Cross-parameter validation."""
        # n_max >= n_min
        n_min = self.modeling_cfg.get("n_min", 5)
        n_max = self.modeling_cfg.get("n_max", 12)
        if n_max < n_min:
            self._errors.append(ValidationError(
                "Modeling", "n_max",
                f"n_max={n_max} 小于 n_min={n_min}。请增大 n_max 或减小 n_min。"))

        # iv_high > iv_low
        iv_low = self.screening_cfg.get("iv_low", 0.02)
        iv_high = self.screening_cfg.get("iv_high")
        if iv_high is not None and iv_high <= iv_low:
            self._errors.append(ValidationError(
                "Screening", "iv_high",
                f"iv_high={iv_high} 必须大于 iv_low={iv_low}。"))

        # variable existence check
        self._validate_variables_exist()

        if self._errors:
            msg = "\n".join(
                f"❌ [{e.sheet}] {e.param}: {e.message}" for e in self._errors
            )
            raise ValueError(f"交叉校验失败 ({len(self._errors)} 项):\n{msg}")

    def _validate_variables_exist(self) -> None:
        """Check that variables in presets exist in the data file."""
        if not self.variable_presets:
            return
        fpath = self.data_cfg.get("data_file")
        if not fpath or not os.path.exists(str(fpath)):
            return  # deferred to execute()
        try:
            sample = pd.read_csv(str(fpath), nrows=5)
            data_cols = set(sample.columns)
        except Exception:
            return

        for var in self.variable_presets:
            if var not in data_cols:
                suspects = [c for c in data_cols if c.lower() == var.lower()]
                hint = f" 数据中存在相似列: {suspects[0]!r}" if suspects else ""
                self._errors.append(ValidationError(
                    "Variables", var,
                    f"变量 {var!r} 在数据文件中不存在。{hint}"))

    # ── variable preset helpers ──────────────────────────────────────────────

    def _build_binning_feature_config(self) -> dict:
        """Extract per-variable binning config from variable presets."""
        if not self.variable_presets:
            return {}
        feature_config = {}
        for var, cfg in self.variable_presets.items():
            vcfg = {}
            if cfg.get("monotonic"):
                vcfg["monotonic"] = cfg["monotonic"]
            if cfg.get("special_values"):
                parts = [x.strip() for x in cfg["special_values"].split(",") if x.strip()]
                if parts:
                    parsed = []
                    for x in parts:
                        num = pd.to_numeric(x, errors="coerce")
                        parsed.append(num if not (isinstance(num, float) and np.isnan(num)) else x)
                    vcfg["special_values"] = parsed
            if vcfg:
                feature_config[var] = vcfg
        return feature_config

    def _build_feature_belong(self) -> dict:
        """Extract dimension -> feature_belong from variable presets."""
        if not self.variable_presets:
            return {}
        belong = {}
        for var, cfg in self.variable_presets.items():
            dim = cfg.get("dimension")
            if dim:
                belong.setdefault(dim, []).append(var)
        return belong

    def _build_force_in(self) -> list[str]:
        """Extract forced_in variables from variable presets."""
        if not self.variable_presets:
            return []
        return [var for var, cfg in self.variable_presets.items() if cfg.get("forced_in") == "on"]

    # ── methods that need ProScore import (lazy) ─────────────────────────────

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        """Load and split data according to config."""
        np.random.seed(int(self.global_cfg.get("random_seed", 42)))

        fpath = self.data_cfg["data_file"]
        time_col = self.data_cfg.get("time_col")
        target = self.data_cfg["target"]
        id_col = self.data_cfg.get("id_col")
        train_ratio = float(self.data_cfg.get("train_ratio", 0.7))

        # Load
        if str(fpath).endswith((".xlsx", ".xls")):
            df = pd.read_excel(fpath)
        else:
            df = pd.read_csv(fpath)

        # Parse time if provided
        if time_col and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])

        # Drop id column
        if id_col and id_col in df.columns:
            df = df.drop(columns=[id_col])

        # Split
        oot = None
        dev_pool = df

        dev_start_raw = self.data_cfg.get("dev_start")
        dev_end_raw = self.data_cfg.get("dev_end")
        oot_start_raw = self.data_cfg.get("oot_start")
        oot_end_raw = self.data_cfg.get("oot_end")

        if time_col and time_col in df.columns:
            time_series = df[time_col]

            # Filter dev pool
            if dev_start_raw is not None and str(dev_start_raw).strip():
                dev_start = pd.to_datetime(str(dev_start_raw))
                dev_pool = dev_pool[time_series >= dev_start]
                time_series = dev_pool[time_col]

            if dev_end_raw is not None and str(dev_end_raw).strip():
                dev_end = pd.to_datetime(str(dev_end_raw))
                dev_pool = dev_pool[time_series <= dev_end]
                time_series = dev_pool[time_col]

            # Cut OOT
            if oot_start_raw is not None and str(oot_start_raw).strip():
                oot_start = pd.to_datetime(str(oot_start_raw))
                oot_mask = df[time_col] >= oot_start
                if oot_end_raw is not None and str(oot_end_raw).strip():
                    oot_end = pd.to_datetime(str(oot_end_raw))
                    oot_mask &= df[time_col] <= oot_end
                oot = df[oot_mask].drop(columns=[time_col], errors="ignore")

            # Drop time_col from dev_pool
            dev_pool = dev_pool.drop(columns=[time_col], errors="ignore")

        # Random split within dev pool
        n = len(dev_pool)
        idx = np.random.permutation(n)
        split = int(n * train_ratio)
        train = dev_pool.iloc[idx[:split]].reset_index(drop=True)
        test = dev_pool.iloc[idx[split:]].reset_index(drop=True)
        if oot is not None:
            oot = oot.reset_index(drop=True)

        # Warn if data is thin
        if len(train) < 50:
            warnings.warn(
                f"训练集仅 {len(train)} 行，模型可能不稳定。", stacklevel=2)
        if oot is not None and len(oot) < 10:
            warnings.warn(
                f"OOT 仅 {len(oot)} 行，评估指标可能不可靠。", stacklevel=2)
        if len(test) < 10:
            warnings.warn(
                f"测试集仅 {len(test)} 行，评估指标可能不可靠。", stacklevel=2)

        return train, test, oot

    def execute(self) -> dict[str, Any]:
        """Run the full pipeline. Returns ``{ps, report_path, monitor_path}``."""
        import proscore as ps

        # Validate required data params at execution time
        fpath = self.data_cfg.get("data_file")
        target = self.data_cfg.get("target")
        if not fpath:
            raise ValueError(
                "[Data] data_file: 必填参数为空。请在 Excel 中填写数据文件路径。")
        if not target:
            raise ValueError(
                "[Data] target: 必填参数为空。请在 Excel 中填写目标列名。")
        if not os.path.exists(str(fpath)):
            raise ValueError(
                f"[Data] data_file: 文件 {fpath!r} 不存在。请确认路径。")
        try:
            import pandas as pd
            sample = pd.read_csv(str(fpath), nrows=5)
            if target not in sample.columns:
                raise ValueError(
                    f"[Data] target: 列 {target!r} 在数据中不存在。"
                    f"可用列: {', '.join(list(sample.columns)[:15])}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"[Data] 无法读取数据文件 {fpath!r}。原始错误: {e}") from e

        train, test, oot = self._load_data()
        target = str(self.data_cfg["target"])

        p = ps.ProScore()
        p.read(train=train, test=test, oot=oot, target=target)

        # Build kwargs from configs
        prefilter_kw = self._build_prefilter_kw()
        binning_kw = self._build_binning_kw()
        refine_kw = self._build_refine_kw()
        select_kw = self._build_select_kw()
        model_kw = self._build_model_kw()

        # Apply variable presets
        preset_fc = self._build_binning_feature_config()
        feature_belong = self._build_feature_belong()
        force_in = self._build_force_in()

        if preset_fc or feature_belong or force_in:
            # Use BinningProcess for per-variable config
            from proscore.binning import BinningProcess  # noqa: F811
            binning_kw.pop("feature_config", None)
            if preset_fc:
                binning_kw["feature_config"] = preset_fc

        if feature_belong:
            select_kw["feature_belong"] = feature_belong
        if force_in:
            select_kw["force_in"] = force_in

        # Execute pipeline
        if self.steps.get("detect", True):
            p.detect()
        if self.steps.get("quality", True):
            p.quality()

        if self.steps.get("prefilter", True):
            p.prefilter(**prefilter_kw)

        p.bin(**binning_kw)

        if self.steps.get("refine", True):
            p.refine(**refine_kw)

        p.transform()

        if self.steps.get("select", True):
            p.select(**select_kw)

        p.fit(**model_kw)
        p.scorecard()

        if self.steps.get("evaluate", True):
            p.evaluate()

        result: dict[str, Any] = {"ps": p}

        # Report
        project_name = str(self.global_cfg.get("project_name", "scorecard"))
        output_dir = f"{project_name}_report"
        if self.steps.get("report", True):
            try:
                from proscore.report import ReportBuilder

                rb = ReportBuilder.from_proscore(p)
                rb.save(f"{output_dir}/report.md")
                result["report_path"] = f"{output_dir}/report.md"
            except Exception as e:
                warnings.warn(f"报告生成失败: {e}", stacklevel=2)

        # Monitor
        if self.steps.get("monitor", False):
            try:
                from proscore.monitor import ModelMonitor

                monitor = ModelMonitor.from_proscore(p)
                monitor.save(f"{output_dir}/monitor.json")
                result["monitor_path"] = f"{output_dir}/monitor.json"
            except Exception as e:
                warnings.warn(f"监控基线创建失败: {e}", stacklevel=2)

        self._print_summary(p)
        return result

    def _build_prefilter_kw(self) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        for key in ("max_missing_rate", "max_one_value_rate"):
            if key in self.screening_cfg:
                kw[key] = self.screening_cfg[key]
        return kw

    def _build_binning_kw(self) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        cfg = self.binning_cfg
        for key in ("method", "n_bins", "min_bin_pct"):
            if key in cfg:
                kw[key] = cfg[key]
        if cfg.get("adjust_shape") is not None:
            kw["adjust_shape"] = cfg["adjust_shape"] == "on" if isinstance(cfg["adjust_shape"], str) else bool(cfg["adjust_shape"])
        # Excel "none" → Python None
        mc = cfg.get("missing_combine")
        if mc is not None and str(mc).strip().lower() != "none":
            kw["missing_combine"] = str(mc).strip()
        return kw

    def _build_refine_kw(self) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        cfg = self.screening_cfg
        iv_low = cfg.get("iv_low", 0.02)
        iv_high = cfg.get("iv_high")
        kw["iv_range"] = (float(iv_low), float(iv_high) if iv_high is not None else None)
        if cfg.get("max_psi") is not None:
            kw["max_psi"] = float(cfg["max_psi"])
        if cfg.get("max_corr") is not None:
            kw["max_corr"] = float(cfg["max_corr"])
        if cfg.get("max_vif") is not None:
            kw["max_vif"] = float(cfg["max_vif"])
        if cfg.get("min_auc") is not None:
            kw["min_auc"] = float(cfg["min_auc"])
        return kw

    def _build_select_kw(self) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        cfg = self.modeling_cfg
        for key in ("n_min", "n_max", "pvalue_threshold"):
            if key in cfg:
                kw[key] = cfg[key]
        cs = cfg.get("coef_sign", "positive")
        kw["coef_sign"] = cs if cs else None
        for key in ("force_fill", "perturbation"):
            v = cfg.get(key)
            kw[key] = v == "on" if isinstance(v, str) else bool(v)
        return kw

    def _build_model_kw(self) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        cfg = self.modeling_cfg
        for key in ("odds", "pdo", "base_score"):
            if key in cfg:
                kw[key] = cfg[key]
        return kw

    def _print_summary(self, p) -> None:
        """Print a concise summary to stdout."""
        print(f"\n{'='*60}")
        print(f"  {self.global_cfg.get('project_name', 'scorecard')} — 建模完成")
        print(f"{'='*60}")
        try:
            print(f"  入模变量: {' | '.join(p.support_)}")
        except Exception:
            pass
        er = getattr(p, "eval_result", None) or {}
        for label, key in [("Train KS", "trn_ks"), ("Test  KS", "test_ks"),
                           ("OOT   KS", "oot_ks")]:
            if key in er:
                print(f"  {label}: {er[key]:.4f}")
        for label, key in [("Train AUC", "trn_auc"), ("Test  AUC", "test_auc"),
                           ("OOT   AUC", "oot_auc")]:
            if key in er:
                print(f"  {label}: {er[key]:.4f}")
        if "psi" in er:
            print(f"  PSI(train vs test): {er['psi']:.4f}")
        if "psi_oot" in er:
            print(f"  PSI(train vs OOT):  {er['psi_oot']:.4f}")
        print(f"{'='*60}\n")

    # ── code generation ──────────────────────────────────────────────────────

    def emit_script(self, output_path: str) -> None:
        """Write a self-contained Python script equivalent to this config."""
        lines = []
        _w = lines.append

        proj = self.global_cfg.get("project_name", "scorecard")
        seed = self.global_cfg.get("random_seed", 42)
        data_file = self.data_cfg.get("data_file", "")
        target = self.data_cfg.get("target", "")
        time_col = self.data_cfg.get("time_col", "")
        id_col = self.data_cfg.get("id_col", "")
        train_ratio = self.data_cfg.get("train_ratio", 0.7)

        _w(f"# Generated by: proscore run pipeline.xlsx --output-script {output_path}")
        _w(f"# 项目: {proj}")
        _w(f"# 日期: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
        _w("")
        _w("import numpy as np")
        _w("import pandas as pd")
        _w("import proscore as ps")
        _w("")
        _w(f"np.random.seed({seed})")
        _w("")
        _w(f"df = pd.read_csv({data_file!r})")
        if time_col:
            _w(f"df[{time_col!r}] = pd.to_datetime(df[{time_col!r}])")
        if id_col:
            _w(f"df = df.drop(columns=[{id_col!r}])")
        _w("")
        # Split logic
        _w("# ── 数据切分 ──")
        if time_col:
            dev_start = self.data_cfg.get("dev_start")
            dev_end = self.data_cfg.get("dev_end")
            oot_start = self.data_cfg.get("oot_start")
            oot_end = self.data_cfg.get("oot_end")
            _w("dev_pool = df.copy()")
            if dev_start:
                _w(f"dev_pool = dev_pool[dev_pool[{time_col!r}] >= pd.Timestamp({str(dev_start)!r})]")
            if dev_end:
                _w(f"dev_pool = dev_pool[dev_pool[{time_col!r}] <= pd.Timestamp({str(dev_end)!r})]")
            if oot_start:
                _w(f"oot_mask = df[{time_col!r}] >= pd.Timestamp({str(oot_start)!r})")
                if oot_end:
                    _w(f"oot_mask &= df[{time_col!r}] <= pd.Timestamp({str(oot_end)!r})")
                _w("oot = df[oot_mask].drop(columns=[" + repr(time_col) + "]).reset_index(drop=True)")
            _w(f"dev_pool = dev_pool.drop(columns=[{time_col!r}])")
            _w("")
            _w(f"idx = np.random.permutation(len(dev_pool))")
            _w(f"n_train = int(len(dev_pool) * {train_ratio})")
            _w("train = dev_pool.iloc[idx[:n_train]].reset_index(drop=True)")
            _w("test  = dev_pool.iloc[idx[n_train:]].reset_index(drop=True)")
        else:
            _w(f"idx = np.random.permutation(len(df))")
            _w(f"n_train = int(len(df) * {train_ratio})")
            _w("train = df.iloc[idx[:n_train]].reset_index(drop=True)")
            _w("test  = df.iloc[idx[n_train:]].reset_index(drop=True)")
            _w("oot = None")

        _w("")
        _w("# ── 建模流水线 ──")
        _w("p = ps.ProScore()")

        read_args = [f"train=train, test=test, target={target!r}"]
        if time_col and oot_start:
            read_args.append("oot=oot")
        _w("p.read(" + ", ".join(read_args) + ")")

        if self.steps.get("detect", True):
            _w("p.detect()")
        if self.steps.get("quality", True):
            _w("p.quality()")

        # prefilter
        if self.steps.get("prefilter", True):
            pf = self._build_prefilter_kw()
            _w(f"p.prefilter({_fmt_kw(pf)})")

        # binning — use BinningProcess when variable presets exist
        preset_fc = self._build_binning_feature_config()
        bk = self._build_binning_kw()

        if preset_fc:
            _w("# ── 分箱（含变量预设：单调趋势 / 特殊值）──")
            _w("from proscore.binning import BinningProcess")
            _w(f"feature_config = {repr(preset_fc)}")
            _w(f"bp = BinningProcess(")
            _w(f"    feature_config=feature_config,")
            _w(f"    default_method={bk.get('method', 'chi')!r},")
            _w(f"    default_n_bins={bk.get('n_bins', 5)},")
            extra_bk = {k: v for k, v in bk.items() if k not in ("method", "n_bins")}
            if extra_bk:
                _w(f"    {_fmt_kw(extra_bk)},")
            _w(f")")
            _w(f"bp.fit(train, y=target)")
        else:
            _w(f"p.bin({_fmt_kw(bk)})")
        # refine must come after bin (IV/PSI/AUC depend on binning)
        if self.steps.get("refine", True):
            rk = self._build_refine_kw()
            _w(f"p.refine({_fmt_kw(rk)})")
        _w("p.transform()")

        # select
        if self.steps.get("select", True):
            sk = self._build_select_kw()
            feature_belong = self._build_feature_belong()
            force_in = self._build_force_in()
            if feature_belong:
                sk["feature_belong"] = feature_belong
            if force_in:
                sk["force_in"] = force_in
            _w(f"p.select({_fmt_kw(sk)})")

        # fit + scorecard
        mk = self._build_model_kw()
        _w(f"p.fit({_fmt_kw(mk)})")
        _w("p.scorecard()")

        if self.steps.get("evaluate", True):
            _w("p.evaluate()")

        _w("")
        _w("# ── 报告 ──")
        if self.steps.get("report", True):
            _w("from proscore.report import ReportBuilder")
            _w("rb = ReportBuilder.from_proscore(p)")
            _w(f"rb.save(f\"{proj}_report/report.md\")")

        _w("")
        _w("# ── 监控 ──")
        if self.steps.get("monitor", False):
            _w("from proscore.monitor import ModelMonitor")
            _w("monitor = ModelMonitor.from_proscore(p)")
            _w(f"monitor.save(f\"{proj}_report/monitor.json\")")

        _w("")
        _w("print('入模变量:', p.support_)")
        _w("er = p.eval_result or {}")
        _w("for k in ['trn_ks', 'test_ks', 'oot_ks']:")
        _w("    if k in er: print(f'{k}: {er[k]:.4f}')")
        _w("print('完成。')")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"脚本已生成: {output_path}")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _check_numeric(
        self, sheet: str, cfg: dict, key: str, low: float, high: float
    ) -> None:
        val = cfg.get(key)
        if val is None:
            return
        try:
            v = float(val)
            if not (low <= v <= high):
                self._errors.append(ValidationError(
                    sheet, key,
                    f"取值 {v} 不在 [{low}, {high}] 范围内。"
                    f"已使用默认值。"))
                cfg[key] = _DEFAULT_DATA.get(key, low)
        except (ValueError, TypeError):
            self._errors.append(ValidationError(
                sheet, key, f"取值 {val!r} 不是有效数字。"))


# ── module-level helpers ─────────────────────────────────────────────────────


def _cell(row: pd.Series, col: str, default: Any) -> Any:
    """Read a cell value, returning *default* when missing."""
    if col not in row.index:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    if isinstance(val, str) and val.strip() == "":
        return default
    return val


def _fmt_kw(kw: dict[str, Any]) -> str:
    """Format a kwargs dict as a Python call string."""
    parts = []
    for k, v in kw.items():
        if isinstance(v, str):
            parts.append(f"{k}={v!r}")
        elif isinstance(v, bool):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def run_pipeline(config_path: str, *, output_script: str | None = None) -> dict:
    """Entry point: read config, optionally emit script, execute pipeline."""
    cfg = PipelineConfig.from_excel(config_path)

    if output_script:
        cfg.emit_script(output_script)

    return cfg.execute()


# ── template generator ───────────────────────────────────────────────────────


def generate_template(out_dir: str = ".") -> str:
    """Create a blank ``pipeline_template.xlsx`` in *out_dir*.

    Returns the absolute path to the generated file.
    """
    import openpyxl  # noqa: F401

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pipeline_template.xlsx"

    def _str_default(val):
        if val is None:
            return ""
        return str(val)

    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:

        # ── Global ──────────────────────────────────────────────────────────
        pd.DataFrame([
            {"参数名": k, "默认值": _str_default(_DEFAULT_GLOBAL.get(k)), "可选范围": "",
             "中文说明": _GLOBAL_DATA_DESC.get(k, ""),
             "类型": "str", "必填": _GLOBAL_DATA_REQUIRED.get(k, "否"), "您的取值": ""}
            for k in _DEFAULT_GLOBAL
        ]).to_excel(writer, sheet_name="Global", index=False)

        # ── Data ────────────────────────────────────────────────────────────
        pd.DataFrame([
            {"参数名": k, "默认值": _str_default(_DEFAULT_DATA.get(k)), "可选范围": "",
             "中文说明": _GLOBAL_DATA_DESC.get(k, ""),
             "类型": _DATA_TYPES.get(k, "str"),
             "必填": _GLOBAL_DATA_REQUIRED.get(k, "否"), "您的取值": ""}
            for k in _DEFAULT_DATA
        ]).to_excel(writer, sheet_name="Data", index=False)

        # ── Steps ───────────────────────────────────────────────────────────
        pd.DataFrame([
            {"参数名": k, "默认值": v, "可选范围": "on / off",
             "中文说明": _STEP_DESC.get(k, ""),
             "类型": "str", "必填": "否", "您的取值": v}
            for k, v in _DEFAULT_STEPS.items()
        ]).to_excel(writer, sheet_name="Steps", index=False)

        # ── Binning ─────────────────────────────────────────────────────────
        _write_params_sheet(writer, "Binning",
                            ["method", "n_bins", "min_bin_pct", "adjust_shape",
                             "missing_combine"])

        # ── Screening ───────────────────────────────────────────────────────
        _write_params_sheet(writer, "Screening",
                            ["max_missing_rate", "max_one_value_rate",
                             "iv_low", "iv_high", "max_psi", "max_corr",
                             "max_vif", "min_auc"])

        # ── Modeling ────────────────────────────────────────────────────────
        _write_params_sheet(writer, "Modeling",
                            ["n_min", "n_max", "pvalue_threshold", "coef_sign",
                             "force_fill", "perturbation", "odds", "pdo", "base_score"])

        # ── Variables ───────────────────────────────────────────────────────
        var_examples = [
            ["income", "年收入", "还款能力", "decreasing", "-999", ""],
            ["debt_ratio", "负债收入比", "负债水平", "increasing", "", ""],
            ["age", "年龄", "个人信息", "u", "", "on"],
            ["education", "学历", "个人信息", "", "", ""],
            ["", "", "", "", "", ""],
        ]
        var_desc = pd.DataFrame(var_examples, columns=[
            "variable *", "name_cn", "dimension", "monotonic",
            "special_values", "forced_in",
        ])
        # Add a header comment row
        var_desc.loc[-1] = [
            "列名（必须与数据一致）",
            "中文名（报告展示）",
            "业务维度（同维度竞争名额）",
            "increasing / decreasing / u / inverted_u / 空=自动",
            "特殊值逗号分隔：-999, missing / 空=无",
            "on=强制入模 / 空=正常",
        ]
        var_desc.index = var_desc.index + 1
        var_desc.sort_index(inplace=True)
        var_desc.to_excel(writer, sheet_name="Variables", index=False)

    return str(path.resolve())


_STEP_DESC = {
    "detect": "数据探查（关了报告无探查章节）",
    "quality": "IV/AUC/KS 排序（关了报告无排序表）",
    "prefilter": "粗筛：缺失率/单值率（关了全量进分箱）",
    "refine": "精筛：IV/AUC/PSI/VIF/相关性",
    "select": "逐步回归（关了 refine 结果全部入模）",
    "evaluate": "模型评估：KS/AUC/PSI",
    "report": "生成 Markdown 报告",
    "monitor": "建立监控基线（上线后推荐开启）",
}

_GLOBAL_DATA_DESC = {
    "project_name": "项目名称，报告标题用",
    "modeler": "建模负责人/团队，报告署名",
    "purpose": "建模目的，报告页眉",
    "random_seed": "全局随机种子，保证可复现",
    "data_file": "数据文件路径，支持 csv / xlsx",
    "target": "目标列名（0=好 1=坏）",
    "time_col": "时间列名，用于切分 OOT（可选）",
    "id_col": "ID 列名，不参与建模（可选）",
    "dev_start": "开发池起始时间（含），空=最早",
    "dev_end": "开发池截止时间（含），空=最晚",
    "train_ratio": "开发池内 train 占比，剩余=test",
    "oot_start": "OOT 起始时间（含），空=无 OOT",
    "oot_end": "OOT 截止时间（含），空=最晚",
}

_GLOBAL_DATA_REQUIRED = {
    "data_file": "是", "target": "是",
    "project_name": "是", "train_ratio": "是",
}

_DATA_TYPES = {
    "data_file": "str", "target": "str", "time_col": "str", "id_col": "str",
    "dev_start": "str", "dev_end": "str", "train_ratio": "float",
    "oot_start": "str", "oot_end": "str",
}


def _write_params_sheet(writer, sheet_name: str, keys: list[str]) -> None:
    """Write a params sheet with unified columns."""
    rows = []
    for key in keys:
        spec = _PARAM_SPEC[key]
        default = spec[0]
        choices = spec[1]
        ptype = spec[2]
        low = spec[3] if len(spec) > 3 else None
        high = spec[4] if len(spec) > 4 else None
        desc = spec[-1]

        if choices is not None:
            choices_str = " / ".join(str(c) for c in choices)
        elif low is not None and high is not None:
            choices_str = f"{low} – {high}"
        else:
            choices_str = "—"

        rows.append({
            "参数名": key,
            "默认值": str(default),
            "可选范围": choices_str,
            "中文说明": desc,
            "类型": ptype,
            "必填": "否",
            "您的取值": "",
        })

    pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)
