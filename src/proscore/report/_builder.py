"""Markdown report builder for scorecard modelling results."""

from __future__ import annotations

import warnings
from datetime import date as dt_date
from typing import Any

import numpy as np
import pandas as pd


class ReportBuilder:
    """
    Generate a Markdown-formatted scorecard modelling report.

    Accepts data from a :class:`ProScore` chain instance (via
    :meth:`from_proscore`) or individual ``with_*`` methods for partial
    reports, then calls :meth:`build` to produce a Markdown string or
    :meth:`save` to write a ``.md`` file.
    """

    def __init__(
        self,
        title: str = "评分卡建模报告",
        project: str = "",
        modeler: str = "",
        purpose: str = "",
        plot_dir: str = "plots",
    ):
        self.title = title
        self.project = project
        self.modeler = modeler
        self.purpose = purpose
        self.plot_dir = plot_dir
        self._date = dt_date.today().isoformat()

        # Lazy-import viz for plots
        self._plots_module = None
        try:
            from proscore import viz as _v
            self._plots_module = _v
        except ImportError:
            pass

        # ── data slots ──
        self._detect: pd.DataFrame | None = None
        self._quality: pd.DataFrame | None = None
        self._corr: pd.DataFrame | None = None
        self._vif: pd.DataFrame | None = None
        self._target_bad_rate: float | None = None
        self._target_dist: pd.DataFrame | None = None

        self._bin_tables: dict[str, Any] = {}
        self._iv_table: pd.DataFrame | None = None

        self._filter_stages: list[dict[str, Any]] = []
        self._filter_quality: pd.DataFrame | None = None
        self._filter_support: list[str] = []
        self._filter_params: dict[str, Any] = {}
        self._pipeline_halts: list[dict[str, str]] = []

        self._stepwise_record: dict[int, Any] = {}
        self._stepwise_support: list[str] = []
        self._stepwise_best: dict[str, Any] = {}
        self._stepwise_params: dict[str, Any] = {}

        self._scorecard: Any = None  # ScoreCard instance
        self._score_table: pd.DataFrame | None = None

        self._eval_result: dict[str, Any] = {}
        self._diagnosis: Any = None
        self._oot_period_eval: pd.DataFrame | None = None
        self._stability_result: pd.DataFrame | None = None
        self._monitor: Any = None

        # Plot data
        self._plot_ytest: np.ndarray | None = None
        self._plot_probtest: np.ndarray | None = None
        self._plot_scores_trn: np.ndarray | None = None
        self._plot_scores_tst: np.ndarray | None = None

        self._run_summary: dict[str, Any] = {}
        self._report_dir: str = ""
        self._embed_images: bool = False

    # ── factory ────────────────────────────────────────────────────────────

    @classmethod
    def from_proscore(cls, ps, **kwargs) -> ReportBuilder:
        """
        Populate all sections from a fitted :class:`ProScore` chain instance.

        The *ps* object must have completed the full pipeline:
        ``read → detect → filter → bin → transform → select → fit → scorecard → evaluate``.
        """
        rb = cls(**kwargs)
        if ps.detect_result is not None:
            rb.with_inspect(detect=ps.detect_result, quality=ps.quality_result)
        if getattr(ps, "_prefilter", None) is not None:
            rb.with_filter(ps._prefilter, stage="粗筛（prefilter）")
        if ps.filter_ is not None:
            rb.with_filter(ps.filter_, stage="精筛（refine）")
        elif getattr(ps, "_refine_skipped", False):
            rb.with_pipeline_halt(
                "精筛（refine）",
                "无数值变量进入精筛，已跳过（可仅使用类别变量继续）。",
            )
        if getattr(ps, "halted_", False):
            rb.with_pipeline_halt("建模流水线", ps.halt_message_)
        for outcome in getattr(ps, "screen_outcomes_", []):
            if not outcome.ok:
                rb.with_pipeline_halt(outcome.stage, outcome.message)
        if ps.binner_ is not None:
            rb.with_binning(ps.binner_.bin_table_, ps.binner_.iv_)
        if ps.train_df is not None and ps.target:
            rb.with_target_distribution(ps.train_df, ps.target)
        if ps.selector_ is not None:
            rb.with_stepwise(ps.selector_)
        if ps.scorecard_ is not None:
            rb.with_model(ps.scorecard_)
        if ps.eval_result:
            rb.with_evaluate(ps.eval_result)
            # Auto-extract plot data from ProScore when available
            try:
                support = ps.selector_.support_ if ps.selector_ else []
                if support and ps.transformer_ and ps.scorecard_:
                    trn_w = ps.transformer_.transform(ps.train_df[support])
                    trn_score = ps.scorecard_.predict(trn_w).values
                    rb.with_plot_data(train_scores=trn_score)
            except Exception:
                pass
            # Auto-attach diagnosis when available (or best-effort generate from eval_result)
            if getattr(ps, "diagnosis_", None) and ps.diagnosis_.issues:
                rb.with_diagnosis(ps.diagnosis_)
            elif ps.eval_result:
                rb.with_diagnosis()  # will generate a lightweight report from eval only
        return rb

    # ── data input methods ─────────────────────────────────────────────────

    def with_inspect(
        self,
        detect: pd.DataFrame | None = None,
        quality: pd.DataFrame | None = None,
        corr: pd.DataFrame | None = None,
        vif: pd.DataFrame | None = None,
    ) -> ReportBuilder:
        self._detect = detect
        self._quality = quality
        self._corr = corr
        self._vif = vif
        return self

    def with_run_summary(
        self,
        *,
        n_train: int | None = None,
        n_test: int | None = None,
        n_oot: int | None = None,
        n_candidates: int | None = None,
        n_model_pool: int | None = None,
        n_final: int | None = None,
        final_features: list[str] | None = None,
    ) -> ReportBuilder:
        """Sample sizes and feature counts for §0 建模概览."""
        self._run_summary = {
            "n_train": n_train,
            "n_test": n_test,
            "n_oot": n_oot,
            "n_candidates": n_candidates,
            "n_model_pool": n_model_pool,
            "n_final": n_final,
            "final_features": final_features or [],
        }
        return self

    def with_target_distribution(
        self, df: pd.DataFrame, target: str, time_col: str | None = None
    ) -> ReportBuilder:
        self._target_bad_rate = float(df[target].mean())
        if time_col and time_col in df.columns:
            self._target_dist = (
                df.groupby(time_col)[target]
                .agg(n="count", bad_rate="mean")
                .reset_index()
            )
        return self

    def with_binning(
        self, bin_tables: dict, iv_table: pd.DataFrame | None = None
    ) -> ReportBuilder:
        self._bin_tables = bin_tables
        self._iv_table = iv_table
        return self

    def with_pipeline_halt(self, stage: str, message: str) -> ReportBuilder:
        """Record that modelling was skipped at *stage* (normal, not an error)."""
        self._pipeline_halts.append({"stage": stage, "message": message})
        return self

    def with_filter(self, f, *, stage: str = "特征筛选") -> ReportBuilder:
        """Attach a fitted :class:`~proscore.selection.Filter` result.

        Empty ``support_`` after :meth:`~proscore.selection.Filter.fit` is
        allowed — the report will show all candidates as dropped.
        """
        if not hasattr(f, "quality_"):
            raise TypeError(
                "with_filter() expects a Filter instance with quality_"
            )
        qf = f.quality_
        if qf is None:
            qf = pd.DataFrame()
        if "dropped" not in qf.columns and "selected" not in qf.columns:
            raise ValueError(
                f"Filter quality_ must contain 'dropped' or 'selected'; "
                f"got columns {list(qf.columns)}. "
                "Do not pass inspect.quality() output to with_filter()."
            )
        params = {
            "max_missing_rate": getattr(f, "max_missing_rate", ""),
            "max_one_value_rate": getattr(f, "max_one_value_rate", ""),
            "iv_range": getattr(f, "iv_range", ""),
            "max_corr": getattr(f, "max_corr", ""),
            "max_vif": getattr(f, "max_vif", ""),
            "max_psi": getattr(f, "max_psi", ""),
            "n_selected": getattr(f, "n_selected", ""),
        }
        support = list(getattr(f, "support_", []))
        exhausted = getattr(f, "exhausted_", len(support) == 0)
        if exhausted:
            self.with_pipeline_halt(
                stage,
                f"{stage}：{getattr(f, 'n_candidates_in_', len(qf))} 个候选均未保留，"
                "未进入后续建模（属正常筛选结果）。",
            )
        self._filter_stages.append(
            {
                "stage": stage,
                "quality": qf,
                "support": support,
                "params": params,
                "exhausted": exhausted,
            }
        )
        # Legacy single-slot fields (last stage wins)
        self._filter_quality = qf
        self._filter_support = support
        self._filter_params = params
        return self

    def with_stepwise(self, selector) -> ReportBuilder:
        self._stepwise_record = getattr(selector, "record_", {})
        self._stepwise_support = getattr(selector, "support_", [])
        self._stepwise_best = getattr(selector, "best_performance_", {})
        self._stepwise_params = {
            "objective": getattr(selector, "objective", ""),
            "pvalue_threshold": getattr(selector, "pvalue_threshold", ""),
            "coef_sign": getattr(selector, "coef_sign", ""),
            "vif_threshold": getattr(selector, "vif_threshold", ""),
            "corr_threshold": getattr(selector, "corr_threshold", ""),
            "n_min": getattr(selector, "n_min", ""),
            "n_max": getattr(selector, "n_max", ""),
        }
        return self

    def with_model(self, scorecard) -> ReportBuilder:
        self._scorecard = scorecard
        if hasattr(scorecard, "score_table_"):
            self._score_table = scorecard.score_table_
        return self

    def with_evaluate(self, eval_result: dict) -> ReportBuilder:
        self._eval_result = eval_result
        return self

    def with_diagnosis(self, report=None) -> ReportBuilder:
        """Attach a :class:`~proscore.evaluate.DiagnosisReport`.

        Pass a report object directly, or ``None`` to have the builder
        generate one from ``_eval_result`` (requires evaluate data).
        """
        if report is not None:
            self._diagnosis = report
        else:
            from proscore.evaluate import diagnose
            self._diagnosis = diagnose(self._eval_result)
        return self

    def with_oot_period_eval(self, oot_period_df: pd.DataFrame) -> ReportBuilder:
        """Attach multi-period OOT metrics from :func:`evaluate_by_period`."""
        self._oot_period_eval = oot_period_df
        return self

    def with_stability(self, stability_result: pd.DataFrame) -> ReportBuilder:
        self._stability_result = stability_result
        return self

    def with_monitor(self, monitor) -> ReportBuilder:
        """Attach a :class:`ModelMonitor` instance (renders monitoring history)."""
        self._monitor = monitor
        return self

    def with_plot_data(
        self,
        y_test: np.ndarray | None = None,
        prob_test: np.ndarray | None = None,
        train_scores: np.ndarray | None = None,
        test_scores: np.ndarray | None = None,
    ) -> ReportBuilder:
        """Provide data for KS / ROC / score-distribution diagnostic plots."""
        self._plot_ytest = y_test
        self._plot_probtest = prob_test
        self._plot_scores_trn = train_scores
        self._plot_scores_tst = test_scores
        return self

    # ── build ──────────────────────────────────────────────────────────────

    def build(self) -> str:
        """Assemble and return the complete Markdown report."""
        parts: list[str] = []

        parts.append(self._render_header())
        parts.append(self._render_overview())

        if self._detect is not None:
            parts.append(self._render_inspect())

        if self._bin_tables:
            parts.append(self._render_binning())

        if self._filter_stages or self._stepwise_record:
            parts.append(self._render_selection())

        if self._scorecard is not None:
            parts.append(self._render_model())

        if self._eval_result:
            parts.append(self._render_evaluate())
        if self._diagnosis and self._diagnosis.issues:
            parts.append(self._render_diagnosis())

        parts.append(self._render_conclusions())

        return "\n\n".join(p for p in parts if p)

    def save(
        self,
        path: str,
        *,
        embed_images: bool = True,
        write_html: bool = True,
    ) -> str:
        """Build the report, write to *path*, and return the absolute path.

        Creates parent directories if they do not exist.  When the ``viz``
        module is available, saves diagnostic plot images into
        ``<report_dir>/<plot_dir>/``.

        Parameters
        ----------
        embed_images : bool
            If ``True`` (default), embed PNGs as base64 inside HTML ``<img>`` tags.
            Cursor/VS Code **Markdown 预览通常不显示** ``![...](data:image/png;...)``，
            请用生成的 ``.html`` 在浏览器中查看，或依赖 HTML 标签渲染。
        write_html : bool
            If ``True`` (default), also write ``<stem>.html`` next to the Markdown
            file (relative ``plots/*.png`` paths — reliable in any browser).
        """
        import os

        report_dir = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(report_dir, exist_ok=True)
        self._report_dir = report_dir
        self._embed_images = embed_images

        # Generate diagnostic plots (before build so images can be referenced)
        if self._plots_module is not None:
            plot_path = os.path.join(report_dir, self.plot_dir)
            os.makedirs(plot_path, exist_ok=True)
            try:
                self._save_plots(plot_path)
            except Exception as e:
                warnings.warn(f"Plot generation failed: {e}", stacklevel=2)

        content = self.build()
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        md_path = os.path.abspath(path)
        if write_html:
            base, _ = os.path.splitext(md_path)
            html_path = base + ".html" if base else md_path + ".html"
            self.save_html(html_path, markdown_path=md_path)
        return md_path

    def save_html(self, path: str, *, markdown_path: str | None = None) -> str:
        """Write an HTML report with the same content (images use relative paths).

        Open the returned ``.html`` file in Chrome / Safari / Edge to view all figures.
        """
        import os

        md_src = markdown_path or path.replace(".html", ".md")
        if not os.path.isfile(md_src):
            md_src = path
        with open(md_src, encoding="utf-8") as f:
            md_text = f.read()
        html_body = _markdown_to_html(md_text)
        html = _HTML_TEMPLATE.format(body=html_body)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return os.path.abspath(path)

    def _save_plots(self, plot_path: str) -> None:
        """Save all diagnostic plots to *plot_path*."""
        viz = self._plots_module
        if viz is None:
            return
        import matplotlib.pyplot as plt

        # 1. Binning plots
        for var, bt in self._bin_tables.items():
            try:
                fig = viz.plot_binning(bt, figsize=(10, 4.5))
                fig.savefig(f"{plot_path}/{_safe_filename(var)}_binning.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                warnings.warn(f"Plot generation failed for {var}: {e}", stacklevel=2)

        # 2. KS curve
        if self._plot_ytest is not None and self._plot_probtest is not None:
            try:
                fig = viz.plot_ks(self._plot_ytest, self._plot_probtest)
                fig.savefig(f"{plot_path}/ks_curve.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                warnings.warn(f"Plot generation failed for KS curve: {e}", stacklevel=2)

        # 3. ROC curve
        if self._plot_ytest is not None and self._plot_probtest is not None:
            try:
                fig = viz.plot_roc(self._plot_ytest, self._plot_probtest)
                fig.savefig(f"{plot_path}/roc_curve.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                warnings.warn(f"Plot generation failed for ROC curve: {e}", stacklevel=2)

        # 4. Score distribution
        if self._plot_scores_trn is not None:
            try:
                fig = viz.plot_score_distribution(self._plot_scores_trn, self._plot_scores_tst)
                fig.savefig(f"{plot_path}/score_distribution.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                warnings.warn(f"Plot generation failed for score distribution: {e}", stacklevel=2)

    # ── renderers ──────────────────────────────────────────────────────────

    def _meta(self, k: str, v: Any) -> str:
        if v is None or v == "":
            return ""
        return f"**{k}**：{v}"

    def _render_header(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            self._meta("项目", self.project),
            self._meta("日期", self._date),
            self._meta("人员", self.modeler),
        ]
        if self.purpose:
            lines += ["", f"> {self.purpose}"]
        return "\n".join(ln for ln in lines if ln)

    def _render_overview(self) -> str:
        lines = ["---", "", "## 0. 建模概览", ""]

        rs = self._run_summary
        n_train = rs.get("n_train")
        if n_train is None and self._detect is not None and "count" in self._detect.columns:
            n_train = int(self._detect["count"].iloc[0])
        n_test = rs.get("n_test", "?")
        n_oot = rs.get("n_oot")
        train_br = f"{self._target_bad_rate * 100:.1f}%" if self._target_bad_rate else "?"

        # Binning
        bin_method = next(iter(self._bin_tables.values())).method if self._bin_tables else "?"

        # Selection: 入模 / 建模候选池 / 初始候选
        n_final = rs.get("n_final")
        if n_final is None:
            n_final = len(self._stepwise_support)
        n_pool = rs.get("n_model_pool")
        if n_pool is None and self._filter_stages:
            n_pool = len(self._filter_stages[-1]["support"])
        n_cand = rs.get("n_candidates")
        if n_cand is None and self._detect is not None:
            n_cand = len(self._detect)

        # Performance
        ev = self._eval_result
        trn_ks = f"{ev['trn_ks']:.4f}" if ev else "?"
        tst_ks = f"{ev['test_ks']:.4f}" if ev else "?"
        trn_auc = f"{ev['trn_auc']:.4f}" if ev else "?"
        tst_auc = f"{ev['test_auc']:.4f}" if ev else "?"
        psi = f"{ev['psi']:.4f}" if ev else "?"

        lines += [
            "| 项目 | 内容 |",
            "|------|------|",
            f"| 训练样本量 | {n_train if n_train is not None else '?'} |",
            f"| 测试样本量 | {n_test} |",
            f"| OOT 样本量 | {n_oot if n_oot is not None else '—'} |",
            f"| 训练坏账率 | {train_br} |",
            f"| 分箱方法 | {bin_method} |",
            f"| 入模变量 / 建模池 / 初始候选 "
            f"| {n_final} / {n_pool if n_pool is not None else '?'} "
            f"/ {n_cand if n_cand is not None else '?'} |",
            f"| Train KS / AUC | {trn_ks} / {trn_auc} |",
            f"| Test KS / AUC | {tst_ks} / {tst_auc} |",
            f"| 分数 PSI | {psi} |",
        ]

        # Overall assessment
        ks_score = ev.get("test_ks", 0) if ev else 0
        psi_score = ev.get("psi", 1) if ev else 1
        ks_level = "良好" if ks_score > 0.4 else ("一般" if ks_score > 0.25 else "不足")
        psi_level = "稳定" if psi_score < 0.1 else ("轻微漂移" if psi_score < 0.25 else "需关注")

        lines += [
            "",
            f"> **结论**：模型区分力 **{ks_level}**（Test KS={tst_ks}），",
            f"> 稳定性 **{psi_level}**（PSI={psi}）。",
        ]
        return "\n".join(lines)

    def _render_inspect(self) -> str:
        lines = ["---", "", "## 1. 数据探查", ""]
        sec = 1

        det = self._detect
        if det is not None and len(det) > 0:
            lines += [
                f"- 探查变量数（不含目标）：{len(det)}",
                "",
                f"### 1.{sec} 变量质量（detect）",
                "",
            ]
            sec += 1
            det_cols = [
                c for c in [
                    "variable", "dtype", "count", "missing_pct", "one_value_pct",
                    "n_unique", "special_pct", "target_pearson", "target_cramers_v",
                ]
                if c in det.columns
            ]
            lines.append(_md_table_safe(det[det_cols].head(30)))
            lines.append("")

        qdf = self._quality
        if qdf is not None and len(qdf) > 0:
            lines += [f"### 1.{sec} 预测力指标（quality，Train + Test PSI）", ""]
            sec += 1
            qcols = [
                c for c in [
                    "variable", "dtype", "iv", "auc", "ks", "psi", "missing_pct", "n_unique",
                ]
                if c in qdf.columns
            ]
            lines.append(_md_table_safe(qdf[qcols]))
            lines.append("")

        if self._corr is not None and len(self._corr) > 0:
            lines += [f"### 1.{sec} 高相关变量对", "", _md_table_safe(self._corr.head(20)), ""]
            sec += 1

        if self._vif is not None and len(self._vif) > 0:
            lines += [f"### 1.{sec} VIF 汇总", "", _md_table_safe(self._vif.head(20)), ""]
            sec += 1

        if self._stability_result is not None and len(self._stability_result) > 0:
            lines.extend(self._render_stability_block(f"### 1.{sec} 时序稳定性（stability）"))
            sec += 1

        if self._target_dist is not None:
            lines += [
                f"### 1.{sec} 目标变量分布",
                "",
                _md_table_safe(self._target_dist),
                "",
            ]

        return "\n".join(lines)

    def _render_binning(self) -> str:
        lines = ["---", "", "## 2. 分箱", ""]

        if self._iv_table is not None and len(self._iv_table) > 0:
            lines += [
                "### 2.1 IV 汇总",
                "",
                _md_table_safe(self._iv_table),
                "",
            ]

        lines += ["### 2.2 变量分箱详情", ""]
        for var, bt in list(self._bin_tables.items())[:10]:  # top-10 by IV
            iv_str = f"IV={bt.iv_total:.4f}" if bt.iv_total else ""
            trend_str = {0: "无", 1: "递增", 2: "递减", 3: "U型", 4: "倒U型"}.get(
                bt.monotonic, "?"
            )
            match_str = "✓" if bt.trend_match else "⚠ 不符"
            lines += [
                f"#### {var}  {iv_str}  趋势：{trend_str}  {match_str}",
                "",
            ]
            rows = []
            for b in bt.bins:
                rows.append({
                    "箱": b.bin_no,
                    "区间": b.bin_label,
                    "样本": b.count,
                    "坏账率": round(b.bad_rate, 4),
                    "WOE": round(b.woe, 4),
                    "IV": round(b.iv, 4),
                })
            lines.append(_md_table_safe(pd.DataFrame(rows)))
            img = _html_image(
                self._report_dir,
                self.plot_dir,
                f"{_safe_filename(var)}_binning.png",
                embed=self._embed_images,
                alt=f"{var} 分箱图",
            )
            if img:
                lines += ["", "**分箱图**", "", img, ""]
            lines.append("")

        return "\n".join(lines)

    def _render_stability_block(self, heading: str) -> list[str]:
        """Stability subsection (used in §1 数据探查)."""
        sr = self._stability_result
        if sr is None or len(sr) == 0:
            return []
        lines = [heading, ""]
        for flag_col, flag, label in [
            ("psi_flag", "unstable", "PSI 分布漂移"),
            ("bad_rate_flag", "trending_up", "坏账率上升"),
            ("bad_rate_flag", "trending_down", "坏账率下降"),
        ]:
            if flag_col not in sr.columns:
                continue
            subset = sr[sr[flag_col] == flag]
            if len(subset) == 0:
                continue
            show = [
                c for c in [
                    "variable", "time_period", "bad_rate", "bad_rate_change",
                    "psi_vs_first", "psi_flag", "bad_rate_flag",
                ]
                if c in subset.columns
            ]
            lines += [
                f"**{label}**（{len(subset['variable'].unique())} 个变量）",
                "",
                _md_table_safe(subset[show].head(15)),
                "",
            ]
        return lines

    @staticmethod
    def _filter_dropped_rows(qf: pd.DataFrame) -> pd.DataFrame:
        if "dropped" in qf.columns:
            return qf[qf["dropped"]]
        return qf[~qf["selected"]]

    def _render_filter_stage(
        self,
        quality: pd.DataFrame,
        support: list[str],
        params: dict[str, Any],
        *,
        heading: str,
    ) -> list[str]:
        dropped = self._filter_dropped_rows(quality)
        feat_col = "feature" if "feature" in quality.columns else "variable"
        show_cols = [c for c in [feat_col, "reason", "missing_rate", "iv"] if c in dropped.columns]
        lines = [
            heading,
            "",
            f"- 阈值：missing≤{params.get('max_missing_rate', '')}，"
            f"one_value≤{params.get('max_one_value_rate', '')}，"
            f"IV∈{params.get('iv_range', '')}，"
            f"corr≤{params.get('max_corr', '')}，"
            f"PSI≤{params.get('max_psi', '')}",
            f"- 剔除 {len(dropped)} 个，保留 {len(support)} 个",
            "",
        ]
        if len(support) == 0 and len(quality) > 0:
            all_cols = [c for c in [feat_col, "reason", "missing_rate", "iv", "dropped"] if c in quality.columns]
            lines += [
                "**本阶段无保留变量**（正常筛选结果），全部候选状态：",
                "",
                _md_table_safe(quality[all_cols].head(30)),
                "",
            ]
        elif len(dropped) > 0 and show_cols:
            lines += [
                "**剔除变量**：",
                "",
                _md_table_safe(dropped[show_cols].head(20)),
                "",
            ]
        return lines

    def _render_selection(self) -> str:
        lines = ["---", "", "## 3. 特征筛选", ""]

        if self._pipeline_halts:
            lines += [
                "> **流水线提示**（筛选后无可建模变量或阶段已跳过，"
                "以下为正常业务结论，非程序错误）",
                "",
            ]
            for h in self._pipeline_halts:
                lines += [f"- **{h['stage']}**：{h['message']}", ""]
            lines.append("")

        # Filter stage(s)
        for idx, st in enumerate(self._filter_stages, start=1):
            lines.extend(
                self._render_filter_stage(
                    st["quality"],
                    st["support"],
                    st["params"],
                    heading=f"### 3.{idx} {st['stage']}",
                )
            )

        # Stepwise
        step_idx = len(self._filter_stages) + 1
        if self._stepwise_record:
            n_rounds = len(self._stepwise_record)
            best = self._stepwise_best
            lines += [
                f"### 3.{step_idx} 双向迭代（StepwiseSelector）",
                "",
                f"- 目标函数：{self._stepwise_params.get('objective','?')}",
                f"- P值阈值：{self._stepwise_params.get('pvalue_threshold','无')}",
                f"- 系数符号：{self._stepwise_params.get('coef_sign','无')}",
                f"- 迭代轮次：{n_rounds}，最终变量数：{len(self._stepwise_support)}",
                f"- 最佳性能：trn_KS={best.get('trn_ks',0):.4f}，"
                f"test_KS={best.get('test_ks',0):.4f}",
                "",
            ]
            # Iteration summary
            rec_rows = []
            for rnd, rec in sorted(self._stepwise_record.items()):
                rec_rows.append({
                    "轮次": rnd,
                    "变量数": rec.get("n_vars", "?"),
                    "得分": round(rec.get("score", 0), 4),
                    "trn_KS": round(rec.get("trn_ks", 0), 4),
                    "test_KS": round(rec.get("test_ks", 0), 4),
                    "通过": "✓" if rec.get("passed") else "",
                })
            lines.append(_md_table_safe(pd.DataFrame(rec_rows)))
            lines.append("")
            finals = self._run_summary.get("final_features") or self._stepwise_support
            if finals:
                lines += [f"- **最终入模变量**：{', '.join(finals)}", ""]

        return "\n".join(lines)

    def _render_model(self) -> str:
        lines = ["---", "", "## 4. 评分卡模型", ""]
        sc = self._scorecard

        # LR coefficients from fitted model params (not summary2 column mis-parse)
        if sc is not None and hasattr(sc, "model_") and sc.model_ is not None:
            lines += ["### 4.1 LR 系数", "", f"截距：{sc.intercept_:.6f}", ""]
            params = sc.model_.params
            coef_rows = []
            for name in params.index:
                if str(name).lower() in ("const", "intercept"):
                    continue
                coef_rows.append({
                    "变量": str(name),
                    "系数": round(float(params[name]), 6),
                })
            if coef_rows:
                lines += [_md_table_safe(pd.DataFrame(coef_rows)), ""]
            else:
                lines += ["（无特征系数）", ""]

        # Scorecard table
        if self._score_table is not None and len(self._score_table) > 0:
            lines += [
                "### 4.2 评分卡",
                "",
                f"参数：odds={sc.odds}, pdo={sc.pdo}, base_score={sc.base_score}",
                "",
            ]
            display = self._score_table[
                ["variable", "bin_label", "woe", "coef", "points"]
            ].copy()
            display["woe"] = display["woe"].round(4)
            display["coef"] = display["coef"].round(4)
            display["points"] = display["points"].round(1)
            lines.append(_md_table_safe(display))
            lines.append("")

        return "\n".join(lines)

    def _render_diagnosis(self) -> str:
        """Render the diagnosis report as a Markdown section."""
        lines = ["## 模型诊断", ""]
        report = self._diagnosis
        for level, label in [("critical", "严重"), ("warning", "警告"), ("info", "提示")]:
            items = [i for i in report.issues if i.level == level]
            if not items:
                continue
            lines.append(f"### {label}")
            for item in items:
                lines.append(f"- **{item.title}** — {item.evidence}")
                lines.append(f"  → {item.suggestion}")
            lines.append("")
        return "\n".join(lines)

    def _render_evaluate(self) -> str:
        lines = ["---", "", "## 5. 模型评估", ""]
        ev = self._eval_result

        lines += [
            "### 5.1 核心指标",
            "",
            "| 指标 | 训练集 | 测试集 | 衰退 |",
            "|------|--------|--------|------|",
            f"| KS | {ev.get('trn_ks',0):.4f} | {ev.get('test_ks',0):.4f} "
            f"| {ev.get('ks_reduce',0):.4f} |",
            f"| AUC | {ev.get('trn_auc',0):.4f} | {ev.get('test_auc',0):.4f} "
            f"| {ev.get('trn_auc',0)-ev.get('test_auc',0):.4f} |",
            f"| Acc | {ev.get('trn_acc',0):.4f} | {ev.get('test_acc',0):.4f} | — |",
            f"| PSI | — | {ev.get('psi',0):.4f} | — |",
            f"| KS 相对衰退 | — | — | {ev.get('ks_rel_gap',0):.4f} |",
            "",
        ]

        if ev.get("oot_ks") is not None:
            lines += [
                "**合并 OOT（全量时间外）**：",
                "",
                f"- KS={ev.get('oot_ks', 0):.4f}，AUC={ev.get('oot_auc', 0):.4f}，"
                f"Acc={ev.get('oot_acc', 0):.4f}",
                "",
            ]

        if self._oot_period_eval is not None and len(self._oot_period_eval) > 0:
            lines += [
                "### 5.2 分年度 OOT 表现",
                "",
                _md_table_safe(self._oot_period_eval),
                "",
            ]

        # Score distribution
        st = ev.get("score_table")
        if st is not None and len(st) > 0:
            lines += [
                "### 5.3 评分分布与排序性",
                "",
                _md_table_safe(st),
                "",
            ]

        # Monitoring history
        if self._monitor is not None:
            try:
                hist = self._monitor.history
                if len(hist) > 0:
                    lines += [
                        "### 5.5 监控历史",
                        "",
                        _md_table_safe(hist),
                        "",
                    ]
                    # Latest snapshot detail
                    latest = self._monitor._snapshots[-1]
                    if latest.alerts:
                        lines += ["**最新告警**：", ""]
                        for a in latest.alerts:
                            lines.append(f"- {a}")
                        lines += [
                            "",
                            f"> {latest.recommendation}",
                            "",
                        ]
            except Exception:
                pass

        # Diagnostic plots (only embed files that were saved)
        diag_parts: list[str] = []
        for title, fname in [
            ("KS 曲线", "ks_curve.png"),
            ("ROC 曲线", "roc_curve.png"),
            ("评分分布", "score_distribution.png"),
        ]:
            img = _html_image(
                self._report_dir,
                self.plot_dir,
                fname,
                embed=self._embed_images,
                alt=title,
            )
            if img:
                diag_parts += [f"**{title}**", "", img, ""]
        if diag_parts:
            lines += ["### 5.4 诊断图", ""] + diag_parts
        elif self._plots_module is not None:
            lines += [
                "### 5.4 诊断图",
                "",
                "（未生成：请确认已调用 `with_plot_data` 且 `report.save()` 成功保存图片）",
                "",
            ]

        return "\n".join(lines)

    def _render_conclusions(self) -> str:
        lines = ["---", "", "## 6. 结论与建议", ""]
        ev = self._eval_result

        # Strengths
        ks = ev.get("test_ks", 0) if ev else 0
        psi = ev.get("psi", 1) if ev else 1
        ks_r = ev.get("ks_rel_gap", 1) if ev else 1

        strengths, risks = [], []

        if ks > 0.4:
            strengths.append(f"模型区分力良好（Test KS={ks:.4f}）")
        elif ks > 0.25:
            risks.append(f"模型区分力一般（Test KS={ks:.4f}），建议增加特征或调整分箱")

        if psi < 0.1:
            strengths.append(f"分数分布稳定（PSI={psi:.4f}）")
        elif psi < 0.25:
            risks.append(f"分数轻微漂移（PSI={psi:.4f}），建议持续监控")
        else:
            risks.append(f"分数分布漂移明显（PSI={psi:.4f}），需排查原因")

        if ks_r > 0.1:
            risks.append(f"KS 衰退较大（相对={ks_r:.4f}），模型可能过拟合")

        strength_lines = [f"- {s}" for s in strengths] if strengths else ["- （待补充）"]
        risk_lines = [f"- {r}" for r in risks] if risks else ["- 暂无显著风险"]

        lines += ["**优势**："] + strength_lines + [""]
        lines += ["**风险点**："] + risk_lines + [""]
        lines += ["**建议**：",
            "- 定期监控 KS / AUC / PSI，建议频率：季度",
            "- 关注趋势不符变量（trend_match=False），必要时手工分箱",
            "- 若 PSI 持续上升，考虑模型重新训练",
            "",
        ]

        # Appendix note
        lines += [
            "---",
            "",
            "## 附录",
            "",
            f"- 报告生成时间：{self._date}",
            "- 完整配置、数据表、迭代日志可通过对应模块的导出方法获取",
            "- 详表：`Filter.quality_`、`Binning.bin_table_`、`ScoreCard.score_table_`、",
            "  `StepwiseSelector.record_`、`evaluate()` 输出、`stability()` 输出",
        ]

        return "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────────


def _safe_filename(name: str) -> str:
    """Replace characters unsafe for filenames."""
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _html_image(
    report_dir: str,
    plot_dir: str,
    filename: str,
    *,
    embed: bool = False,
    alt: str = "",
) -> str:
    """HTML img tag (works in Cursor/VS Code preview; MD preview often ignores data: URIs)."""
    import base64
    import html as html_mod
    import os

    label = html_mod.escape(alt or filename)
    full = os.path.join(report_dir, plot_dir, filename) if report_dir else ""
    if report_dir and not os.path.isfile(full):
        return ""
    rel = f"{plot_dir}/{filename}".replace("\\", "/")
    src = rel
    if embed and report_dir and full:
        with open(full, "rb") as img_f:
            b64 = base64.b64encode(img_f.read()).decode("ascii")
        src = f"data:image/png;base64,{b64}"
    return (
        f'<p align="center"><img src="{src}" alt="{label}" '
        f'style="max-width:960px;width:100%;height:auto;"/></p>'
    )


def _md_image(
    report_dir: str,
    plot_dir: str,
    filename: str,
    *,
    embed: bool = False,
    alt: str = "",
) -> str:
    """Alias for :func:`_html_image` (backward compatibility)."""
    return _html_image(report_dir, plot_dir, filename, embed=embed, alt=alt)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ProScore 评分卡报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1100px; margin: 24px auto; line-height: 1.5; color: #222; }}
h1,h2,h3,h4 {{ color: #1a1a1a; }}
table {{ border-collapse: collapse; margin: 12px 0; font-size: 13px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; }}
th {{ background: #f4f4f4; }}
img {{ max-width: 100%; height: auto; }}
blockquote {{ background: #f8f8f8; border-left: 4px solid #888; padding: 8px 12px; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _markdown_to_html(md_text: str) -> str:
    """Lightweight Markdown → HTML (tables, headings, images, blockquotes)."""
    import html as html_mod
    import re

    lines = md_text.split("\n")
    out: list[str] = []
    i = 0
    img_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("|") and stripped.count("|") >= 2:
            block: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(_md_table_block_to_html(block))
            continue

        if stripped.startswith("### "):
            out.append(f"<h3>{html_mod.escape(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            out.append(f"<h2>{html_mod.escape(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            out.append(f"<h1>{html_mod.escape(stripped[2:])}</h1>")
        elif stripped.startswith("> "):
            out.append(f"<blockquote><p>{html_mod.escape(stripped[2:])}</p></blockquote>")
        elif stripped == "---":
            out.append("<hr/>")
        elif stripped == "":
            out.append("")
        elif "<img" in line:
            out.append(line)
        else:

            def _img_sub(m: re.Match[str]) -> str:
                alt = html_mod.escape(m.group(1))
                src = m.group(2)
                return (
                    f'<img src="{src}" alt="{alt}" '
                    f'style="max-width:960px;width:100%;height:auto;"/>'
                )

            para = img_re.sub(_img_sub, line)
            if para == line:
                para = html_mod.escape(line)
            out.append(f"<p>{para}</p>")
        i += 1

    return "\n".join(out)


def _md_table_block_to_html(block: list[str]) -> str:
    """Convert a GFM pipe table block to HTML."""
    import html as html_mod

    rows = [r.strip() for r in block if r.strip()]
    if len(rows) < 2:
        return "\n".join(block)
    sep_idx = None
    for idx, row in enumerate(rows):
        if set(row.replace("|", "").strip()) <= {"-", ":"}:
            sep_idx = idx
            break
    if sep_idx is not None:
        header_rows = rows[:sep_idx]
        data_rows = rows[sep_idx + 1 :]
    else:
        header_rows = rows[:1]
        data_rows = rows[1:]

    def split_row(row: str) -> list[str]:
        parts = [c.strip() for c in row.strip("|").split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        return parts

    html_rows: list[str] = []
    for row in header_rows:
        cells = [html_mod.escape(c) for c in split_row(row)]
        if cells:
            html_rows.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
    for row in data_rows:
        if set(row.replace("|", "").strip()) <= {"-", ":"}:
            continue
        cells = [html_mod.escape(c) for c in split_row(row)]
        if cells:
            html_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    if not html_rows:
        return ""
    return "<table>\n" + "\n".join(html_rows) + "\n</table>\n"


def _md_table_safe(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table, with NaN/Inf handling."""
    if df is None or len(df) == 0:
        return "（无数据）"
    safe = df.copy()
    for col in safe.columns:
        if safe[col].dtype in (float, np.float64):
            safe[col] = safe[col].apply(
                lambda x: "" if pd.isna(x) or np.isinf(x) else x
            )
    try:
        return safe.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError):
        return safe.to_string(index=False)
