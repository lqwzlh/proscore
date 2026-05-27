"""Verify that code examples in README and key docs are executable.

Covers:
  - README: chain API example (with test_data.csv)
  - pipeline-config.md: template + run flow (CLI smoke)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEST_DATA = HERE / "test_data.csv"


# ── helpers ──────────────────────────────────────────────────────────────────


def _train_test_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Produce train/test/oot from test_data.csv, matching README example."""
    df = pd.read_csv(TEST_DATA, parse_dates=["apply_date"])
    train = df[df["apply_date"].dt.year <= 2021].drop(
        columns=["apply_date"]
    )
    test = df[df["apply_date"].dt.year == 2022].drop(
        columns=["apply_date"]
    )
    oot = df[df["apply_date"].dt.year == 2023].drop(
        columns=["apply_date"]
    )
    return train, test, oot


# ── README: chain API example (Scenario B) ──────────────────────────────────


class TestReadmeChainAPI:
    """README 'B. 链式 API' code block — must copy-paste run without error."""

    def test_chain_api_runs(self):
        import proscore as ps

        train, test, oot = _train_test_split()

        p = (
            ps.ProScore()
            .read(train=train, test=test, oot=oot, target="bad_flag")
            .detect()
            .prefilter()
            .bin(method="chi", n_bins=5)
            .refine(iv_range=(0.02, None))
            .transform()
            .select()
            .fit(odds=20, pdo=20, base_score=600)
            .scorecard()
            .evaluate()
        )

        assert p.eval_result is not None
        assert "trn_ks" in p.eval_result
        assert "test_ks" in p.eval_result
        assert "oot_ks" in p.eval_result
        assert len(p.support_) >= 3

    def test_chain_api_train_only(self):
        """README notes 'train required, test/oot optional' — verify."""
        import proscore as ps

        train, _, _ = _train_test_split()

        p = (
            ps.ProScore()
            .read(train=train, target="bad_flag")
            .prefilter()
            .bin(method="chi", n_bins=5)
            .refine(iv_range=(0.02, None))
            .transform()
            .select()
            .fit(odds=20, pdo=20, base_score=600)
            .scorecard()
            .evaluate()
        )

        assert "trn_ks" in p.eval_result


# ── README: Excel config example (Scenario C) ────────────────────────────────


class TestReadmeExcelFlow:
    """README 'C. Excel 配置驱动' code block — template + run flow."""

    def test_template_creates_valid_file(self, tmp_path):
        """proscore template → generates .xlsx with all 8 sheets."""
        result = subprocess.run(
            [sys.executable, "-m", "proscore", "template", str(tmp_path)],
            capture_output=True, text=True, timeout=30,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        xlsx = tmp_path / "pipeline_template.xlsx"
        assert xlsx.exists()

        sheets = pd.ExcelFile(xlsx, engine="openpyxl").sheet_names
        for name in ("Global", "Data", "Steps", "Binning", "Screening",
                     "Modeling", "Rules", "Variables"):
            assert name in sheets, f"Missing sheet: {name}"
        steps = pd.read_excel(xlsx, sheet_name="Steps", engine="openpyxl")
        mine_row = steps.loc[steps["参数名"] == "mine_rules", "默认值"]
        assert len(mine_row) == 1 and str(mine_row.iloc[0]).strip().lower() == "off"

    def test_filled_template_runs(self, tmp_path):
        """Fill the template with test_data.csv and run."""
        import openpyxl  # noqa: F401

        # 1. Generate template
        subprocess.run(
            [sys.executable, "-m", "proscore", "template", str(tmp_path)],
            capture_output=True, timeout=30,
            cwd=str(tmp_path),
        )
        xlsx = tmp_path / "pipeline_template.xlsx"

        # 2. Fill Data sheet
        with pd.ExcelFile(xlsx, engine="openpyxl") as xl:
            sheets = {s: xl.parse(s) for s in xl.sheet_names}

        data_sheet = sheets["Data"]
        # Avoid FutureWarning: convert the "您的取值" column to object dtype
        # before assigning string values (it defaults to float64 due to NaN).
        data_sheet["您的取值"] = data_sheet["您的取值"].astype(object)
        data_sheet.loc[
            data_sheet["参数名"] == "data_file", "您的取值"
        ] = str(TEST_DATA.resolve())
        data_sheet.loc[
            data_sheet["参数名"] == "target", "您的取值"
        ] = "bad_flag"
        data_sheet.loc[
            data_sheet["参数名"] == "time_col", "您的取值"
        ] = "apply_date"
        data_sheet.loc[
            data_sheet["参数名"] == "dev_end", "您的取值"
        ] = 2021
        data_sheet.loc[
            data_sheet["参数名"] == "oot_start", "您的取值"
        ] = 2022

        filled = tmp_path / "filled_pipeline.xlsx"
        with pd.ExcelWriter(filled, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)

        # 3. Run pipeline
        result = subprocess.run(
            [sys.executable, "-m", "proscore", "run", str(filled),
             "--output-script", str(tmp_path / "generated.py")],
            capture_output=True, text=True, timeout=120,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"

        # 4. Check outputs
        report = tmp_path / "scorecard_report" / "report.md"
        script = tmp_path / "generated.py"
        assert report.exists(), f"Report missing at {report}"
        assert script.exists(), f"Generated script missing at {script}"
        content = script.read_text()
        assert "ProScore" in content
        assert "p.read(" in content or "p.bin(" in content


class TestImportSmoke:
    """Every public import listed in docs must resolve."""

    def test_inspect_imports(self):
        from proscore.inspect import detect, quality, stability, vif  # noqa: F401

    def test_selection_imports(self):
        from proscore.selection import Filter, StepwiseSelector  # noqa: F401

    def test_binning_imports(self):
        from proscore.binning import Binning, BinningProcess  # noqa: F401

    def test_transform_imports(self):
        from proscore.transform import WOETransformer  # noqa: F401

    def test_modeling_imports(self):
        from proscore.modeling import ScoreCard  # noqa: F401

    def test_evaluate_imports(self):
        from proscore.evaluate import evaluate  # noqa: F401

    def test_report_imports(self):
        from proscore.report import ReportBuilder  # noqa: F401

    def test_viz_imports(self):
        from proscore.viz import (  # noqa: F401
            plot_binning,
            plot_ks,
            plot_roc,
            plot_score_distribution,
        )

    def test_monitor_imports(self):
        from proscore.monitor import ModelMonitor  # noqa: F401

    def test_utils_imports(self):
        from proscore.utils import load_presets  # noqa: F401

    def test_pipeline_config_import(self):
        from proscore import PipelineConfig, run_pipeline  # noqa: F401
