"""Excel pipeline — Rules sheet, rules_cfg bare keys, mine_rules step."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from proscore._pipeline_config import PipelineConfig, generate_template


@pytest.fixture
def template_xlsx(tmp_path: Path) -> Path:
    path = generate_template(str(tmp_path))
    return Path(path)


class TestRulesExcel:
    def test_rules_sheet_uses_bare_param_names(self, template_xlsx: Path) -> None:
        df = pd.read_excel(template_xlsx, sheet_name="Rules", engine="openpyxl")
        names = set(df["参数名"].astype(str).str.strip())
        assert "method" in names
        assert "min_lift" in names
        assert "rm_method" not in names

    def test_rules_cfg_bare_keys_from_excel(self, template_xlsx: Path, tmp_path: Path) -> None:
        with pd.ExcelFile(template_xlsx, engine="openpyxl") as xl:
            sheets = {s: xl.parse(s) for s in xl.sheet_names}
        rules = sheets["Rules"]
        rules["您的取值"] = rules["您的取值"].astype(object)
        rules.loc[rules["参数名"] == "method", "您的取值"] = "tree"
        rules.loc[rules["参数名"] == "min_lift", "您的取值"] = 2.5
        filled = tmp_path / "rules_parse.xlsx"
        with pd.ExcelWriter(filled, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
        cfg = PipelineConfig.from_excel(str(filled))
        assert cfg.rules_cfg["method"] == "tree"
        assert cfg.rules_cfg["min_lift"] == pytest.approx(2.5)
        assert "rm_method" not in cfg.rules_cfg

    def test_build_rules_kw_uses_bare_keys(self, template_xlsx: Path) -> None:
        cfg = PipelineConfig.from_excel(str(template_xlsx))
        cfg.rules_cfg["method"] = "apriori"
        cfg.rules_cfg["max_rules"] = 5
        kw = cfg._build_rules_kw()
        assert kw["method"] == "apriori"
        assert kw["max_rules"] == 5
        assert all(not k.startswith("rm_") for k in kw)

    def test_mine_rules_requires_refine(self, template_xlsx: Path, tmp_path: Path) -> None:
        with pd.ExcelFile(template_xlsx, engine="openpyxl") as xl:
            sheets = {s: xl.parse(s) for s in xl.sheet_names}
        steps = sheets["Steps"]
        steps["您的取值"] = steps["您的取值"].astype(object)
        steps.loc[steps["参数名"] == "mine_rules", "您的取值"] = "on"
        steps.loc[steps["参数名"] == "refine", "您的取值"] = "off"
        filled = tmp_path / "bad_steps.xlsx"
        with pd.ExcelWriter(filled, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
        with pytest.raises(ValueError, match="mine_rules"):
            PipelineConfig.from_excel(str(filled))
