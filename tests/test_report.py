"""Tests for ReportBuilder markdown output."""

from __future__ import annotations

import pandas as pd

from proscore.report import ReportBuilder


def test_overview_uses_run_summary():
    rb = ReportBuilder()
    rb.with_run_summary(n_train=263, n_test=114, n_candidates=11, n_final=5, n_model_pool=6)
    rb.with_inspect(
        detect=pd.DataFrame({"variable": ["a"], "count": [1], "missing_pct": [0]}),
    )
    text = rb.build()
    assert "| 测试样本量 | 114 |" in text
    assert "5 / 6 / 11" in text


def test_inspect_shows_variable_names():
    rb = ReportBuilder()
    rb.with_inspect(
        detect=pd.DataFrame({
            "variable": ["income"],
            "dtype": ["numeric"],
            "missing_pct": [1.0],
            "one_value_pct": [0.1],
            "n_unique": [100],
        }),
        quality=pd.DataFrame({
            "variable": ["income"],
            "iv": [0.15],
            "auc": [0.55],
            "psi": [0.02],
        }),
    )
    text = rb.build()
    assert "income" in text
    assert "variable" in text.lower() or "income" in text


def test_embed_images_uses_data_uri(tmp_path):
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()
    (plot_dir / "t.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    rb = ReportBuilder(plot_dir="plots")
    rb._report_dir = str(tmp_path)
    rb._embed_images = True
    rb._bin_tables = {
        "x": type(
            "BT", (),
            {"bins": [], "iv_total": 0, "monotonic": 0, "trend_match": True, "method": "chi"},
        )()
    }
    from proscore.report._builder import _html_image

    html = _html_image(str(tmp_path), "plots", "t.png", embed=True, alt="图")
    assert '<img src="data:image/png;base64,' in html
    assert 'alt="图"' in html


def test_save_writes_html(tmp_path):
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()
    (plot_dir / "ks.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    rb = ReportBuilder(plot_dir="plots")
    rb.with_run_summary(n_train=10, n_test=5, n_candidates=3, n_final=2, n_model_pool=2)
    md_path = tmp_path / "report.md"
    rb.save(str(md_path), embed_images=False, write_html=True)
    html_path = tmp_path / "report.html"
    assert html_path.is_file()
    body = html_path.read_text(encoding="utf-8")
    assert "plots/ks.png" in body or "<table>" in body


def test_lr_coef_uses_param_names():
    class FakeModel:
        params = pd.Series(
            {"age": 1.17, "debt_ratio": 1.08, "const": -2.2},
        )

    class FakeSC:
        intercept_ = -2.2
        odds = 20
        pdo = 20
        base_score = 600
        model_ = FakeModel()
        score_table_ = pd.DataFrame()

    rb = ReportBuilder()
    rb.with_model(FakeSC())
    text = rb.build()
    assert "| age |" in text or "age" in text
    assert "debt_ratio" in text
