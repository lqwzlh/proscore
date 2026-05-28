#!/usr/bin/env python3
"""Prepare a single ProScore-ready CSV from a real public dataset.

Writes ``data/processed/real_scorecard.csv`` with columns:
``apply_date``, numeric features, ``bad_flag``.

Auto-detects source (first match) when ``--input`` is omitted:
  1. data/home_credit/application_train.csv
  2. data/lending_club/LC_loans_granting_model_dataset.csv
  3. data/gmsc_train.csv

Lending Club **Kaggle 风格宽表**（含 ``loan_status`` + ``issue_d``，如
``lendingclub_data2018.csv``）：使用 ``--input /path/to.csv``，自动走
``lending_club_wide`` 分支（剔除还款后泄漏列，仅保留已结清/逾期样本）。

Usage::

    python scripts/prepare_real_scorecard_data.py
    python scripts/prepare_real_scorecard_data.py --sample 50000
    python scripts/prepare_real_scorecard_data.py --input ~/data/lendingclub_data2018.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "processed" / "real_scorecard.csv"
META = DATA / "processed" / "real_scorecard_meta.txt"

TARGET = "bad_flag"
DATE_COL = "apply_date"
MAX_FEATURES = 80
MISSING_RATE_CAP = 0.55

_LC_BAD_STATUSES = frozenset({
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Late (16-30 days)",
    "Does not meet the credit policy. Status:Charged Off",
})
_LC_GOOD_STATUSES = frozenset({"Fully Paid"})

# 还款/贷后信息，不应用于申请时点评分卡
_LC_LEAK_SUBSTR = (
    "total_pymnt",
    "total_rec_",
    "out_prncp",
    "recoveries",
    "collection_recovery",
    "last_pymnt",
    "next_pymnt",
    "last_credit_pull",
    "chargeoff_within",
)


def _infer_source(path: Path, columns: list[str]) -> str:
    cols = set(columns)
    if "TARGET" in cols:
        return "home_credit"
    if "SeriousDlqin2yrs" in cols:
        return "gmsc"
    if "Default" in cols or "default" in cols:
        return "lending_club"
    if "loan_status" in cols and "issue_d" in cols:
        return "lending_club_wide"
    raise ValueError(
        f"无法识别数据格式: {path}。列示例: {list(columns)[:25]}..."
    )


def _is_leaky_lc_column(name: str) -> bool:
    n = name.lower()
    return any(tok in n for tok in _LC_LEAK_SUBSTR)


def _parse_lc_percent(series: pd.Series) -> pd.Series:
    def one(v: object) -> float:
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        t = str(v).strip().rstrip("%").strip()
        if not t:
            return np.nan
        return float(pd.to_numeric(t, errors="coerce"))

    return series.map(one)


def _prepare_lending_club_wide(path: Path) -> pd.DataFrame:
    """Lending Club CSV with ``loan_status`` / ``issue_d`` (Kaggle-style export)."""
    df = pd.read_csv(path, low_memory=False)
    if "loan_status" not in df.columns or "issue_d" not in df.columns:
        raise ValueError(f"{path} 需要包含 loan_status 与 issue_d 列")

    ls = df["loan_status"].astype(str).str.strip()
    keep = ls.isin(_LC_BAD_STATUSES) | ls.isin(_LC_GOOD_STATUSES)
    df = df.loc[keep].copy()
    if len(df) < 50:
        raise ValueError(
            f"{path} 在剔除 Current/Issued 等状态后仅剩 {len(df)} 行，"
            "不足以建模。请换更晚年份或含更多 Charged Off / Fully Paid 的导出。"
        )

    df[TARGET] = (
        df["loan_status"].astype(str).str.strip().isin(_LC_BAD_STATUSES).astype(int)
    )
    df[DATE_COL] = pd.to_datetime(df["issue_d"], format="%y-%b", errors="coerce")
    if df[DATE_COL].isna().all():
        df[DATE_COL] = pd.to_datetime(df["issue_d"], errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    if "int_rate" in df.columns:
        df["int_rate"] = _parse_lc_percent(df["int_rate"])
    if "revol_util" in df.columns:
        df["revol_util"] = _parse_lc_percent(df["revol_util"])

    if "grade" in df.columns:
        gmap = {c: i + 1 for i, c in enumerate("ABCDEFG")}
        df["grade_ord"] = df["grade"].astype(str).str.strip().str.upper().map(gmap)

    drop_names = {
        "loan_status",
        "issue_d",
        "emp_title",
        "title",
        "zip_code",
        "addr_state",
        "purpose",
        "home_ownership",
        "verification_status",
        "application_type",
        "initial_list_status",
        "grade",
        "sub_grade",
        "emp_length",
        "earliest_cr_line",
        "member_id",
        TARGET,
        DATE_COL,
    }
    for c in list(df.columns):
        if c in drop_names:
            continue
        if _is_leaky_lc_column(c):
            df = df.drop(columns=[c], errors="ignore")

    num = df.select_dtypes(include=[np.number]).copy()
    for c in list(num.columns):
        if c in drop_names or _is_leaky_lc_column(c):
            num = num.drop(columns=[c], errors="ignore")

    miss = num.isna().mean()
    keep = miss[miss <= MISSING_RATE_CAP].index.tolist()
    if len(keep) > MAX_FEATURES:
        keep = miss[keep].sort_values().head(MAX_FEATURES).index.tolist()

    out = pd.concat([df[DATE_COL], num[keep], df[TARGET]], axis=1)
    out[TARGET] = out[TARGET].astype(int)
    return out


def _detect_source(explicit: Path | None) -> tuple[str, Path]:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--input 文件不存在: {p}")
        hdr = pd.read_csv(p, nrows=0)
        return _infer_source(p, list(hdr.columns)), p

    candidates = [
        ("home_credit", DATA / "home_credit" / "application_train.csv"),
        ("lending_club", DATA / "lending_club" / "LC_loans_granting_model_dataset.csv"),
        ("gmsc", DATA / "gmsc_train.csv"),
    ]
    for name, path in candidates:
        if path.exists() and path.stat().st_size > 1000:
            return name, path
    raise FileNotFoundError(
        "未找到真实数据文件。请先运行: python scripts/download_real_data.py gmsc\n"
        "或使用: python scripts/prepare_real_scorecard_data.py --input /path/to.csv"
    )


def _prepare_home_credit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={"TARGET": TARGET})
    if TARGET not in df.columns:
        raise ValueError(f"{path} 缺少 TARGET 列")

    # application_train 通常无真实申请日历列；用 SK_ID_CURR 生成稳定伪时间便于 Train/Test/OOT 切分
    if "DAYS_DECISION" in df.columns:
        anchor = pd.Timestamp("2016-01-01")
        days = pd.to_numeric(df["DAYS_DECISION"], errors="coerce")
        df[DATE_COL] = anchor + pd.to_timedelta(-days, unit="D")
    elif "SK_ID_CURR" in df.columns:
        sid = pd.to_numeric(df["SK_ID_CURR"], errors="coerce").fillna(0).astype(np.int64)
        years = 2014 + (sid % 3)
        df[DATE_COL] = pd.to_datetime(years.astype(str) + "-06-15")
    else:
        df[DATE_COL] = pd.date_range("2014-01-01", periods=len(df), freq="D")[: len(df)]

    drop_cols = {
        TARGET,
        DATE_COL,
        "SK_ID_CURR",
        "SK_ID_PREV",
        "SK_ID_BUREAU",
    }
    num = df.select_dtypes(include=[np.number]).copy()
    for c in list(num.columns):
        if c in drop_cols:
            num = num.drop(columns=[c], errors="ignore")
    miss = num.isna().mean()
    keep = miss[miss <= MISSING_RATE_CAP].index.tolist()
    if "DAYS_EMPLOYED" in keep:
        num.loc[num["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    if len(keep) > MAX_FEATURES:
        keep = miss[keep].sort_values().head(MAX_FEATURES).index.tolist()
    out = pd.concat([df[DATE_COL], num[keep], df[TARGET]], axis=1)
    return out


def _prepare_lending_club(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    target_col = "Default" if "Default" in df.columns else "default"
    if target_col not in df.columns:
        raise ValueError(f"{path} 缺少 Default 列")
    df = df.rename(columns={target_col: TARGET})
    df[DATE_COL] = pd.to_datetime(df["issue_d"], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET])

    numeric_cols = [
        "revenue",
        "dti_n",
        "loan_amnt",
        "fico_n",
        "experience_c",
    ]
    use = [c for c in numeric_cols if c in df.columns]
    out = pd.concat([df[DATE_COL], df[use], df[TARGET]], axis=1)
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce").fillna(0).astype(int)
    return out


def _prepare_gmsc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.iloc[:, 1:]
    df = df.rename(columns={"SeriousDlqin2yrs": TARGET})
    feat_cols = [c for c in df.columns if c != TARGET]
    rng = np.random.default_rng(42)
    years = rng.choice(
        [2012, 2013, 2014, 2015, 2016],
        size=len(df),
        p=[0.22, 0.22, 0.22, 0.22, 0.12],
    )
    df[DATE_COL] = pd.to_datetime(
        [f"{int(y)}-06-15" for y in years]
    )
    out = df[[DATE_COL] + feat_cols + [TARGET]].copy()
    out[TARGET] = out[TARGET].astype(int)
    return out


def prepare(source: str, path: Path, sample: int | None) -> pd.DataFrame:
    builders = {
        "home_credit": _prepare_home_credit,
        "lending_club": _prepare_lending_club,
        "lending_club_wide": _prepare_lending_club_wide,
        "gmsc": _prepare_gmsc,
    }
    df = builders[source](path)
    if sample is not None and len(df) > sample:
        df = df.sample(n=sample, random_state=42).sort_values(DATE_COL).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="原始 CSV 绝对或相对路径（自动识别 GMSC / Home Credit / LC 宽表等）",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional row cap (e.g. 50000) for faster notebook runs",
    )
    args = parser.parse_args()

    explicit = Path(args.input) if args.input else None
    source, path = _detect_source(explicit)
    df = prepare(source, path, args.sample)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    num_cols = [c for c in df.columns if c not in (TARGET, DATE_COL)]
    meta = (
        f"source={source}\n"
        f"raw_path={path}\n"
        f"rows={len(df)}\n"
        f"features={len(num_cols)}\n"
        f"bad_rate={df[TARGET].mean():.4f}\n"
        f"date_min={df[DATE_COL].min()}\n"
        f"date_max={df[DATE_COL].max()}\n"
    )
    META.write_text(meta, encoding="utf-8")
    print(f"Wrote {OUT} ({len(df)} rows, {len(num_cols)} features) from {source}")
    print(meta)


if __name__ == "__main__":
    main()
