"""Synthetic credit-application data for ProScore tests and demos.

Two profiles:

- **test** (default, 800 rows): edge cases (missing, -999, rare categories) for unit tests.
- **demo** (6000 rows): larger sample, latent-risk design, stable Train/Test KS for notebooks.

Usage::

    python -m tests.synthetic_credit              # writes test_data.csv + demo_scorecard_data.csv
    python tests/generate_test_data.py            # same (legacy entry)
"""

from __future__ import annotations

import argparse
from typing import Literal

import numpy as np
import pandas as pd

Profile = Literal["test", "demo"]


def generate_credit_data(
    n: int = 800,
    *,
    profile: Profile = "test",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic credit DataFrame with ``apply_date`` and ``bad_flag``.

    Parameters
    ----------
    n
        Number of rows.
    profile
        ``test`` keeps the original stress-test quirks; ``demo`` optimises
        for notebook scorecard metrics (larger *n*, stronger signal).
    seed
        Random seed (reproducible).
    """
    rng = np.random.default_rng(seed)

    if profile == "demo":
        return _build_demo(n, rng)
    return _build_test(n, rng)


def _assign_dates(
    rng: np.random.Generator,
    n: int,
    *,
    year_weights: dict[int, float] | None = None,
) -> pd.DatetimeIndex:
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    if year_weights is None:
        year_weights = {2020: 0.22, 2021: 0.22, 2022: 0.28, 2023: 0.28}
    years = np.array(list(year_weights.keys()))
    probs = np.array([year_weights[y] for y in years], dtype=float)
    probs /= probs.sum()
    chosen_years = rng.choice(years, size=n, p=probs)
    out = []
    for yr in chosen_years:
        mask = dates.year == yr
        out.append(rng.choice(dates[mask]))
    return pd.DatetimeIndex(out)


def _build_demo(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Latent-risk design: features driven by ``z``, moderate OOT drift."""
    apply_date = _assign_dates(rng, n)

    # Latent risk (OOT years slightly riskier for monitor demo, not catastrophic)
    z = rng.standard_normal(n)
    year = apply_date.year
    z = z + np.where(year >= 2022, 0.22, 0.0)

    # ── Features (correlated with z) ─────────────────────────────────────
    income = np.exp(10.6 - 0.38 * z + rng.normal(0, 0.35, n))
    income = np.clip(income, 8_000, 400_000)
    income[rng.choice(n, size=max(1, n // 200), replace=False)] = np.nan
    income[rng.choice(n, size=max(1, n // 400), replace=False)] = -999

    debt_ratio = np.clip(0.22 + 0.14 * z + rng.normal(0, 0.06, n), 0.03, 0.92)
    debt_ratio[rng.choice(n, size=max(1, n // 300), replace=False)] = np.nan

    age = np.clip(44 + 6 * z + rng.normal(0, 9, n), 21, 72)
    # U-shape increment on top of z-linked age
    age_u = 0.35 * ((age - 45) / 12) ** 2

    utilization = np.clip(28 + 22 * z + rng.exponential(12, n), 0, 100)
    utilization[rng.choice(n, size=max(1, n // 150), replace=False)] = np.nan

    credit_months = np.clip(130 - 45 * z + rng.normal(0, 35, n), 6, 420)

    num_inquiries = np.clip(rng.poisson(1.5 + np.maximum(0, z + 0.5)), 0, 12).astype(float)
    num_inquiries[rng.choice(n, size=max(1, n // 250), replace=False)] = np.nan

    edu_base = np.array(["high_school", "bachelor", "master", "phd", "other"])
    edu_p = np.array([0.22, 0.45, 0.22, 0.06, 0.05])
    edu_shift = np.clip(-z * 0.15, -0.08, 0.08)
    edu_probs = np.clip(edu_p + edu_shift[:, None], 0.01, None)
    edu_probs = edu_probs / edu_probs.sum(axis=1, keepdims=True)
    education = pd.Series(
        [rng.choice(edu_base, p=edu_probs[i]) for i in range(n)],
        dtype=object,
    )
    education.iloc[rng.choice(n, size=max(1, n // 80), replace=False)] = "missing"

    emp_opts = np.array(["salaried", "self_employed", "unemployed", "retired"])
    emp_p0 = np.array([0.58, 0.22, 0.12, 0.08])
    employment_type = _categorical_from_z(
        rng, n, z, emp_opts, emp_p0, risk_map={
            "salaried": -0.35, "self_employed": 0.05, "unemployed": 0.55, "retired": 0.15,
        },
    )
    employment_type.iloc[rng.choice(n, size=max(1, n // 200), replace=False)] = None

    home_opts = np.array(["mortgage", "own", "rent", "other"])
    home_p0 = np.array([0.45, 0.35, 0.17, 0.03])
    home_ownership = _categorical_from_z(
        rng, n, z, home_opts, home_p0, risk_map={
            "own": -0.3, "mortgage": 0.0, "rent": 0.35, "other": 0.5,
        },
    )

    purpose_opts = np.array(
        ["personal", "auto", "mortgage", "education", "business", "medical"]
    )
    purpose_p0 = np.array([0.28, 0.24, 0.22, 0.10, 0.09, 0.07])
    loan_purpose = _categorical_from_z(
        rng, n, z, purpose_opts, purpose_p0, risk_map={
            "personal": 0.25, "auto": -0.05, "mortgage": -0.25,
            "education": -0.35, "business": 0.15, "medical": 0.0,
        },
    )

    # ── Target (known weights ≈ generation structure) ────────────────────
    dr = np.nan_to_num(debt_ratio, nan=0.28)
    inc_med = float(np.nanmedian(income[income > 0]))
    inc = np.nan_to_num(income, nan=inc_med)
    inc = np.where(inc <= 0, inc_med, inc)
    ut = np.nan_to_num(utilization, nan=35.0)
    logit = (
        -2.05
        + 0.95 * z
        + 0.55 * age_u
        + 1.0 * (dr - 0.28) / 0.12
        - 0.45 * (np.log(inc) - 10.6) / 0.35
        + 0.35 * (ut - 35) / 25
        - 0.40 * (credit_months - 130) / 50
        + 0.30 * (np.nan_to_num(num_inquiries, nan=2.0) - 2) / 2
    )
    logit += _cat_logit(education, {
        "high_school": 0.2, "other": 0.15, "bachelor": 0.0,
        "master": -0.25, "phd": -0.4, "missing": -0.05,
    })
    logit += _cat_logit(employment_type, {
        "salaried": -0.25, "self_employed": 0.05, "unemployed": 0.55, "retired": 0.12,
    })
    logit += _cat_logit(home_ownership, {
        "own": -0.28, "mortgage": 0.0, "rent": 0.32, "other": 0.45,
    })
    logit += _cat_logit(loan_purpose, {
        "personal": 0.2, "auto": -0.05, "mortgage": -0.22,
        "education": -0.3, "business": 0.12, "medical": 0.0,
    })
    logit += rng.normal(0, 0.25, n)

    prob = 1 / (1 + np.exp(-logit))
    prob = np.clip(prob, 0.02, 0.45)
    bad_flag = rng.binomial(1, prob).astype(int)

    df = pd.DataFrame({
        "apply_date": apply_date,
        "income": income,
        "debt_ratio": debt_ratio,
        "age": age,
        "utilization": utilization,
        "credit_months": credit_months,
        "num_inquiries": num_inquiries,
        "education": education,
        "employment_type": employment_type,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "bad_flag": bad_flag,
    })
    return df.sample(frac=1, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)


def _build_test(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Original 800-row stress-test dataset (independent features + composite logit)."""
    apply_date = _assign_dates(
        rng, n, year_weights={2020: 0.25, 2021: 0.25, 2022: 0.25, 2023: 0.25},
    )

    income = np.exp(rng.normal(10.8, 0.5, n))
    income[rng.choice(n, size=30, replace=False)] = np.nan
    income[rng.choice(n, size=10, replace=False)] = -999

    debt_ratio = np.clip(rng.beta(2, 5, n) + rng.normal(0, 0.03, n), 0.01, 0.95)
    debt_ratio[rng.choice(n, size=15, replace=False)] = np.nan

    age = rng.integers(20, 76, n).astype(float)
    age[rng.choice(n, size=5, replace=False)] = np.nan

    utilization = np.clip(rng.exponential(25, n), 0, 100)
    utilization[rng.choice(n, size=60, replace=False)] = np.nan

    credit_months = np.clip(rng.normal(120, 60, n), 6, 480)

    num_inquiries = np.clip(rng.poisson(2, n), 0, 20).astype(float)
    num_inquiries[rng.choice(n, size=25, replace=False)] = np.nan

    edu_opts = ["high_school", "bachelor", "master", "phd", "other"]
    edu_probs = [0.30, 0.42, 0.20, 0.05, 0.03]
    education = pd.Series(rng.choice(edu_opts, n, p=edu_probs), dtype=object)
    education.iloc[rng.choice(n, 35, replace=False)] = "missing"
    education.iloc[rng.choice(n, 10, replace=False)] = None

    emp_opts = ["salaried", "self_employed", "unemployed", "retired"]
    emp_probs = [0.55, 0.25, 0.12, 0.08]
    employment_type = pd.Series(rng.choice(emp_opts, n, p=emp_probs), dtype=object)
    employment_type.iloc[rng.choice(n, 20, replace=False)] = None

    home_opts = ["mortgage", "own", "rent", "other"]
    home_probs = [0.48, 0.32, 0.17, 0.03]
    home_ownership = pd.Series(rng.choice(home_opts, n, p=home_probs), dtype=object)

    purpose_opts = ["personal", "auto", "mortgage", "education", "business", "medical"]
    purpose_probs = [0.30, 0.25, 0.22, 0.10, 0.08, 0.05]
    loan_purpose = pd.Series(rng.choice(purpose_opts, n, p=purpose_probs), dtype=object)
    loan_purpose.iloc[rng.choice(n, 12, replace=False)] = None

    risk = np.zeros(n)
    base_risk = -4.0

    income_clean = np.where(
        (np.isnan(income)) | (income == -999),
        np.nanmedian(income[income > 0]),
        income,
    )
    income_clean = np.clip(income_clean, 1000, 500_000)
    risk += -0.8 * (np.log(income_clean) - 10.8) / 0.5

    dr_clean = np.where(np.isnan(debt_ratio), np.nanmedian(debt_ratio), debt_ratio)
    risk += 1.2 * (dr_clean - 0.25) / 0.15

    age_clean = np.where(np.isnan(age), np.nanmedian(age), age)
    risk += 0.6 * ((age_clean - 45) / 15) ** 2

    ut_clean = np.where(np.isnan(utilization), np.nanmedian(utilization), utilization)
    risk += 0.4 * ut_clean / 30

    cm_clean = np.where(
        np.isnan(credit_months), np.nanmedian(credit_months), credit_months,
    )
    risk += -0.5 * (cm_clean - 120) / 60

    inq_clean = np.where(np.isnan(num_inquiries), 2, num_inquiries)
    risk += 0.5 * (inq_clean - 2) / 3

    for i, e in enumerate(education):
        risk[i] += {"high_school": 0.3, "other": 0.2, "bachelor": 0.0,
                    "master": -0.3, "phd": -0.5, "missing": -0.1}.get(str(e), 0.0)
    for i, e in enumerate(employment_type):
        if isinstance(e, str):
            risk[i] += {"salaried": -0.2, "self_employed": 0.1,
                        "unemployed": 0.8, "retired": 0.3}.get(e, 0.0)
    for i, h in enumerate(home_ownership):
        risk[i] += {"own": -0.3, "mortgage": 0.0, "rent": 0.4, "other": 0.6}.get(h, 0.0)
    for i, p in enumerate(loan_purpose):
        if isinstance(p, str):
            risk[i] += {"personal": 0.3, "auto": -0.1, "mortgage": -0.3,
                        "education": -0.5, "business": 0.2, "medical": 0.0}.get(p, 0.0)

    risk = risk + base_risk
    prob = np.clip(1 / (1 + np.exp(-risk)), 0.005, 0.50)
    bad_flag = rng.binomial(1, prob).astype(int)

    df = pd.DataFrame({
        "apply_date": apply_date,
        "income": income,
        "debt_ratio": debt_ratio,
        "age": age,
        "utilization": utilization,
        "credit_months": credit_months,
        "num_inquiries": num_inquiries,
        "education": education,
        "employment_type": employment_type,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "bad_flag": bad_flag,
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def _categorical_from_z(
    rng: np.random.Generator,
    n: int,
    z: np.ndarray,
    options: np.ndarray,
    base_probs: np.ndarray,
    *,
    risk_map: dict[str, float],
) -> pd.Series:
    """Sample categories with probabilities shifted by latent ``z``."""
    out = []
    risks = np.array([risk_map[o] for o in options])
    for i in range(n):
        logits = np.log(base_probs + 1e-6) + 0.4 * z[i] * risks
        logits = logits - logits.max()
        p = np.exp(logits)
        p /= p.sum()
        out.append(rng.choice(options, p=p))
    return pd.Series(out, dtype=object)


def _cat_logit(series: pd.Series, mapping: dict[str, float]) -> np.ndarray:
    out = np.zeros(len(series))
    for i, val in enumerate(series):
        if isinstance(val, str):
            out[i] = mapping.get(val, 0.0)
    return out


def _print_summary(df: pd.DataFrame, label: str) -> None:
    n = len(df)
    print(f"\n=== {label} ===")
    print(f"Rows: {n}, bad rate: {df['bad_flag'].mean():.2%}")
    print("Bad rate by year:")
    for yr, grp in df.groupby(df["apply_date"].dt.year):
        print(f"  {yr}: {len(grp):>5} rows, bad_rate={grp['bad_flag'].mean():.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ProScore synthetic credit CSVs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_df = generate_credit_data(800, profile="test", seed=args.seed)
    test_df.to_csv("tests/test_data.csv", index=False)
    _print_summary(test_df, "test_data.csv (profile=test)")

    demo_df = generate_credit_data(6000, profile="demo", seed=args.seed)
    demo_df.to_csv("tests/demo_scorecard_data.csv", index=False)
    _print_summary(demo_df, "demo_scorecard_data.csv (profile=demo)")


if __name__ == "__main__":
    main()
