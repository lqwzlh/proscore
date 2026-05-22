"""Shared fixtures for ProScore tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proscore.utils import load_presets

HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."


# ── Synthetic small data (fast, deterministic) ──────────────────────────────


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Small 200-row DataFrame with known monotonic relationships."""
    rng = np.random.default_rng(1234)
    n = 200

    x1 = np.concatenate([
        rng.normal(0, 1, 50),  # low risk
        rng.normal(2, 1, 50),
        rng.normal(4, 1, 50),
        rng.normal(6, 1, 50),  # high risk
    ])
    x2 = rng.normal(3, 1.5, n)
    x3 = rng.choice(["A", "B", "C"], size=n, p=[0.3, 0.5, 0.2])
    x4 = rng.normal(50, 15, n)
    x4[:5] = np.nan  # ~2.5% missing

    # Generate target: bad rate increases with x1
    prob = 1 / (1 + np.exp(-(-2.5 + 0.5 * (x1 - 3))))
    y = rng.binomial(1, prob)

    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "target": y.astype(int),
    })


# ── Full synthetic data (800 rows, from test_data.csv) ─────────────────────


@pytest.fixture(scope="session")
def full_df() -> pd.DataFrame:
    """The 800-row synthetic test dataset."""
    return pd.read_csv(f"{HERE}/test_data.csv", parse_dates=["apply_date"])


# ── Train / test split ──────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def train_test(full_df: pd.DataFrame):
    """Split full data by year: 2020-2021 train, 2022 test, 2023 oot."""
    train = full_df[full_df["apply_date"].dt.year <= 2021].copy()
    test = full_df[full_df["apply_date"].dt.year == 2022].copy()
    oot = full_df[full_df["apply_date"].dt.year == 2023].copy()
    return train, test, oot


# ── Feature lists ───────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def features(full_df: pd.DataFrame) -> list[str]:
    """All feature columns (excludes target, date)."""
    return [c for c in full_df.columns if c not in ("bad_flag", "apply_date")]


@pytest.fixture(scope="session")
def num_features(full_df: pd.DataFrame) -> list[str]:
    """Numeric feature columns."""
    return [
        c for c in full_df.columns
        if c not in ("bad_flag", "apply_date")
        and pd.api.types.is_numeric_dtype(full_df[c])
    ]


# ── Presets ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def presets():
    """Load presets from the bundled Excel file."""
    path = f"{HERE}/variable_presets.xlsx"
    try:
        return load_presets(path)
    except ImportError:
        pytest.skip("openpyxl not installed — pip install proscore[excel]")


# ── Binning result (on full data) ──────────────────────────────────────────


@pytest.fixture(scope="session")
def binning_result(full_df: pd.DataFrame, features: list[str]):
    """Fitted Binning on full data, chi method, 5 bins."""
    from proscore.binning import Binning

    b = Binning(method="chi", n_bins=5, monotonic=None, adjust_shape=True)
    b.fit(full_df[features + ["bad_flag"]], y="bad_flag")
    return b


# ── WOE-transformed data ───────────────────────────────────────────────────


@pytest.fixture(scope="session")
def woe_transformer(binning_result):
    """Fitted WOETransformer from full-data binning."""
    from proscore.transform import WOETransformer

    wt = WOETransformer(unseen_strategy="worst")
    wt.fit(binning_result.bin_table_)
    return wt


@pytest.fixture(scope="session")
def woe_train_test(train_test, woe_transformer, num_features):
    """WOE-transformed train/test data (includes target column)."""
    train, test, oot = train_test

    def _woe(df):
        X = df[num_features].copy()
        result = woe_transformer.transform(X.fillna(0))
        result["bad_flag"] = df["bad_flag"].values
        return result

    return _woe(train), _woe(test), _woe(oot)
