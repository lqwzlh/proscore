"""Correlation and VIF utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from proscore.utils import require_unique_column_labels, require_unique_feature_list


def correlation(
    df: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = 0.7,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Identify highly correlated feature pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (must have unique column labels).
    features : list of str, optional
        Columns to analyse. Defaults to all numeric columns.
    threshold : float
        Correlation threshold for flagging.
    method : str
        Correlation method: ``pearson``, ``spearman``, or ``kendall``.

    Returns
    -------
    pd.DataFrame
        Columns: ``var1 | var2 | corr`` (only pairs above *threshold*).
    """
    require_unique_column_labels(df)
    feats = features or df.select_dtypes(include=[np.number]).columns.tolist()
    if features is not None:
        require_unique_feature_list(features, arg_name="features")
    feats = [c for c in feats if df[c].count() >= 2]
    if len(feats) < 2:
        return pd.DataFrame(columns=["var1", "var2", "corr"])
    corr_matrix = df[feats].corr(method=method).abs()

    pairs: list[dict] = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            val = corr_matrix.iloc[i, j]
            if np.isnan(val):
                continue
            if val >= threshold:
                pairs.append({"var1": feats[i], "var2": feats[j], "corr": round(val, 4)})

    result = pd.DataFrame(pairs)
    if len(result) > 0:
        result = result.sort_values("corr", ascending=False).reset_index(drop=True)
    return result


def vif(
    df: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = 10.0,
) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (must have unique column labels).
    features : list of str, optional
        Columns to evaluate. Defaults to all numeric columns.
    threshold : float
        VIF threshold for flagging high multicollinearity.

    Returns
    -------
    pd.DataFrame
        Columns: ``variable | vif | flag`` where flag is ``"high"`` if
        VIF > *threshold*, else ``"ok"``.
    """
    require_unique_column_labels(df)
    feats = features or df.select_dtypes(include=[np.number]).columns.tolist()
    if features is not None:
        require_unique_feature_list(features, arg_name="features")
    X = df[feats].dropna()

    if len(X) < 2:
        return pd.DataFrame(columns=["variable", "vif", "flag"])

    # Drop constant columns that would break statsmodels
    valid_feats = [c for c in feats if X[c].nunique() > 1]
    if len(valid_feats) < 1:
        return pd.DataFrame(columns=["variable", "vif", "flag"])

    X = X[valid_feats]
    X_const = add_constant(X)
    rows: list[dict] = []
    for col in valid_feats:
        try:
            idx = list(X_const.columns).index(col)
            vif_val = float(variance_inflation_factor(X_const.values, idx))
        except (np.linalg.LinAlgError, ValueError):
            vif_val = float("inf")
        rows.append(
            {
                "variable": col,
                "vif": round(vif_val, 4),
                "flag": "high" if vif_val > threshold else "ok",
            }
        )
    result = pd.DataFrame(rows)
    result = result.sort_values("vif", ascending=False).reset_index(drop=True)
    return result
