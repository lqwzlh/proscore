"""Read variable presets from an Excel spreadsheet.

Requires ``openpyxl`` (install with ``pip install proscore[excel]``).

The Excel file must contain at least one sheet named ``"variables"`` with columns:

    - ``variable`` (required): column name in the DataFrame.
    - ``monotonic`` (optional): ``"increasing"`` / ``"decreasing"`` / ``"u"`` /
      ``"inverted_u"`` — mapped to ``BinningProcess.feature_config``.
    - ``special_values`` (optional): comma-separated numbers, e.g.
      ``"-999, -998"`` — mapped to ``BinningProcess.feature_config``.
    - ``dimension`` (optional): business dimension label, e.g.
      ``"debt burden"`` — mapped to ``StepwiseSelector.feature_belong``.

An optional second sheet ``"dimensions"`` (free text) is ignored at runtime.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import pandas as pd

# Valid monotonic strings accepted from the spreadsheet
_ALLOWED_MONOTONIC = frozenset({"increasing", "decreasing", "u", "inverted_u"})


class PresetResult(NamedTuple):
    """Return value of :func:`load_presets`.

    Attributes:
        feature_config: Mapping ``{variable: {...}}`` ready for
            ``BinningProcess(feature_config=...)``.
        feature_belong: Mapping ``{source: [variables]}`` ready for
            ``StepwiseSelector(feature_belong=...)``.
    """

    feature_config: dict[str, dict[str, object]]
    feature_belong: dict[str, list[str]]


def load_presets(
    path: str,
    *,
    sheet_name: str = "variables",
) -> PresetResult:
    """Load variable presets from an Excel file.

    Requires ``openpyxl`` — install with ``pip install proscore[excel]``.

    Args:
        path: Path to the ``.xlsx`` / ``.xls`` file.
        sheet_name: Name of the worksheet to read (default ``"variables"``).

    Returns:
        :class:`PresetResult` with ``feature_config`` and ``feature_belong``.

    Raises:
        ImportError: If ``openpyxl`` is not installed.
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError(
            "openpyxl is required to read Excel presets. "
            "Install it with: pip install proscore[excel]"
        ) from None

    presets: pd.DataFrame = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")

    # ---- feature_config -------------------------------------------------------
    feature_config: dict[str, dict[str, object]] = {}
    has_mono = "monotonic" in presets.columns
    has_spec = "special_values" in presets.columns

    for _, row in presets.iterrows():
        if pd.isna(row.get("variable")):
            warnings.warn("Skipping row with missing variable name in presets", stacklevel=2)
            continue
        var: str = str(row["variable"])
        cfg: dict[str, object] = {}

        if has_mono:
            mono = row.get("monotonic")
            if pd.notna(mono):
                mono_str = str(mono).strip()
                if mono_str and mono_str not in _ALLOWED_MONOTONIC:
                    warnings.warn(
                        f"Variable {var!r}: monotonic={mono_str!r} is not "
                        f"recognised. Expected one of {sorted(_ALLOWED_MONOTONIC)}. "
                        f"Constraint will be ignored.",
                        stacklevel=2,
                    )
                else:
                    cfg["monotonic"] = mono_str

        if has_spec:
            spec = row.get("special_values")
            if pd.notna(spec):
                raw = str(spec).strip()
                if raw:
                    vals = [
                        pd.to_numeric(v.strip(), errors="coerce")
                        for v in raw.split(",")
                    ]
                    vals = [v for v in vals if not (isinstance(v, float) and pd.isna(v))]
                    if vals:
                        cfg["special_values"] = vals

        if cfg:
            feature_config[var] = cfg

    # ---- feature_belong -------------------------------------------------------
    feature_belong: dict[str, list[str]] = {}
    has_dim = "dimension" in presets.columns

    if has_dim:
        for _, row in presets.iterrows():
            dim = row.get("dimension")
            if pd.isna(dim):
                continue
            dim = str(dim).strip()
            if not dim:
                continue
            feature_belong.setdefault(dim, []).append(str(row["variable"]))

    if not feature_config and not feature_belong:
        warnings.warn(
            f"No monotonic/special_values/dimension columns found in sheet "
            f"{sheet_name!r}. Returning empty configs.",
            stacklevel=2,
        )

    return PresetResult(feature_config=feature_config, feature_belong=feature_belong)
