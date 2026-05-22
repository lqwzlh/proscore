"""Feature screening outcomes — warn instead of error when no variables remain."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence


class FeatureScreenWarning(UserWarning):
    """Emitted when a screening stage retains no features (normal business case)."""


@dataclass(frozen=True)
class ScreenOutcome:
    """Result of :func:`assess_screen`.

    Attributes:
        ok: Whether at least *min_required* features remain.
        stage: Human-readable stage label (e.g. ``"粗筛"``).
        message: Empty when *ok*; otherwise guidance for analysts.
        n_candidates: Number of candidates entering the stage.
        n_selected: Number of features retained.
    """

    ok: bool
    stage: str
    message: str
    n_candidates: int
    n_selected: int


def assess_screen(
    support: Sequence[str],
    *,
    stage: str,
    n_candidates: int | None = None,
    min_required: int = 1,
) -> ScreenOutcome:
    """Check whether a screening stage retained enough features.

    Does **not** raise — emits :class:`FeatureScreenWarning` when empty.
    Use the returned :class:`ScreenOutcome` to skip downstream steps or
    to populate reports.

    Parameters
    ----------
    support : sequence of str
        Feature names retained (e.g. ``filter.support_``).
    stage : str
        Label for messages and reports.
    n_candidates : int, optional
        Candidates before screening; defaults to ``len(support)``.
    min_required : int
        Minimum acceptable retained count (default 1).

    Returns
    -------
    ScreenOutcome
    """
    n_out = len(support)
    n_in = n_candidates if n_candidates is not None else n_out
    if n_out >= min_required:
        return ScreenOutcome(
            ok=True,
            stage=stage,
            message="",
            n_candidates=n_in,
            n_selected=n_out,
        )
    msg = (
        f"【{stage}】{n_in} 个候选变量均未保留（剩余 0 个）。"
        "此为正常筛选结果，请放宽阈值或检查数据质量。"
        "可继续生成探查/筛选报告，无需执行后续建模步骤。"
    )
    warnings.warn(msg, FeatureScreenWarning, stacklevel=2)
    return ScreenOutcome(
        ok=False,
        stage=stage,
        message=msg,
        n_candidates=n_in,
        n_selected=n_out,
    )
