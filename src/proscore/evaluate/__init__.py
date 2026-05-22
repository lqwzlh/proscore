"""Model evaluation: KS, AUC, PSI, and multi-period OOT metrics."""

from __future__ import annotations

import importlib

_metrics = importlib.import_module("proscore.evaluate._metrics")

evaluate = _metrics.evaluate
evaluate_by_period = getattr(_metrics, "evaluate_by_period", None)

if evaluate_by_period is None:
    raise ImportError(
        "proscore.evaluate._metrics is missing evaluate_by_period. "
        "Use the project source under src/ and restart the Jupyter kernel "
        "(Kernel → Restart) to clear cached imports."
    )

__all__ = ["evaluate", "evaluate_by_period"]
