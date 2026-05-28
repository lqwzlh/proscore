"""Model evaluation: KS, AUC, PSI, and multi-period OOT metrics."""

from __future__ import annotations

import importlib

_metrics = importlib.import_module("proscore.evaluate._metrics")
_diagnose = importlib.import_module("proscore.evaluate._diagnose")

evaluate = _metrics.evaluate
evaluate_by_period = getattr(_metrics, "evaluate_by_period", None)
diagnose = _diagnose.diagnose
DiagnosisReport = _diagnose.DiagnosisReport
DiagnosisIssue = _diagnose.DiagnosisIssue
DEFAULT_THRESHOLDS = _diagnose.DEFAULT_THRESHOLDS

if evaluate_by_period is None:
    raise ImportError(
        "proscore.evaluate._metrics is missing evaluate_by_period. "
        "Use the project source under src/ and restart the Jupyter kernel "
        "(Kernel → Restart) to clear cached imports."
    )

__all__ = ["evaluate", "evaluate_by_period", "diagnose", "DiagnosisReport", "DiagnosisIssue", "DEFAULT_THRESHOLDS"]
