"""PipelineSpec — compact parameter injection for the ProScore chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineSpec:
    """Collect all pipeline parameters in one place for :meth:`ProScore.apply`.

    Every field is optional.  Parameters provided here are used as defaults;
    explicit kwargs on the chain methods still take precedence.

    See also: ``docs/使用指南/pipeline-spec.md``.

    Usage::

        spec = PipelineSpec(
            prefilter={"max_corr": 0.8, "iv_range": None},
            binning={"method": "chi", "n_bins": 5},
            refine={"iv_range": (0.02, None), "max_psi": 0.25},
            rules={"method": "exhaustive", "min_lift": 3.0},
            select={"n_min": 5, "n_max": 12, "pvalue_threshold": 0.05},
            model={"odds": 20, "pdo": 20, "base_score": 600},
        )

        ps = ProScore()
        ps.read(train=df_train, test=df_test, target="bad_flag") \\
          .apply(spec) \\
          .detect().prefilter().bin().refine().mine_rules() \\
          .transform().select().fit().scorecard().evaluate()
    """

    prefilter: dict[str, Any] = field(default_factory=dict)
    binning: dict[str, Any] = field(default_factory=dict)
    refine: dict[str, Any] = field(default_factory=dict)
    rules: dict[str, Any] = field(default_factory=dict)
    select: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)

    def merge(self, **overrides: dict[str, Any]) -> PipelineSpec:
        """Return a new spec with *overrides* applied on top."""
        new_kwargs = {
            k: {**getattr(self, k, {}), **v}
            for k, v in overrides.items()
        }
        return PipelineSpec(**{**self.__dict__, **new_kwargs})
