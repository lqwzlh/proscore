"""Decision rule mining for credit scorecard strategy development.

Mines single-variable and cross-variable rules from binned candidate features
with Lift / Precision / Recall / Hit-Rate evaluation.
"""

from proscore.rules._miner import RuleMiner

__all__ = ["RuleMiner"]
