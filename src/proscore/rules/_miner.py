"""Rule miner: exhaustive, decision-tree, and Apriori-based rule search."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


@dataclass
class RuleRecord:
    """A single mined rule with evaluation metrics."""
    rule: str
    hit_count: int
    bad_count: int
    good_count: int
    hit_rate: float
    precision: float
    recall: float
    lift: float
    single_hit_count: int = 0
    single_hit_rate: float = 0.0


class RuleMiner:
    """Mine decision rules from binned numeric features.

    - ``"exhaustive"`` (default): try every 1–3 feature combination within bin edges.
    - ``"tree"``: fit a decision tree per feature; each leaf is a rule.
    - ``"apriori"``: find top single-variable rules, then cross only the winners.
      Note: Phase 1 retains all single-variable rules that pass thresholds; only
      cross-rule candidates are limited to top-10 by Lift.

    Parameters
    ----------
    method : str
        ``"exhaustive"`` (default), ``"tree"``, or ``"apriori"``.
    max_depth : int
        Maximum number of features to cross (exhaustive / apriori modes).
    max_tree_depth : int
        Maximum depth of the decision tree (tree mode).
    min_lift : float
        Minimum Lift (precision / overall_bad_rate).
    min_hit_rate : float
        Minimum fraction of total samples a rule must cover.
    max_hit_rate : float
        Maximum fraction of total samples (limit over-rejection).
    max_rules : int
        Maximum number of rules to return (by descending Lift).
    random_state : int or None
        Seed for decision tree reproducibility (tree mode only).
    """

    def __init__(
        self,
        method: str = "exhaustive",
        max_depth: int = 3,
        max_tree_depth: int = 4,
        min_lift: float = 3.0,
        min_hit_rate: float = 0.01,
        max_hit_rate: float = 0.20,
        max_rules: int = 20,
        random_state: int | None = 42,
    ):
        _valid = {"exhaustive", "tree", "apriori"}
        if method not in _valid:
            raise ValueError(f"Unknown method: {method!r}. Valid: {sorted(_valid)}")
        self.method = method
        self.max_depth = max_depth
        self.max_tree_depth = max_tree_depth
        self.min_lift = min_lift
        self.min_hit_rate = min_hit_rate
        self.max_hit_rate = max_hit_rate
        self.max_rules = max_rules
        self.random_state = random_state

        self._rules: list[RuleRecord] = []
        self._overall_bad_rate: float = 0.0
        self._bin_edges: dict[str, list[float]] = {}

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        bin_table: dict | None = None,
    ) -> RuleMiner:
        """Mine rules from *X* (raw features) and *y* (binary target, 1=bad)."""
        y_arr = np.asarray(y).ravel()
        self._validate_target(y_arr)
        self._overall_bad_rate = float(y_arr.mean())

        missing_rules: list[RuleRecord] = []
        if bin_table:
            self._bin_edges = self._edges_from_bin_table(bin_table, list(X.columns))
            missing_rules = self._missing_rules(bin_table, X, y_arr)

        if self.method == "exhaustive":
            mined_rules = self._mine_exhaustive(X, y_arr)
        elif self.method == "tree":
            mined_rules = self._mine_tree(X, y_arr)
        else:
            mined_rules = self._mine_apriori(X, y_arr)

        self._rules = missing_rules + mined_rules

        # Sort by lift descending
        self._rules.sort(key=lambda r: r.lift, reverse=True)

        # Compute single-hit BEFORE truncating (so redundancy is visible)
        if self._rules:
            self._compute_single_hit(X, y_arr)

        # Truncate
        self._rules = self._rules[: self.max_rules]

        return self

    @staticmethod
    def _validate_target(y: np.ndarray) -> None:
        unique = set(np.unique(y))
        expected = {0, 1}
        if not unique <= expected:
            warnings.warn(
                f"Target y has values {sorted(unique)}, expected {{0, 1}}. "
                f"Results may be invalid.",
                stacklevel=3,
            )

    # ── exhaustive search ────────────────────────────────────────────────────

    def _mine_exhaustive(self, X: pd.DataFrame, y: np.ndarray) -> list[RuleRecord]:
        """Try every 1–N feature combination up to max_depth."""
        features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if not features:
            return []

        all_rules: list[RuleRecord] = []

        # Single-feature
        all_rules.extend(self._single_var_rules(X, y, features))

        # Depth 2
        if self.max_depth >= 2 and len(features) >= 2:
            all_rules.extend(self._cross_var_rules(X, y, features))

        # Depth 3
        if self.max_depth >= 3 and len(features) >= 3:
            for f1, f2, f3 in combinations(features, 3):
                e1 = self._bin_edges.get(f1, self._auto_edges(X[f1], 5))
                e2 = self._bin_edges.get(f2, self._auto_edges(X[f2], 5))
                e3 = self._bin_edges.get(f3, self._auto_edges(X[f3], 5))
                for l1, h1 in zip(e1[:-1], e1[1:]):
                    m1 = (X[f1] > l1) & (X[f1] <= h1)
                    if not m1.any():
                        continue
                    for l2, h2 in zip(e2[:-1], e2[1:]):
                        m2 = m1 & (X[f2] > l2) & (X[f2] <= h2)
                        if not m2.any():
                            continue
                        for l3, h3 in zip(e3[:-1], e3[1:]):
                            mask = m2 & (X[f3] > l3) & (X[f3] <= h3)
                            if not mask.any():
                                continue
                            rule = self._eval_rule(
                                mask, y, len(y),
                                [(f1, l1, h1), (f2, l2, h2), (f3, l3, h3)],
                            )
                            if rule:
                                all_rules.append(rule)

        return all_rules

    # ── shared single-var & cross-var helpers ────────────────────────────────

    def _single_var_rules(
        self, X: pd.DataFrame, y: np.ndarray, features: list[str]
    ) -> list[RuleRecord]:
        """Single-feature rules (used by exhaustive & apriori)."""
        rules: list[RuleRecord] = []
        total = len(y)
        for f in features:
            edges = self._bin_edges.get(f, self._auto_edges(X[f], 5))
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (X[f] > lo) & (X[f] <= hi)
                if not mask.any():
                    continue
                rule = self._eval_rule(mask, y, total, [(f, lo, hi)])
                if rule:
                    rules.append(rule)
        return rules

    def _cross_var_rules(
        self, X: pd.DataFrame, y: np.ndarray, features: list[str]
    ) -> list[RuleRecord]:
        """Two-feature cross rules."""
        rules: list[RuleRecord] = []
        total = len(y)
        for f1, f2 in combinations(features, 2):
            e1 = self._bin_edges.get(f1, self._auto_edges(X[f1], 5))
            e2 = self._bin_edges.get(f2, self._auto_edges(X[f2], 5))
            for l1, h1 in zip(e1[:-1], e1[1:]):
                m1 = (X[f1] > l1) & (X[f1] <= h1)
                if not m1.any():
                    continue
                for l2, h2 in zip(e2[:-1], e2[1:]):
                    mask = m1 & (X[f2] > l2) & (X[f2] <= h2)
                    if not mask.any():
                        continue
                    rule = self._eval_rule(mask, y, total, [(f1, l1, h1), (f2, l2, h2)])
                    if rule:
                        rules.append(rule)
        return rules

    # ── decision tree ────────────────────────────────────────────────────────

    def _mine_tree(self, X: pd.DataFrame, y: np.ndarray) -> list[RuleRecord]:
        """Single-feature decision tree: each leaf is a rule."""
        features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if not features:
            return []

        total = len(y)
        all_rules: list[RuleRecord] = []

        for f in features:
            Xf = X[[f]].fillna(X[f].median())
            dt = DecisionTreeClassifier(
                max_depth=self.max_tree_depth,
                min_samples_leaf=max(20, int(total * 0.01)),
                random_state=self.random_state,
            )
            dt.fit(Xf, y)
            # Non-leaf thresholds define split points
            thresholds = sorted(set(
                dt.tree_.threshold[dt.tree_.children_left != -1]
            ))
            if not thresholds:
                continue
            edges = [-np.inf] + thresholds + [np.inf]
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (X[f] > lo) & (X[f] <= hi)
                if not mask.any():
                    continue
                rule = self._eval_rule(mask, y, total, [(f, lo, hi)])
                if rule:
                    all_rules.append(rule)

        return all_rules

    # ── apriori (simplified) ─────────────────────────────────────────────────

    def _mine_apriori(self, X: pd.DataFrame, y: np.ndarray) -> list[RuleRecord]:
        """Apriori-style: find top single rules, then cross only the winners."""
        features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if not features:
            return []

        total = len(y)
        # Phase 1: single-var (reuse shared helper, then filter to top-N)
        all_singles = self._single_var_rules(X, y, features)
        # Keep top-10 by lift for crossing
        all_singles.sort(key=lambda r: r.lift, reverse=True)
        top_singles = all_singles[:10]

        all_rules = list(all_singles)

        # Phase 2: cross top singles (depth 2)
        if self.max_depth >= 2 and len(top_singles) >= 2:
            for r1, r2 in combinations(top_singles, 2):
                m1 = self._rule_mask(r1.rule, X)
                m2 = self._rule_mask(r2.rule, X)
                mask = m1 & m2
                if not mask.any():
                    continue
                # Extract feature/bound info from the two original rules
                p1, p2 = self._parse_feat_bounds(r1.rule, r2.rule)
                rule = self._eval_rule(mask, y, total, p1 + p2)
                if rule:
                    all_rules.append(rule)

        return all_rules

    # ── rule evaluation helper ───────────────────────────────────────────────

    def _eval_rule(
        self,
        mask: pd.Series | np.ndarray,
        y: np.ndarray,
        total: int,
        feat_bounds: list[tuple[str, float, float]],
    ) -> RuleRecord | None:
        """Evaluate a rule and return a RuleRecord if it passes all filters.

        *feat_bounds* is a list of ``(feature, lo, hi)`` tuples.
        """
        hit = int(np.sum(mask))
        hit_rate = hit / total

        if hit_rate < self.min_hit_rate or hit_rate > self.max_hit_rate:
            return None

        mask_arr = np.asarray(mask, dtype=bool)
        bad = int(y[mask_arr].sum())
        good = hit - bad
        precision = bad / hit if hit > 0 else 0.0
        recall = bad / int(y.sum()) if y.sum() > 0 else 0.0
        lift = precision / self._overall_bad_rate if self._overall_bad_rate > 0 else 1.0

        if lift < self.min_lift:
            return None

        rstr = self._format_rule(feat_bounds)

        return RuleRecord(
            rule=rstr,
            hit_count=hit,
            bad_count=bad,
            good_count=good,
            hit_rate=round(hit_rate, 6),
            precision=round(precision, 6),
            recall=round(recall, 6),
            lift=round(lift, 4),
        )

    # ── rule string formatting ───────────────────────────────────────────────

    @staticmethod
    def _format_rule(feat_bounds: list[tuple[str, float, float]]) -> str:
        """Build a human-readable rule string from (feat, lo, hi) tuples."""
        parts = []
        for feat, lo, hi in feat_bounds:
            if np.isneginf(lo):
                parts.append(f"{feat} <= {hi:.2f}")
            elif np.isposinf(hi):
                parts.append(f"{feat} > {lo:.2f}")
            else:
                parts.append(f"{feat} in ({lo:.2f}, {hi:.2f}]")
        return " AND ".join(parts)

    @staticmethod
    def _parse_feat_bounds(
        rule1: str, rule2: str
    ) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]:
        """Parse two rule strings into feat-bound tuples (used by apriori cross)."""
        def _parse_one(r: str) -> list[tuple[str, float, float]]:
            result = []
            for part in r.split(" AND "):
                part = part.strip()
                if " is missing" in part:
                    feat = part.replace(" is missing", "").strip()
                    result.append((feat, -np.inf, np.inf))
                elif " <= " in part:
                    feat, bound = part.split(" <= ")
                    result.append((feat.strip(), -np.inf, float(bound.strip())))
                elif " > " in part:
                    feat, bound = part.split(" > ")
                    result.append((feat.strip(), float(bound.strip()), np.inf))
                else:
                    feat, rest = part.split(" in ", 1)
                    inner = rest.strip("()[]")
                    lo_str, hi_str = inner.split(",")
                    result.append((
                        feat.strip(),
                        float(lo_str.strip()),
                        float(hi_str.strip()),
                    ))
            return result

        p1 = _parse_one(rule1)
        p2 = _parse_one(rule2)
        return p1, p2

    # ── single-hit computation ───────────────────────────────────────────────

    def _compute_single_hit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Assign single-hit counts so redundant rules can be spotted."""
        total = len(y)
        for i, rule in enumerate(self._rules):
            other_hit = np.zeros(total, dtype=bool)
            for j, r2 in enumerate(self._rules):
                if i == j:
                    continue
                other_hit |= self._rule_mask(r2.rule, X)
            rule_mask = self._rule_mask(rule.rule, X)
            single = int((rule_mask & ~other_hit).sum())
            rule.single_hit_count = single
            rule.single_hit_rate = round(single / total, 6)

    def _rule_mask(self, rule_str: str, X: pd.DataFrame) -> np.ndarray:
        """Parse a rule string and return a boolean mask."""
        parts = rule_str.split(" AND ")
        mask = np.ones(len(X), dtype=bool)
        for part in parts:
            part = part.strip()
            if " is missing" in part:
                feat = part.replace(" is missing", "").strip()
                mask &= X[feat.strip()].isna()
            elif " <= " in part:
                feat, bound = part.split(" <= ")
                mask &= X[feat.strip()] <= float(bound.strip())
            elif " > " in part:
                feat, bound = part.split(" > ")
                mask &= X[feat.strip()] > float(bound.strip())
            else:
                feat, rest = part.split(" in ", 1)
                inner = rest.strip("()[]")
                lo_str, hi_str = inner.split(",")
                mask &= (X[feat.strip()] > float(lo_str.strip())) & (
                    X[feat.strip()] <= float(hi_str.strip())
                )
        return mask

    # ── bin edge helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _edges_from_bin_table(
        bin_table: dict, features: list[str]
    ) -> dict[str, list[float]]:
        """Extract cut-off edges from Binning.bin_table_ per feature."""
        edges = {}
        for f in features:
            if f in bin_table:
                bt = bin_table[f]
                cuts = list(getattr(bt, "cutoffs", []))
                if cuts:
                    edges[f] = [-np.inf] + sorted(cuts) + [np.inf]
        return edges

    @staticmethod
    def _auto_edges(series: pd.Series, n_bins: int = 5) -> list[float]:
        """Equal-frequency bin edges when no bin_table is available."""
        clean = series.dropna()
        if len(clean) < n_bins:
            return [-np.inf, np.inf]
        try:
            q = pd.qcut(clean, q=n_bins, duplicates="drop", retbins=True)[1]
            q[0] = -np.inf
            q[-1] = np.inf
            return list(q)
        except Exception:
            warnings.warn(
                f"Feature '{series.name}': could not compute equal-frequency bins. "
                f"Using a single bin (all rules from this feature will be skipped).",
                stacklevel=3,
            )
            return [-np.inf, np.inf]

    # ── missing-bin rules ────────────────────────────────────────────────────

    def _missing_rules(
        self,
        bin_table: dict,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> list[RuleRecord]:
        """Generate 'feat is missing' rules from bin_table has_missing markers."""
        rules: list[RuleRecord] = []
        total = len(y)
        for f, bt in bin_table.items():
            if not getattr(bt, "has_missing", False):
                continue
            if f not in X.columns:
                continue
            mask = X[f].isna()
            if not mask.any():
                continue
            rule = self._eval_rule(mask, y, total, [(f, -np.inf, np.inf)])
            if rule:
                rules.append(RuleRecord(
                    rule=f"{f} is missing",
                    hit_count=rule.hit_count,
                    bad_count=rule.bad_count,
                    good_count=rule.good_count,
                    hit_rate=rule.hit_rate,
                    precision=rule.precision,
                    recall=rule.recall,
                    lift=rule.lift,
                ))
        return rules

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def rules_table_(self) -> pd.DataFrame:
        """Evaluation table, sorted by Lift descending."""
        if not self._rules:
            return pd.DataFrame(
                columns=["rule", "hit_count", "bad_count", "good_count",
                         "hit_rate", "precision", "recall", "lift",
                         "single_hit_count", "single_hit_rate"]
            )
        return pd.DataFrame([r.__dict__ for r in self._rules])

    @property
    def used_features_(self) -> list[str]:
        """Feature names referenced by any mined rule."""
        feats: set[str] = set()
        for r in self._rules:
            for part in r.rule.split(" AND "):
                part = part.strip()
                if " is missing" in part:
                    # Column names containing " is missing" are exceedingly rare; safe in practice.
                    feats.add(part.replace(" is missing", "").strip())
                    continue
                for sep in (" <= ", " > ", " in "):
                    if sep in part:
                        feats.add(part.split(sep)[0].strip())
                        break
        return sorted(feats)

    # ── cumulative coverage ──────────────────────────────────────────────────

    def coverage_report(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        """Cumulative recall as rules are stacked top-to-bottom.

        Returns a DataFrame with columns ``rule, cum_hit, cum_recall, cum_bad_rate``.
        """
        y_arr = np.asarray(y).ravel()
        total_bad = int(y_arr.sum())
        total = len(y)
        hit_union = np.zeros(total, dtype=bool)

        rows = []
        for r in self._rules:
            hit_union |= self._rule_mask(r.rule, X)
            rows.append({
                "rule": r.rule,
                "cum_hit": int(hit_union.sum()),
                "cum_recall": round(
                    float((hit_union & y_arr.astype(bool)).sum()) / total_bad, 6
                ) if total_bad > 0 else 0.0,
                "cum_bad_rate": round(
                    float(y_arr[hit_union].mean()), 6
                ) if hit_union.any() else 0.0,
            })

        return pd.DataFrame(rows)
