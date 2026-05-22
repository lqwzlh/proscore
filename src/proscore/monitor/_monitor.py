"""Model monitor: baseline snapshots, periodic tracking, rule-based alerting."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from proscore.utils._psi import psi_from_distributions


# ── default alert rules ───────────────────────────────────────────────────

_DEFAULT_RULES: list[dict] = [
    {"metric": "psi_score", "op": ">", "value": 0.10, "level": "warn",
     "message": "分数 PSI={value:.3f}，分布轻微漂移"},
    {"metric": "psi_score", "op": ">", "value": 0.25, "level": "critical",
     "message": "分数 PSI={value:.3f}，分布显著漂移，建议重训"},
    {"metric": "psi_feature", "op": ">", "value": 0.25, "level": "warn",
     "message": "变量 {name} PSI={value:.3f}，分布漂移"},
    {"metric": "ks_decay", "op": ">", "value": 0.15, "level": "warn",
     "message": "KS 衰退 {value:.1%}，区分力下降"},
    {"metric": "ks_decay", "op": ">", "value": 0.30, "level": "critical",
     "message": "KS 衰退 {value:.1%}，区分力严重下降，建议重训"},
]


# ── dataclasses ───────────────────────────────────────────────────────────


@dataclass
class MonitorSnapshot:
    """Single monitoring observation (persisted unit)."""
    period: str
    timestamp: str
    n_samples: int
    bad_rate: float
    ks: float | None = None
    auc: float | None = None
    psi_score: float = 0.0
    psi_features: dict[str, float] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)
    status: str = "ok"


@dataclass
class MonitorResult(MonitorSnapshot):
    """Returned by :meth:`ModelMonitor.track` — snapshot + trend info."""
    ks_decay: float | None = None
    auc_decay: float | None = None
    psi_score_trend: list[float] = field(default_factory=list)
    recommendation: str = ""


# ── rule engine ───────────────────────────────────────────────────────────


def _evaluate_rules(
    metrics: dict[str, float],
    rules: list[dict],
    feature_psi: dict[str, float] | None = None,
) -> list[str]:
    """Evaluate alert rules against a metrics dict and per-feature PSI dict.

    Returns a list of formatted alert messages (empty if no rules fire).
    """
    alerts: list[str] = []
    for rule in rules:
        metric = rule["metric"]
        op = rule["op"]
        threshold = rule["value"]
        level = rule.get("level", "warn")
        msg_tmpl = rule.get("message", "")

        if metric == "psi_feature" and feature_psi:
            for fname, pval in feature_psi.items():
                if _op_compare(pval, op, threshold):
                    msg = msg_tmpl.format(name=fname, value=pval) if msg_tmpl else ""
                    alerts.append(f"[{level}] {msg or f'Feature {fname} PSI={pval:.3f}'}")
        elif metric in metrics:
            val = metrics[metric]
            if val is not None and _op_compare(val, op, threshold):
                msg = msg_tmpl.format(value=val) if msg_tmpl else ""
                alerts.append(f"[{level}] {msg or f'{metric}={val:.3f}'}")
    return alerts


def _op_compare(a: float, op: str, b: float) -> bool:
    """Apply a comparison operator string to two numeric values."""
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == "==":
        return abs(a - b) < 1e-9
    return False


def _status_from_alerts(alerts: list[str]) -> str:
    """Derive an overall status from alert messages."""
    if any("critical" in a for a in alerts):
        return "critical"
    if any("warn" in a for a in alerts):
        return "warn"
    return "ok"


def _build_recommendation(status: str, alerts: list[str]) -> str:
    """Produce a natural-language recommendation string."""
    if status == "ok":
        return "模型运行正常，无需干预。"
    if status == "warn":
        return f"建议关注：{alerts[0] if alerts else '部分指标偏离基线'}。下一周期持续观察。"
    return f"建议行动：{alerts[0] if alerts else '多项指标显著偏离'}。考虑启动模型重训流程。"


# ── ModelMonitor ──────────────────────────────────────────────────────────


class ModelMonitor:
    """
    Scorecard model monitor.

    Records a baseline snapshot at model deployment time and compares
    subsequent periods against it, computing score PSI, feature PSI,
    KS/AUC decay, and rule-based alerts.

    Parameters
    ----------
    baseline_scores : pd.Series or np.ndarray
        Training-set scores.
    baseline_y : pd.Series or np.ndarray, optional
        Training-set target (for baseline KS/AUC).
    baseline_features : pd.DataFrame, optional
        Training-set feature values for distribution comparison.
    baseline_bins : dict, optional
        ``Binning.bin_table_`` for feature PSI calculation.
    psi_warn : float
        Score PSI threshold for ``"warn"`` (default 0.10).
    psi_critical : float
        Score PSI threshold for ``"critical"`` (default 0.25).
    ks_decay_warn : float
        KS decay threshold for ``"warn"`` (default 0.15).
    ks_decay_critical : float
        KS decay threshold for ``"critical"`` (default 0.30).
    score_bins : int
        Number of bins for the score distribution histogram (default 20).
    rules : list of dict, optional
        Custom alert rules (overrides built-in defaults).
    """

    def __init__(
        self,
        baseline_scores: pd.Series | np.ndarray | None = None,
        baseline_y: pd.Series | np.ndarray | None = None,
        baseline_features: pd.DataFrame | None = None,
        baseline_bins: dict | None = None,
        psi_warn: float = 0.10,
        psi_critical: float = 0.25,
        ks_decay_warn: float = 0.15,
        ks_decay_critical: float = 0.30,
        score_bins: int = 20,
        rules: list[dict] | None = None,
    ):
        self.psi_warn = psi_warn
        self.psi_critical = psi_critical
        self.ks_decay_warn = ks_decay_warn
        self.ks_decay_critical = ks_decay_critical
        self.score_bins = score_bins
        self.rules = rules or _DEFAULT_RULES

        self._baseline: dict[str, Any] = {}
        self._snapshots: list[MonitorSnapshot] = []

        if baseline_scores is not None:
            self._set_baseline(
                np.asarray(baseline_scores, dtype=float).ravel(),
                np.asarray(baseline_y, dtype=float).ravel() if baseline_y is not None else None,
                baseline_features,
                baseline_bins,
            )

    # ── factory ────────────────────────────────────────────────────────────

    @classmethod
    def from_proscore(
        cls, ps, baseline_date: str | None = None, key_features: list[str] | None = None
    ) -> ModelMonitor:
        """Create a monitor from a fitted :class:`ProScore` chain instance."""
        if ps.scorecard_ is None:
            raise RuntimeError("ProScore instance must have fit() and scorecard() completed.")
        if ps.transformer_ is None or ps.binner_ is None:
            raise RuntimeError("ProScore instance must have bin() and transform() completed.")

        support = key_features or (ps.selector_.support_ if ps.selector_ else [])
        if not support:
            raise RuntimeError("No model features found.")

        df_trn_woe = ps.transformer_.transform(ps.train_df[support])
        scores = ps.scorecard_.predict(df_trn_woe).values

        baseline_y = ps.train_df[ps.target].values if ps.target else None

        return cls(
            baseline_scores=scores,
            baseline_y=baseline_y,
            baseline_features=ps.train_df[support] if support else None,
            baseline_bins=ps.binner_.bin_table_,
        )

    # ── baseline ───────────────────────────────────────────────────────────

    def _set_baseline(
        self,
        scores: np.ndarray,
        y: np.ndarray | None,
        features: pd.DataFrame | None,
        bins: dict | None,
    ) -> None:
        """Compute and store the baseline snapshot (Fail Fast — no silent catch)."""
        self._baseline["scores"] = scores
        hist, bin_edges = np.histogram(scores, bins=self.score_bins)
        self._baseline["score_dist"] = hist
        self._baseline["score_bins"] = bin_edges

        if y is not None:
            self._baseline["ks"] = _ks(scores, y)
            self._baseline["auc"] = float(roc_auc_score(y, scores))
        else:
            self._baseline["ks"] = None
            self._baseline["auc"] = None

        if features is not None and bins is not None:
            self._baseline["feature_dists"] = {}
            self._baseline["bins_meta"] = {}  # lightweight bin metadata for persistence
            for col in features.columns:
                if col not in bins:
                    continue
                bt = bins[col]
                ref_counts = np.array([b.count for b in bt.bins], dtype=float)
                self._baseline["feature_dists"][col] = ref_counts
                self._baseline["bins_meta"][col] = _serialize_bintable(bt)
            self._baseline["bins"] = bins  # runtime reference (not persisted)
        else:
            self._baseline["feature_dists"] = {}
            self._baseline["bins_meta"] = {}
            self._baseline["bins"] = {}

    # ── track ──────────────────────────────────────────────────────────────

    def track(
        self,
        period: str | None = None,
        scores: pd.Series | np.ndarray | None = None,
        y_true: pd.Series | np.ndarray | None = None,
        features: pd.DataFrame | None = None,
    ) -> MonitorResult:
        """
        Compare current data against the baseline.

        Parameters
        ----------
        period : str
            Period label (e.g. ``"2024Q1"``).
        scores : array-like
            Current-period scores (from ``ScoreCard.predict``).
        y_true : array-like, optional
            Current-period target.  When omitted, KS/AUC are skipped.
        features : pd.DataFrame, optional
            Current-period feature values.  When omitted, feature PSI is
            skipped.

        Returns
        -------
        MonitorResult
        """
        if not self._baseline:
            raise RuntimeError("No baseline set. Call from_proscore() or provide baseline data.")
        if scores is None:
            raise ValueError("scores is required for tracking.")

        period = period or datetime.now().strftime("%Y-%m-%d %H:%M")
        sc = np.asarray(scores, dtype=float).ravel()
        yt = np.asarray(y_true, dtype=float).ravel() if y_true is not None else None

        n = len(sc)
        bad_rate = float(yt.mean()) if yt is not None else np.nan

        # Score PSI
        cur_dist = np.histogram(sc, bins=self._baseline["score_bins"])[0]
        psi_score = psi_from_distributions(self._baseline["score_dist"], cur_dist)

        # Feature PSI
        feat_psi = self._compute_feature_psi(features) if features is not None else {}

        # KS / AUC
        ks_cur, auc_cur, ks_decay, auc_decay = None, None, None, None
        if yt is not None:
            ks_cur = _ks(sc, yt)
            auc_cur = float(roc_auc_score(yt, sc))
            bl_ks = self._baseline.get("ks")
            bl_auc = self._baseline.get("auc")
            if bl_ks and bl_ks > 0:
                ks_decay = (bl_ks - ks_cur) / bl_ks
            if bl_auc and bl_auc > 0.5:
                auc_decay = (bl_auc - auc_cur) / bl_auc

        # Alerts
        metrics = {
            "psi_score": psi_score if not np.isnan(psi_score) else 0,
            "ks_decay": ks_decay if ks_decay is not None else 0,
            "auc_decay": auc_decay if auc_decay is not None else 0,
        }
        alerts = _evaluate_rules(metrics, self.rules, feat_psi)
        status = _status_from_alerts(alerts)

        # Trend
        psi_trend = [s.psi_score for s in self._snapshots] + [psi_score]
        psi_trend = [float(v) for v in psi_trend if not np.isnan(v)]

        recommendation = _build_recommendation(status, alerts)

        result = MonitorResult(
            period=period,
            timestamp=datetime.now().isoformat(),
            n_samples=n,
            bad_rate=round(bad_rate, 6) if not np.isnan(bad_rate) else np.nan,
            ks=round(ks_cur, 6) if ks_cur is not None else None,
            auc=round(auc_cur, 6) if auc_cur is not None else None,
            psi_score=round(psi_score, 6) if not np.isnan(psi_score) else 0.0,
            psi_features={k: round(v, 6) for k, v in feat_psi.items()},
            alerts=alerts,
            status=status,
            ks_decay=round(ks_decay, 6) if ks_decay is not None else None,
            auc_decay=round(auc_decay, 6) if auc_decay is not None else None,
            psi_score_trend=psi_trend,
            recommendation=recommendation,
        )

        self._snapshots.append(result)
        return result

    def _compute_feature_psi(self, features: pd.DataFrame) -> dict[str, float]:
        """Compute per-feature PSI vs baseline distributions."""
        result = {}
        skipped: list[str] = []
        bins = self._baseline.get("bins", {})
        for col, ref_counts in self._baseline.get("feature_dists", {}).items():
            if col not in features.columns:
                skipped.append(col)
                continue
            bt = bins.get(col)
            if bt is None and self._baseline.get("bins_meta"):
                # Reconstruct lightweight BinTable from persisted metadata
                bt = _deserialize_bintable(self._baseline["bins_meta"].get(col, {}))
            if bt is None:
                continue
            cur_counts = _feature_distribution(features[col], bt)
            psi_val = psi_from_distributions(ref_counts, cur_counts)
            if not np.isnan(psi_val):
                result[col] = float(psi_val)
        if skipped:
            warnings.warn(
                f"Feature PSI skipped {len(skipped)} features not in baseline: {skipped[:10]}{'...' if len(skipped) > 10 else ''}",
                stacklevel=2,
            )
        return result

    # ── snapshots & persistence ────────────────────────────────────────────

    def add_snapshot(self, snapshot: MonitorSnapshot) -> None:
        """Manually append a historical snapshot (e.g. for offline backfill)."""
        self._snapshots.append(snapshot)

    @property
    def history(self) -> pd.DataFrame:
        """All snapshots as a sorted DataFrame."""
        if not self._snapshots:
            return pd.DataFrame()
        rows = []
        for s in self._snapshots:
            rows.append({
                "period": s.period, "timestamp": s.timestamp,
                "n": s.n_samples, "bad_rate": s.bad_rate,
                "ks": s.ks, "auc": s.auc,
                "psi_score": s.psi_score,
                "psi_features_n": len(s.psi_features),
                "n_alerts": len(s.alerts), "status": s.status,
            })
        return pd.DataFrame(rows)

    def generate_alerts(self) -> list[str]:
        """Re-run alert rules on the most recent snapshot."""
        if not self._snapshots:
            return []
        return self._snapshots[-1].alerts

    def save(self, path: str) -> str:
        """Persist baseline + all snapshots as JSON. Returns absolute path."""
        import os

        # Persist feature distributions and bin metadata for full load() support
        data = {
            "baseline": {
                "score_bins": self._baseline.get("score_bins", []).tolist(),
                "score_dist": self._baseline.get("score_dist", []).tolist(),
                "ks": self._baseline.get("ks"),
                "auc": self._baseline.get("auc"),
                "feature_dists": {
                    k: v.tolist()
                    for k, v in self._baseline.get("feature_dists", {}).items()
                },
                "bins_meta": self._baseline.get("bins_meta", {}),
            },
            "snapshots": [
                {
                    "period": s.period, "timestamp": s.timestamp,
                    "n_samples": s.n_samples, "bad_rate": s.bad_rate,
                    "ks": s.ks, "auc": s.auc, "psi_score": s.psi_score,
                    "psi_features": s.psi_features, "alerts": s.alerts,
                    "status": s.status,
                }
                for s in self._snapshots
            ],
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return os.path.abspath(path)

    @classmethod
    def load(cls, path: str) -> ModelMonitor:
        """Restore a monitor from a JSON file saved by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        bl = data["baseline"]
        monitor = cls()
        monitor._baseline = {
            "score_bins": np.array(bl["score_bins"]),
            "score_dist": np.array(bl["score_dist"]),
            "ks": bl.get("ks"),
            "auc": bl.get("auc"),
            "feature_dists": {
                k: np.array(v, dtype=float)
                for k, v in bl.get("feature_dists", {}).items()
            },
            "bins_meta": bl.get("bins_meta", {}),
            # "bins" is intentionally empty after load — BinTable objects are
            # not JSON-serializable.  _compute_feature_psi reconstructs the
            # necessary bin metadata from "bins_meta" on demand via
            # _deserialize_bintable.
            "bins": {},
        }
        for s in data.get("snapshots", []):
            monitor._snapshots.append(MonitorSnapshot(
                period=s["period"], timestamp=s["timestamp"],
                n_samples=s["n_samples"], bad_rate=s["bad_rate"],
                ks=s.get("ks"), auc=s.get("auc"),
                psi_score=s.get("psi_score", 0), psi_features=s.get("psi_features", {}),
                alerts=s.get("alerts", []), status=s.get("status", "ok"),
            ))
        return monitor


# ── helpers ───────────────────────────────────────────────────────────────


def _ks(scores: np.ndarray, y: np.ndarray) -> float:
    """KS statistic from scores (higher score = lower risk)."""
    order = np.argsort(scores)[::-1]
    y_sorted = y[order]
    total_bad = y_sorted.sum()
    total_good = len(y_sorted) - total_bad
    if total_bad == 0 or total_good == 0:
        return 0.0
    cum_bad = np.cumsum(y_sorted) / total_bad
    cum_good = np.cumsum(1 - y_sorted) / total_good
    return float(np.abs(cum_bad - cum_good).max())


def _feature_distribution(series: pd.Series, bt: Any) -> np.ndarray:
    """Build per-bin counts for a feature using baseline bin boundaries."""
    counts = np.zeros(len(bt.bins), dtype=float)
    col_data = series.dropna()
    for i, b in enumerate(bt.bins):
        if b.bin_label == "missing":
            counts[i] = float(series.isna().sum())
        elif bt.dtype == "continuous" and i < len(bt.cutoffs):
            lo = bt.cutoffs[i - 1] if i > 0 else -np.inf
            hi = bt.cutoffs[i] if i < len(bt.cutoffs) else np.inf
            counts[i] = float(((col_data > lo) & (col_data <= hi)).sum())
        elif bt.dtype == "categorical" and bt.cat_mapping:
            vals = [v for v, bin_no in bt.cat_mapping.items() if bin_no == b.bin_no]
            counts[i] = float(col_data.isin(vals).sum())
        else:
            counts[i] = 0.0
    return counts


def _serialize_bintable(bt: Any) -> dict:
    """Extract minimal bin metadata for JSON persistence."""
    return {
        "dtype": bt.dtype,
        "cutoffs": [float(c) for c in bt.cutoffs],
        "cat_mapping": {str(k): int(v) for k, v in bt.cat_mapping.items()},
        "bins": [
            {"bin_no": b.bin_no, "bin_label": b.bin_label}
            for b in bt.bins
        ],
    }


class _LightBin:
    """Lightweight bin-like object for distribution counting after load()."""
    def __init__(self, meta: dict):
        self.bin_no = meta["bin_no"]
        self.bin_label = meta["bin_label"]


class _LightBinTable:
    """Lightweight BinTable-like object reconstructed from persisted metadata.

    Used **only** after ``load()`` to enable ``_compute_feature_psi`` without
    requiring the original ``Binning.bin_table_`` runtime objects (which are
    not JSON-serializable).
    """
    def __init__(self, meta: dict):
        self.dtype = meta.get("dtype", "continuous")
        self.cutoffs = meta.get("cutoffs", [])
        self.cat_mapping = {k: int(v) for k, v in meta.get("cat_mapping", {}).items()}
        self.bins = [_LightBin(b) for b in meta.get("bins", [])]


def _deserialize_bintable(meta: dict) -> Any:
    """Reconstruct a lightweight BinTable-like object from persisted metadata."""
    if not meta:
        return None
    return _LightBinTable(meta)
