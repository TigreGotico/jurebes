"""Metrics + RunResult dataclass."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class RunResult:
    name: str
    accuracy: float
    macro_f1: float
    micro_f1: float
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    train_seconds: float = 0.0
    # Per-fold mean of within-fold percentiles (legacy / variance signal).
    predict_ms_p50: float = 0.0
    predict_ms_p95: float = 0.0
    predict_ms_p99: float = 0.0
    # Pooled percentiles across every individual prediction across all
    # folds — statistically meaningful for tail-latency.
    predict_ms_p50_pooled: float = 0.0
    predict_ms_p95_pooled: float = 0.0
    predict_ms_p99_pooled: float = 0.0
    predict_ms_mean: float = 0.0
    model_size_bytes: int = 0
    confusion_matrix: List[List[int]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    extra_scores: Dict[str, float] = field(default_factory=dict)
    group: str = ""
    # Per-fold scores for each scoring metric (empty for train_test runs).
    fold_scores: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def pooled_percentiles(samples: List[float]) -> Dict[str, float]:
    """Return p50/p95/p99/mean over a flat pool of per-prediction latencies (ms)."""
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
    }
