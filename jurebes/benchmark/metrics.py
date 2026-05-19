"""Metrics + RunResult dataclass."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class RunResult:
    name: str
    accuracy: float
    macro_f1: float
    micro_f1: float
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    train_seconds: float = 0.0
    predict_ms_p50: float = 0.0
    predict_ms_p95: float = 0.0
    predict_ms_p99: float = 0.0
    model_size_bytes: int = 0
    confusion_matrix: List[List[int]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
