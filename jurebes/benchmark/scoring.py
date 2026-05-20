"""Scoring callable registry for the benchmark harness."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)

from jurebes.benchmark.calibration import (
    brier_score as _brier,
    expected_calibration_error as _ece,
)


def _f1_macro(y_true, y_pred, _probs=None, classes=None):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _f1_micro(y_true, y_pred, _probs=None, classes=None):
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def _accuracy(y_true, y_pred, _probs=None, classes=None):
    return float(accuracy_score(y_true, y_pred))


def _balanced_accuracy(y_true, y_pred, _probs=None, classes=None):
    return float(balanced_accuracy_score(y_true, y_pred))


def _log_loss(y_true, y_pred, probs=None, classes=None):
    if probs is None or classes is None:
        return float("nan")
    return float(log_loss(y_true, probs, labels=list(classes)))


def _top_k_accuracy(y_true, y_pred, probs=None, classes=None):
    if probs is None or classes is None:
        return float("nan")
    return float(top_k_accuracy_score(y_true, probs, k=3, labels=list(classes)))


def _ece_score(y_true, y_pred, probs=None, classes=None):
    return _ece(y_true, probs, classes)


def _brier_score(y_true, y_pred, probs=None, classes=None):
    return _brier(y_true, probs, classes)


SCORERS: Dict[str, Callable] = {
    "f1_macro": _f1_macro,
    "f1_micro": _f1_micro,
    "accuracy": _accuracy,
    "balanced_accuracy": _balanced_accuracy,
    "log_loss": _log_loss,
    "top_k_accuracy": _top_k_accuracy,
    "ece": _ece_score,
    "brier": _brier_score,
}


def get(name: str) -> Callable:
    """Look up a scoring callable by name."""
    if name not in SCORERS:
        raise KeyError(f"unknown scoring metric {name!r}; choose from {sorted(SCORERS)}")
    return SCORERS[name]


def available() -> list:
    return sorted(SCORERS.keys())
