"""Benchmark harness for comparing classical-ML intent classifiers."""

from jurebes.benchmark.harness import (
    ComparisonResult,
    compare,
    cross_validate,
    train_test,
)
from jurebes.benchmark.metrics import RunResult
from jurebes.benchmark.report import to_json, to_markdown

__all__ = [
    "RunResult",
    "ComparisonResult",
    "train_test",
    "cross_validate",
    "compare",
    "to_markdown",
    "to_json",
]
