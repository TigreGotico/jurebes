"""Benchmark harness for comparing classical-ML intent classifiers."""

from jurebes.benchmark.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from jurebes.benchmark.harness import (
    ComparisonResult,
    compare,
    cross_validate,
    train_test,
)
from jurebes.benchmark.metrics import RunResult, pooled_percentiles
from jurebes.benchmark.report import to_json, to_markdown
from jurebes.benchmark.scoring import SCORERS as SCORING

__all__ = [
    "RunResult",
    "ComparisonResult",
    "train_test",
    "cross_validate",
    "compare",
    "to_markdown",
    "to_json",
    "pooled_percentiles",
    "SCORING",
    "expected_calibration_error",
    "brier_score",
    "reliability_curve",
]
