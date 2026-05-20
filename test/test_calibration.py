"""Tests for jurebes.benchmark.calibration."""

import numpy as np
import pytest

from jurebes.benchmark.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from jurebes.benchmark.scoring import SCORERS


# ── ECE ──────────────────────────────────────────────────────────────


def test_ece_perfect_calibration_low():
    """Predictions with confidence matching accuracy: low ECE."""
    classes = ["a", "b"]
    y_true = ["a"] * 80 + ["b"] * 20
    # confident correct on a's (1.0), confident wrong on b's (1.0) → not calibrated
    # better: each prediction has confidence equal to its empirical accuracy
    proba = np.array([[0.9, 0.1]] * 80 + [[0.1, 0.9]] * 20)
    ece = expected_calibration_error(y_true, proba, classes)
    assert ece < 0.2  # most predictions correct AND confident


def test_ece_overconfident_high():
    """Overconfident model (always 1.0 but only 50% correct): high ECE."""
    classes = ["a", "b"]
    y_true = ["a"] * 50 + ["b"] * 50
    proba = np.array([[1.0, 0.0]] * 100)  # always predicts a with full confidence
    ece = expected_calibration_error(y_true, proba, classes)
    assert ece > 0.4  # 50% accuracy at 100% confidence → 0.5 gap


def test_ece_nan_when_no_probs():
    assert np.isnan(expected_calibration_error(["a"], None, ["a", "b"]))


def test_ece_bounded():
    """ECE always falls in [0, 1]."""
    classes = ["a", "b", "c"]
    rng = np.random.default_rng(0)
    proba = rng.dirichlet([1, 1, 1], size=200)
    y_true = [classes[i] for i in rng.integers(0, 3, 200)]
    ece = expected_calibration_error(y_true, proba, classes)
    assert 0.0 <= ece <= 1.0


# ── Brier ────────────────────────────────────────────────────────────


def test_brier_perfect_prediction_zero():
    classes = ["a", "b"]
    proba = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = ["a", "b"]
    assert brier_score(y, proba, classes) == 0.0


def test_brier_uniform_chance():
    """Uniform predictions on 2-class: Brier = 2 * (0.5)^2 = 0.5."""
    classes = ["a", "b"]
    proba = np.array([[0.5, 0.5]] * 10)
    y = ["a"] * 5 + ["b"] * 5
    assert brier_score(y, proba, classes) == pytest.approx(0.5)


def test_brier_nan_when_no_probs():
    assert np.isnan(brier_score(["a"], None, ["a"]))


# ── reliability_curve ───────────────────────────────────────────────


def test_reliability_curve_shapes():
    classes = ["a", "b"]
    rng = np.random.default_rng(0)
    proba = rng.dirichlet([1, 1], size=200)
    y_true = [classes[i] for i in rng.integers(0, 2, 200)]
    centers, acc, conf, counts = reliability_curve(y_true, proba, classes, n_bins=10)
    assert len(centers) == len(acc) == len(conf) == len(counts) == 10
    assert centers[0] < centers[-1]
    assert counts.sum() == 200


def test_reliability_curve_handles_no_probs():
    centers, acc, conf, counts = reliability_curve(["a"], None, ["a", "b"], n_bins=5)
    assert len(centers) == 5
    assert counts.sum() == 0


# ── scoring registry wiring ─────────────────────────────────────────


def test_ece_in_scorers_registry():
    assert "ece" in SCORERS
    assert "brier" in SCORERS


def test_ece_scorer_callable():
    classes = ["a", "b"]
    proba = np.array([[0.9, 0.1], [0.1, 0.9]])
    val = SCORERS["ece"](["a", "b"], None, proba, classes)
    assert 0.0 <= val <= 1.0


def test_brier_scorer_callable():
    classes = ["a", "b"]
    proba = np.array([[0.9, 0.1], [0.1, 0.9]])
    val = SCORERS["brier"](["a", "b"], None, proba, classes)
    assert 0.0 <= val <= 2.0


# ── end-to-end with compare ────────────────────────────────────────


def test_compare_with_calibration_scoring():
    from jurebes.benchmark import compare

    X = ["hello there"] * 6 + ["see you later"] * 6 + ["thank you"] * 6
    y = ["greet"] * 6 + ["bye"] * 6 + ["thanks"] * 6
    result = compare(["nb_multinomial"], X, y, k=2, scoring=("ece", "brier"))
    row = result.rows[0]
    assert "ece" in row.fold_scores
    assert "brier" in row.fold_scores
    for v in row.fold_scores["ece"] + row.fold_scores["brier"]:
        assert not np.isnan(v)
