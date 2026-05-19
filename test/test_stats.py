"""Tests for jurebes.benchmark.stats."""

import numpy as np
import pytest

from jurebes.benchmark.stats import (
    critical_difference,
    friedman_nemenyi,
    mcnemar_test,
    paired_t_test_cv,
    wilcoxon_signed_rank_cv,
)


def test_paired_t_test_identical_scores():
    scores = [0.8, 0.85, 0.82, 0.79, 0.83]
    r = paired_t_test_cv(scores, scores, alpha=0.05)
    assert r.method == "paired_t"
    assert r.pvalue >= 0.05
    assert r.reject_null is False
    assert r.n == 5


def test_paired_t_test_clearly_different():
    a = [0.9] * 10
    b = [0.5] * 10
    # add tiny noise so variance isn't zero
    rng = np.random.default_rng(0)
    a = [v + rng.normal(0, 0.01) for v in a]
    b = [v + rng.normal(0, 0.01) for v in b]
    r = paired_t_test_cv(a, b, alpha=0.05)
    assert r.reject_null is True
    assert r.pvalue < 0.05


def test_wilcoxon_matches_t_test_sign():
    rng = np.random.default_rng(1)
    a = [0.9 + rng.normal(0, 0.01) for _ in range(10)]
    b = [0.5 + rng.normal(0, 0.01) for _ in range(10)]
    rt = paired_t_test_cv(a, b)
    rw = wilcoxon_signed_rank_cv(a, b)
    assert rt.reject_null == rw.reject_null is True


def test_mcnemar_contingency():
    # 10 samples, a correct on idx 0-7, b correct on idx 2-9
    y_true = ["x"] * 10
    preds_a = ["x"] * 8 + ["y"] * 2  # a correct on 0..7
    preds_b = ["y"] * 2 + ["x"] * 8  # b correct on 2..9
    # discordant: indices 0,1 (a right, b wrong) c_=2; indices 8,9 (b right, a wrong) b_=2
    r = mcnemar_test(preds_a, preds_b, y_true)
    assert r.method == "mcnemar"
    # symmetric → not significant
    assert r.reject_null is False


def test_mcnemar_clear_difference():
    y_true = ["x"] * 20
    preds_a = ["x"] * 20  # always right
    preds_b = ["y"] * 20  # always wrong
    r = mcnemar_test(preds_a, preds_b, y_true, alpha=0.05)
    assert r.reject_null is True


def test_friedman_three_baselines():
    # b1 always best, b2 middle, b3 worst across 5 folds
    fold_scores = {
        "b1": [0.90, 0.92, 0.91, 0.89, 0.93],
        "b2": [0.80, 0.82, 0.81, 0.79, 0.83],
        "b3": [0.70, 0.72, 0.71, 0.69, 0.73],
    }
    r = friedman_nemenyi(fold_scores, alpha=0.05)
    assert r.reject_null is True
    assert r.mean_ranks["b1"] < r.mean_ranks["b2"] < r.mean_ranks["b3"]


def test_critical_difference_groups():
    # b1 clearly best, b3 clearly worst, b2 in between
    fold_scores = {
        "b1": [0.95, 0.96, 0.94, 0.95, 0.93, 0.96, 0.95, 0.94, 0.96, 0.95],
        "b2": [0.85, 0.86, 0.84, 0.85, 0.83, 0.86, 0.85, 0.84, 0.86, 0.85],
        "b3": [0.55, 0.56, 0.54, 0.55, 0.53, 0.56, 0.55, 0.54, 0.56, 0.55],
    }
    cd = critical_difference(fold_scores, alpha=0.05)
    assert cd.cd_threshold > 0
    # b1 and b3 should not be in the same group
    for grp in cd.groups:
        assert not ({"b1", "b3"} <= grp)
    assert set(cd.mean_ranks.keys()) == {"b1", "b2", "b3"}


def test_friedman_too_few_baselines():
    with pytest.raises(ValueError):
        friedman_nemenyi({"only_one": [0.8, 0.9]})


def test_cd_diagram_ascii_render():
    fold_scores = {
        "b1": [0.9, 0.91, 0.89],
        "b2": [0.7, 0.71, 0.69],
    }
    cd = critical_difference(fold_scores)
    text = cd.to_ascii()
    assert "Critical Difference" in text
    assert "b1" in text and "b2" in text
