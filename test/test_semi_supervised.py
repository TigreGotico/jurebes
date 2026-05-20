"""Tests for the semi-supervised module."""

from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from jurebes import IntentClassifier
from jurebes import semi_supervised
from jurebes.semi_supervised import (
    SELECTION_STRATEGIES,
    pseudo_label,
    select_high_confidence,
)


def _seed_clf() -> IntentClassifier:
    clf = IntentClassifier(
        Pipeline([("v", TfidfVectorizer()), ("c", LogisticRegression(max_iter=500))]),
    )
    clf.add_intent("greet", ["hello", "hi there", "good morning", "hey"])
    clf.add_intent("bye", ["goodbye", "see you", "bye now", "later"])
    clf.fit()
    return clf


def test_pseudo_label_shape_and_ordering():
    clf = _seed_clf()
    unlabeled = ["hello friend", "see you tomorrow", "good evening"]
    scored = pseudo_label(clf, unlabeled)
    assert len(scored) == len(unlabeled)
    for label, conf in scored:
        assert isinstance(label, str) and label in {"greet", "bye"}
        assert 0.0 <= conf <= 1.0


def test_select_high_confidence_per_class_quota_balances():
    # Synthetic scored list with 5 "a" and 3 "b" — quota=2 should give at most 2 of each.
    scored = [
        ("a", 0.95), ("a", 0.9), ("a", 0.85), ("a", 0.8), ("a", 0.75),
        ("b", 0.99), ("b", 0.7), ("b", 0.65),
    ]
    picked = select_high_confidence(
        scored, strategy="per_class_quota", k=2, threshold=0.5,
    )
    labels = [scored[i][0] for i in picked]
    assert labels.count("a") == 2
    assert labels.count("b") == 2


def test_select_high_confidence_global_top_k_honours_k():
    scored = [("a", 0.9), ("b", 0.8), ("a", 0.7), ("b", 0.6), ("a", 0.5)]
    picked = select_high_confidence(
        scored, strategy="global_top_k", k=3, threshold=0.0,
    )
    assert len(picked) == 3
    # Top three by confidence: indices 0, 1, 2
    assert set(picked) == {0, 1, 2}


def test_select_high_confidence_unknown_strategy_raises():
    with pytest.raises(KeyError):
        select_high_confidence(
            [("a", 0.9)], strategy="not_a_strategy", k=1, threshold=0.0,
        )


def test_select_high_confidence_threshold_filters():
    scored = [("a", 0.9), ("a", 0.2), ("b", 0.3)]
    picked = select_high_confidence(
        scored, strategy="global_top_k", k=10, threshold=0.5,
    )
    assert picked == [0]


def test_pseudo_label_primitives_exposed():
    public = set(semi_supervised.__all__)
    assert {
        "SELECTION_STRATEGIES",
        "pseudo_label",
        "select_high_confidence",
    }.issubset(public)
    assert set(SELECTION_STRATEGIES) >= {"global_top_k", "per_class_quota"}
