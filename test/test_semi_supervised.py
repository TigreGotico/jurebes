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
    self_train,
    co_train,
)


def _template_clf() -> IntentClassifier:
    return IntentClassifier(
        Pipeline([("v", TfidfVectorizer()), ("c", LogisticRegression(max_iter=500))]),
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


def test_self_train_adds_samples_when_threshold_permits():
    X = ["hello", "hi", "hey there", "good morning", "goodbye", "bye now", "see you", "later"]
    y = ["greet"] * 4 + ["bye"] * 4
    unlabeled = ["hello friend", "see you tomorrow", "good day", "bye for now"]
    res = self_train(
        _template_clf(), X, y, unlabeled,
        confidence_threshold=0.5, k_per_round=2, max_rounds=3,
    )
    assert sum(res.added_per_round) > 0
    assert len(res.labeled_X) > len(X)
    assert len(res.labeled_X) == len(res.labeled_y)


def test_self_train_early_stops_on_plateau():
    X = ["hello", "hi", "hey", "good morning", "goodbye", "bye", "see you", "later"]
    y = ["greet"] * 4 + ["bye"] * 4
    unlabeled = ["hello there", "bye now", "good evening", "see ya", "morning",
                 "later friend", "hi again", "bye bye"]
    eval_X = ["hello", "goodbye"]
    eval_y = ["greet", "bye"]
    res = self_train(
        _template_clf(), X, y, unlabeled,
        confidence_threshold=0.3, k_per_round=1, max_rounds=20,
        eval_X=eval_X, eval_y=eval_y, early_stop_patience=2,
    )
    # Should not run all 20 rounds on a trivial dataset.
    assert len(res.added_per_round) < 20
    assert res.stopped_early or len(res.added_per_round) <= len(unlabeled)


def test_self_train_threshold_schedule_called_per_round():
    calls: List[int] = []

    def schedule(r, t):
        calls.append(r)
        return t

    X = ["hello", "hi", "goodbye", "bye"]
    y = ["greet", "greet", "bye", "bye"]
    unlabeled = ["hello there", "see you", "morning"]
    self_train(
        _template_clf(), X, y, unlabeled,
        confidence_threshold=0.5, k_per_round=1, max_rounds=4,
        threshold_schedule=schedule,
    )
    assert len(calls) >= 1
    assert calls[0] == 0


def _word_view():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return IntentClassifier(
        Pipeline([("v", TfidfVectorizer()), ("c", LogisticRegression(max_iter=500))]),
    )


def _char_view():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return IntentClassifier(
        Pipeline([
            ("v", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
            ("c", LogisticRegression(max_iter=500)),
        ]),
    )


def test_co_train_returns_two_view_history():
    X = ["hello", "hi", "hey", "good morning", "goodbye", "bye", "see you", "later"]
    y = ["greet"] * 4 + ["bye"] * 4
    unlabeled = ["hello there", "see you tomorrow", "good day", "bye for now",
                 "morning", "later friend"]
    res = co_train(
        _word_view, _char_view, X, y, unlabeled,
        confidence_threshold=0.4, k_per_round=2, max_rounds=3,
    )
    assert len(res.added_per_round_view_a) >= 1
    assert len(res.added_per_round_view_b) == len(res.added_per_round_view_a)


def test_co_train_adds_pseudo_labels_from_each_view():
    X = ["hello", "hi", "hey", "good morning", "goodbye", "bye", "see you", "later"]
    y = ["greet"] * 4 + ["bye"] * 4
    unlabeled = ["hello there", "see you tomorrow", "good day", "bye for now",
                 "morning", "later friend", "hi friend", "bye bye"]
    res = co_train(
        _word_view, _char_view, X, y, unlabeled,
        confidence_threshold=0.4, k_per_round=2, max_rounds=4,
    )
    assert sum(res.added_per_round_view_a) > 0
    assert sum(res.added_per_round_view_b) > 0
    assert len(res.labeled_X) > len(X)


def test_pseudo_label_primitives_exposed():
    public = set(semi_supervised.__all__)
    assert {
        "SELECTION_STRATEGIES",
        "pseudo_label",
        "select_high_confidence",
    }.issubset(public)
    assert set(SELECTION_STRATEGIES) >= {"global_top_k", "per_class_quota"}
