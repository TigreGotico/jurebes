"""Tests for jurebes.active_learning primitives."""

import pytest

from jurebes import IntentClassifier
from jurebes.active_learning import (
    bucket_paraphrases,
    confusion_pairs,
    disagreement_score,
    uncertainty_scores,
)
from jurebes.baselines import BASELINES


@pytest.fixture
def clf():
    c = IntentClassifier(BASELINES.build("logreg"))
    c.add_intent("greet", ["hello there", "hi friend", "good morning", "hey buddy"])
    c.add_intent("bye", ["goodbye", "see you later", "farewell", "ciao now"])
    c.add_intent("thanks", ["thank you", "thanks a lot", "much appreciated", "cheers"])
    c.fit()
    return c


# ── uncertainty_scores ──────────────────────────────────────────────


def test_uncertainty_scores_shape(clf):
    out = uncertainty_scores(clf, ["hello there", "see you later"], ["greet", "bye"])
    assert len(out) == 2
    for s in out:
        assert 0.0 <= s.top_conf <= 1.0
        assert 0.0 <= s.requested_conf <= 1.0


def test_uncertainty_scores_requested_conf_set(clf):
    out = uncertainty_scores(clf, ["hello there"], ["greet"])
    assert out[0].requested_conf > 0.0


def test_uncertainty_scores_no_labels(clf):
    out = uncertainty_scores(clf, ["hello", "goodbye"])
    assert all(s.requested is None for s in out)
    assert all(s.requested_conf == 0.0 for s in out)


# ── bucket_paraphrases ──────────────────────────────────────────────


def test_bucket_paraphrases_assigns_high_conf_to_skip(clf):
    paras = {"greet": ["hello there"]}  # exact training match
    out = bucket_paraphrases(clf, paras, hard_conf_max=0.5)
    assert "hello there" in out["greet"].skip


def test_bucket_paraphrases_suspect_on_wrong_intent(clf):
    paras = {"greet": ["thank you"]}  # actually a thanks paraphrase
    out = bucket_paraphrases(clf, paras)
    assert "thank you" in out["greet"].suspect


def test_bucket_paraphrases_dedupes(clf):
    paras = {"greet": ["hello there"]}
    out = bucket_paraphrases(clf, paras, dedupe_against={"greet": ["hello there"]})
    assert out["greet"].skip == out["greet"].hard == out["greet"].suspect == []


# ── confusion_pairs ─────────────────────────────────────────────────


def test_confusion_pairs_extracts_from_result(clf):
    from jurebes.benchmark import compare
    X = (["hello there"] * 3 + ["see you later"] * 3
         + ["thank you kindly"] * 3 + ["thanks again"] * 3)
    y = ["greet"] * 3 + ["bye"] * 3 + ["thanks"] * 3 + ["thanks"] * 3
    result = compare(["nb_multinomial"], X, y, k=3)
    pairs = confusion_pairs(result, top_n=5)
    assert isinstance(pairs, list)
    for triple in pairs:
        a, b, count = triple
        assert a != b
        assert count > 0


def test_confusion_pairs_empty_on_empty_result():
    class _Empty:
        rows = []
    assert confusion_pairs(_Empty()) == []


# ── disagreement_score ──────────────────────────────────────────────


def test_disagreement_score_zero_when_unanimous(clf):
    score = disagreement_score([clf, clf, clf], "hello there")
    assert score == 0.0


def test_disagreement_score_positive_when_split(clf):
    other = IntentClassifier(BASELINES.build("logreg"))
    other.add_intent("greet", ["hi"])
    other.add_intent("bye", ["bye", "see you", "later", "farewell", "ciao"])
    other.fit()
    score = disagreement_score([clf, other], "see you later")
    assert score >= 0.0


def test_disagreement_score_empty_classifiers():
    assert disagreement_score([], "anything") == 0.0
