from pathlib import Path

import pytest

from jurebes import IntentClassifier, IntentResult
from jurebes.baselines import BASELINES


def _train_toy():
    clf = IntentClassifier(BASELINES.build("logreg"))
    clf.add_intent("hello", ["hello", "hi", "hey there", "hello friend", "hi friend"])
    clf.add_intent("joke", ["tell me a joke", "say a joke", "make me laugh", "do you know a joke"])
    clf.add_intent("name", ["what is your name", "who are you", "tell me your name"])
    clf.fit()
    return clf


def test_train_and_predict():
    clf = _train_toy()
    r = clf.predict("hello there")
    assert isinstance(r, IntentResult)
    assert r.intent == "hello"
    assert 0.0 < r.confidence <= 1.0


def test_predict_proba_sorted():
    clf = _train_toy()
    ranked = clf.predict_proba("tell me a joke please")
    assert ranked[0].intent == "joke"
    confs = [r.confidence for r in ranked]
    assert confs == sorted(confs, reverse=True)


def test_save_load_roundtrip(tmp_path: Path):
    clf = _train_toy()
    p = tmp_path / "model.joblib"
    clf.save(p)
    clf2 = IntentClassifier.load(p)
    assert clf2.predict("hi friend").intent == "hello"


def test_calibrate_wraps_linear_svc():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    raw = Pipeline([("v", TfidfVectorizer()), ("c", LinearSVC())])
    clf = IntentClassifier(raw)
    assert hasattr(clf.estimator, "predict_proba") or clf.estimator.__class__.__name__ == "CalibratedClassifierCV"


def test_fit_requires_two_classes():
    clf = IntentClassifier(BASELINES.build("logreg"))
    clf.add_intent("hello", ["hi", "hello"])
    with pytest.raises(ValueError):
        clf.fit()
