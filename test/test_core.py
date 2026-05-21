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


def test_save_load_full_state(tmp_path: Path):
    from jurebes.slots import SklearnIOBTagger
    tagger = SklearnIOBTagger()
    clf = IntentClassifier(BASELINES.build("logreg"), tagger=tagger)
    clf.add_entity("name", ["bob", "alice", "tom"])
    clf.add_intent("greet_name", ["hi {name}", "hello {name}", "hey {name}"])
    clf.add_intent("plain", ["foo", "bar", "baz"])
    clf.fit()
    p = tmp_path / "m.joblib"
    clf.save(p)
    clf2 = IntentClassifier.load(p)
    assert clf2._samples == clf._samples
    assert clf2._entity_samples == clf._entity_samples
    assert clf2.tagger is not None
    assert clf2._fitted is True


def test_intent_classifier_with_tagger_end_to_end():
    from jurebes.slots import SklearnIOBTagger
    tagger = SklearnIOBTagger()
    clf = IntentClassifier(BASELINES.build("logreg"), tagger=tagger)
    clf.add_entity("name", ["bob", "alice", "tom", "jarbas"])
    clf.add_intent("greet_name", [
        "hi {name}", "hello {name}", "hey {name}",
        "my name is {name}", "call me {name}",
    ])
    clf.add_intent("hello", ["hello there", "hi friend", "hey", "hello"])
    clf.fit()
    r = clf.predict("my name is bob")
    assert r.intent in {"greet_name", "hello"}
    # tagger should have populated entities for at least one prediction
    r2 = clf.predict("call me alice")
    assert isinstance(r2.entities, dict)


def test_calibrate_false_without_proba_raises():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    raw = Pipeline([("v", TfidfVectorizer()), ("c", LinearSVC())])
    with pytest.raises(ValueError):
        IntentClassifier(raw, calibrate=False)


def test_calibrate_always_wraps():
    from jurebes.baselines import BASELINES
    clf = IntentClassifier(BASELINES.build("logreg"), calibrate="always")
    assert clf.estimator.__class__.__name__ == "CalibratedClassifierCV"


def test_predict_batch_matches_predict():
    clf = _train_toy()
    utts = ["hello there", "tell me a joke", "what is your name", "hi friend"]
    single = [clf.predict(u).intent for u in utts]
    batch = [r.intent for r in clf.predict_batch(utts)]
    assert single == batch


def test_predict_batch_empty_returns_empty_list():
    clf = _train_toy()
    assert clf.predict_batch([]) == []


def test_calibrate_always_wraps_even_when_proba_exists():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    raw = Pipeline([("v", TfidfVectorizer()), ("c", LogisticRegression(max_iter=500))])
    clf = IntentClassifier(raw, calibrate="always")
    assert clf.estimator.__class__.__name__ == "CalibratedClassifierCV"


def test_calibrate_false_raises_for_non_proba_estimator():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    raw = Pipeline([("v", TfidfVectorizer()), ("c", LinearSVC())])
    with pytest.raises(ValueError):
        IntentClassifier(raw, calibrate=False)


def test_save_load_preserves_tagger_state(tmp_path: Path):
    from jurebes.slots import SklearnIOBTagger
    tagger = SklearnIOBTagger()
    clf = IntentClassifier(BASELINES.build("logreg"), tagger=tagger)
    clf.add_entity("name", ["bob", "alice", "tom", "jarbas"])
    clf.add_intent("greet_name", [
        "hi {name}", "hello {name}", "hey {name}",
        "my name is {name}", "call me {name}", "I am {name}",
    ])
    clf.add_intent("hello", ["hello there", "hi friend", "hey", "hello"])
    clf.fit()
    p = tmp_path / "m.joblib"
    clf.save(p)
    loaded = IntentClassifier.load(p)
    assert loaded.tagger is not None
    assert loaded.tagger.fitted
    r = loaded.predict("my name is bob")
    assert isinstance(r.entities, dict)


def test_save_emits_jurebes_version(tmp_path: Path):
    from jurebes.version import __version__
    clf = _train_toy()
    p = tmp_path / "v.joblib"
    clf.save(p)
    loaded = IntentClassifier.load(p)
    assert loaded._loaded_from_version == __version__


def test_fit_requires_two_classes():
    clf = IntentClassifier(BASELINES.build("logreg"))
    clf.add_intent("hello", ["hi", "hello"])
    with pytest.raises(ValueError):
        clf.fit()


# ── generic-text-classification aliases ─────────────────────────────


def test_text_classifier_alias_is_intent_classifier():
    from jurebes import TextClassifier, TextResult
    assert TextClassifier is IntentClassifier
    from jurebes.core import IntentResult
    assert TextResult is IntentResult


def test_add_class_aliases_add_intent():
    clf = IntentClassifier(BASELINES.build("logreg"))
    clf.add_class("spam", ["win money now", "free prize click", "cheap meds"])
    clf.add_class("ham", ["meeting at noon", "see you tomorrow", "lunch plans"])
    clf.fit()
    r = clf.predict("free prize")
    assert r.label == r.intent
    assert r.text == r.utterance
    assert r.label in {"spam", "ham"}
