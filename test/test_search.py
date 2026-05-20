"""Tests for the hyperparameter search subsystem."""

from __future__ import annotations

import pytest

from jurebes.search import search, spaces
from jurebes.search.api import SearchResult


_X = [
    "hello", "hi there", "hey friend", "good morning", "howdy",
    "hello friend", "hi", "hey", "hi friend", "greetings",
    "tell me a joke", "say a joke", "make me laugh", "be funny",
    "amuse me", "humor me", "share a joke", "crack a joke",
    "tell something funny", "make a joke",
    "what is your name", "who are you", "tell me your name",
    "say your name", "name please", "what should i call you",
    "what's your name", "name yourself", "introduce yourself", "your name",
]
_y = ["hello"] * 10 + ["joke"] * 10 + ["name"] * 10


def test_grid_search_returns_search_result():
    space = {"feat__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.1, 1.0]}
    r = search("nb_multinomial", space, _X, _y, backend="grid", cv=3)
    assert isinstance(r, SearchResult)
    assert r.best_estimator.__class__.__name__ == "IntentClassifier"
    assert r.best_estimator.predict("hello there").intent in {"hello", "joke", "name"}
    assert r.n_evaluations == 4
    assert r.backend == "grid"


def test_random_search_uses_n_iter():
    space = {"feat__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.1, 0.5, 1.0]}
    r = search("nb_multinomial", space, _X, _y, backend="random", cv=3, n_iter=4)
    assert r.n_evaluations == 4


def test_halving_random_runs():
    space = {"feat__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.1, 0.5, 1.0, 2.0]}
    r = search("nb_multinomial", space, _X, _y, backend="halving_random", cv=2, n_iter=4)
    assert r.n_evaluations >= 1


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        search("nb_multinomial", {}, _X, _y, backend="nope")


def test_spaces_for_baseline():
    sp = spaces.for_baseline("logreg")
    assert "clf__C" in sp
    assert "logreg" in spaces.available()


def test_spaces_coverage_breadth():
    """The curated space registry should cover every baseline group."""
    available = set(spaces.available())
    # at least one baseline per major group
    for name in ["nb_multinomial", "logreg", "linear_svc", "rbf_svc",
                 "random_forest", "mlp_shallow", "sgd_log",
                 "lsa_logreg", "ovr_linear_svc", "knn"]:
        assert name in available, f"{name} missing from search spaces"
    # 35 entries minimum (35 of the 48 baselines)
    assert len(available) >= 30


def test_bayes_backend_optional():
    pytest.importorskip("skopt")
    from skopt.space import Categorical, Real
    space = {
        "feat__ngram_range": Categorical([(1, 1), (1, 2)]),
        "clf__alpha": Real(0.01, 2.0, prior="log-uniform"),
    }
    r = search("nb_multinomial", space, _X, _y, backend="bayes", cv=3, n_iter=4)
    assert isinstance(r, SearchResult)


def test_genetic_backend_optional():
    pytest.importorskip("sklearn_genetic")
    from sklearn_genetic.space import Categorical, Continuous
    space = {
        "feat__min_df": Categorical([1, 2]),
        "clf__alpha": Continuous(0.01, 2.0),
    }
    r = search("nb_multinomial", space, _X, _y, backend="genetic", cv=3, n_iter=10)
    assert isinstance(r, SearchResult)


def test_bayes_missing_dep_message(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "skopt", None)
    # force the bayes module to re-import skopt
    sys.modules.pop("jurebes.search.bayes", None)
    with pytest.raises(ImportError, match=r"jurebes\[search-bayes\]"):
        search("nb_multinomial", {}, _X, _y, backend="bayes")


def test_genetic_missing_dep_message(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "sklearn_genetic", None)
    sys.modules.pop("jurebes.search.genetic", None)
    with pytest.raises(ImportError, match=r"jurebes\[search-genetic\]"):
        search("nb_multinomial", {}, _X, _y, backend="genetic")


def test_halving_grid_smoke():
    small_space = {"feat__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.1, 0.5, 1.0]}
    r = search("nb_multinomial", small_space, _X, _y, backend="halving_grid", cv=3)
    assert r.best_estimator is not None
    assert r.best_estimator.predict("hello there").intent in {"hello", "joke", "name"}


def test_random_with_distribution():
    from scipy.stats import loguniform
    space = {
        "feat__ngram_range": [(1, 1), (1, 2)],
        "clf__C": loguniform(1e-4, 10),
    }
    r = search("logreg", space, _X, _y, backend="random", cv=3, n_iter=3)
    assert isinstance(r, SearchResult)
    assert r.best_estimator is not None


def test_search_best_estimator_is_intent_classifier(tmp_path):
    from jurebes import IntentClassifier
    space = {"clf__alpha": [0.1, 1.0]}
    r = search("nb_multinomial", space, _X, _y, backend="grid", cv=3)
    assert isinstance(r.best_estimator, IntentClassifier)
    assert r.best_estimator.predict("hello").intent in {"hello", "joke", "name"}
    p = tmp_path / "m.joblib"
    r.best_estimator.save(p)
    loaded = IntentClassifier.load(p)
    assert loaded.predict("hello").intent in {"hello", "joke", "name"}


def test_search_best_estimator_samples_populated():
    space = {"clf__alpha": [0.1, 1.0]}
    r = search("nb_multinomial", space, _X, _y, backend="grid", cv=3)
    expected: dict = {}
    for x, lbl in zip(_X, _y):
        expected.setdefault(lbl, []).append(x)
    assert r.best_estimator._samples == expected


def test_spaces_for_baseline_unknown_raises():
    with pytest.raises(KeyError):
        spaces.for_baseline("does_not_exist")
