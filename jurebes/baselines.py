"""Baselines registry — ~20 canonical classical-ML pipelines for text intents."""

from __future__ import annotations

from typing import Callable, Dict

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from jurebes.featurizers import (
    char_word_union,
    count_word,
    tfidf_char,
    tfidf_word,
)


class _Registry:
    def __init__(self):
        self._items: Dict[str, Callable[[], Pipeline]] = {}

    def register(self, name: str, factory: Callable[[], Pipeline]) -> None:
        self._items[name] = factory

    def build(self, name: str) -> Pipeline:
        if name not in self._items:
            raise KeyError(f"unknown baseline: {name}")
        return self._items[name]()

    def names(self):
        return list(self._items.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self):
        return iter(self._items)


BASELINES = _Registry()


def _cal(est):
    return CalibratedClassifierCV(est, cv=3)


def _p(feat, clf):
    return Pipeline([("feat", feat), ("clf", clf)])


BASELINES.register("nb_multinomial", lambda: _p(tfidf_word(), MultinomialNB()))
BASELINES.register("nb_complement", lambda: _p(tfidf_word(), ComplementNB()))
BASELINES.register("nb_bernoulli", lambda: _p(count_word(binary=True), BernoulliNB()))
BASELINES.register("logreg", lambda: _p(tfidf_word(), LogisticRegression(max_iter=1000)))
BASELINES.register("logreg_char", lambda: _p(tfidf_char(), LogisticRegression(max_iter=1000)))
BASELINES.register("linear_svc", lambda: _p(tfidf_word(), _cal(LinearSVC())))
BASELINES.register("linear_svc_char", lambda: _p(tfidf_char(), _cal(LinearSVC())))
BASELINES.register("rbf_svc", lambda: _p(tfidf_word(), SVC(kernel="rbf", probability=True)))
BASELINES.register("knn", lambda: _p(tfidf_word(), KNeighborsClassifier()))
BASELINES.register("sgd_log", lambda: _p(tfidf_word(), SGDClassifier(loss="log_loss")))
BASELINES.register("sgd_hinge", lambda: _p(tfidf_word(), _cal(SGDClassifier(loss="hinge"))))
BASELINES.register("passive_aggressive", lambda: _p(tfidf_word(), _cal(PassiveAggressiveClassifier())))
BASELINES.register("perceptron", lambda: _p(tfidf_word(), _cal(Perceptron())))
BASELINES.register("ridge", lambda: _p(tfidf_word(), _cal(RidgeClassifier())))
BASELINES.register("random_forest", lambda: _p(tfidf_word(), RandomForestClassifier()))
BASELINES.register("extra_trees", lambda: _p(tfidf_word(), ExtraTreesClassifier()))
BASELINES.register("gradient_boosting", lambda: _p(tfidf_word(), GradientBoostingClassifier()))
BASELINES.register("hist_gbm", lambda: _p(tfidf_word(), HistGradientBoostingClassifier()))
BASELINES.register("decision_tree", lambda: _p(tfidf_word(), DecisionTreeClassifier()))
BASELINES.register("mlp_shallow", lambda: _p(tfidf_word(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=500)))


def _voting_soft():
    return _p(
        tfidf_word(),
        VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=1000)),
                ("svc", _cal(LinearSVC())),
                ("mnb", MultinomialNB()),
            ],
            voting="soft",
        ),
    )


def _stacking():
    return _p(
        tfidf_word(),
        StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=1000)),
                ("svc", _cal(LinearSVC())),
                ("mnb", MultinomialNB()),
            ],
            final_estimator=LogisticRegression(max_iter=1000),
        ),
    )


BASELINES.register("voting_soft", _voting_soft)
BASELINES.register("stacking", _stacking)
BASELINES.register("union_logreg", lambda: _p(char_word_union(), LogisticRegression(max_iter=1000)))


def default() -> Pipeline:
    return BASELINES.build("linear_svc")
