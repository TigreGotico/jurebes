"""Baselines registry — canonical classical-ML pipelines for text intents."""

from __future__ import annotations

from typing import Callable, Dict, Set

from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    BaggingClassifier,
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
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier

from jurebes.featurizers import (
    char_word_union,
    count_word,
    feature_union,
    hashing_word,
    lda_topics,
    lsa,
    nmf,
    text_stats,
    tfidf_char,
    tfidf_word,
)


def _cal(est):
    return CalibratedClassifierCV(est, cv=3)


def _p(feat, clf):
    return Pipeline([("feat", feat), ("clf", clf)])


def _hist_gbm():
    return Pipeline([
        ("feat", tfidf_word()),
        ("dense", FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)),
        ("clf", HistGradientBoostingClassifier(min_samples_leaf=1)),
    ])


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


# All factories live in a dict literal — easier to scan and to extend.
BASELINE_SPECS: Dict[str, Callable[[], Pipeline]] = {
    # ── naive bayes ────────────────────────────────────────────────
    "nb_multinomial": lambda: _p(tfidf_word(), MultinomialNB()),
    "nb_complement": lambda: _p(tfidf_word(), ComplementNB()),
    "nb_bernoulli": lambda: _p(count_word(binary=True), BernoulliNB()),
    # ── linear models ──────────────────────────────────────────────
    "logreg": lambda: _p(tfidf_word(), LogisticRegression(max_iter=1000)),
    "logreg_char": lambda: _p(tfidf_char(), LogisticRegression(max_iter=1000)),
    "linear_svc": lambda: _p(tfidf_word(), _cal(LinearSVC())),
    "linear_svc_char": lambda: _p(tfidf_char(), _cal(LinearSVC())),
    "sgd_log": lambda: _p(tfidf_word(), SGDClassifier(loss="log_loss")),
    "sgd_hinge": lambda: _p(tfidf_word(), _cal(SGDClassifier(loss="hinge"))),
    "passive_aggressive": lambda: _p(tfidf_word(), _cal(PassiveAggressiveClassifier())),
    "perceptron": lambda: _p(tfidf_word(), _cal(Perceptron())),
    "ridge": lambda: _p(tfidf_word(), _cal(RidgeClassifier())),
    # ── kernel / non-linear ────────────────────────────────────────
    "rbf_svc": lambda: _p(tfidf_word(), SVC(kernel="rbf", probability=True)),
    "knn": lambda: _p(tfidf_word(), KNeighborsClassifier()),
    # ── trees & ensembles ──────────────────────────────────────────
    "random_forest": lambda: _p(tfidf_word(), RandomForestClassifier()),
    "extra_trees": lambda: _p(tfidf_word(), ExtraTreesClassifier()),
    "gradient_boosting": lambda: _p(tfidf_word(), GradientBoostingClassifier()),
    "hist_gbm": _hist_gbm,
    "decision_tree": lambda: _p(tfidf_word(), DecisionTreeClassifier()),
    # ── neural ─────────────────────────────────────────────────────
    "mlp_shallow": lambda: _p(tfidf_word(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=500)),
    # ── voting / stacking / unions ─────────────────────────────────
    "voting_soft": _voting_soft,
    "stacking": _stacking,
    "union_logreg": lambda: _p(char_word_union(), LogisticRegression(max_iter=1000)),
    # ── reduced-dim ────────────────────────────────────────────────
    "lsa_logreg": lambda: _p(lsa(50, tfidf_word()), LogisticRegression(max_iter=1000)),
    "lsa_linear_svc": lambda: _p(lsa(50), _cal(LinearSVC())),
    "lsa_rbf_svc": lambda: _p(lsa(50), SVC(kernel="rbf", probability=True)),
    "nmf_logreg": lambda: _p(nmf(50), LogisticRegression(max_iter=1000)),
    "lda_logreg": lambda: _p(lda_topics(20), LogisticRegression(max_iter=1000)),
    # ── online / hashing ───────────────────────────────────────────
    "hashing_sgd_log": lambda: _p(hashing_word(), SGDClassifier(loss="log_loss")),
    "hashing_sgd_hinge": lambda: _p(hashing_word(), _cal(SGDClassifier(loss="hinge"))),
    # ── extra naive bayes ──────────────────────────────────────────
    "complement_nb_count": lambda: _p(count_word(), ComplementNB()),
    # ── penalty / loss sweeps ──────────────────────────────────────
    "logreg_l1": lambda: _p(tfidf_word(), LogisticRegression(penalty="l1", solver="saga", max_iter=2000)),
    "logreg_elasticnet": lambda: _p(
        tfidf_word(),
        LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=2000),
    ),
    "linear_svc_hinge": lambda: _p(tfidf_word(), _cal(LinearSVC(loss="hinge"))),
    "sgd_modified_huber": lambda: _p(tfidf_word(), SGDClassifier(loss="modified_huber")),
    # ── multiclass strategies ──────────────────────────────────────
    "ovr_linear_svc": lambda: _p(tfidf_word(), _cal(OneVsRestClassifier(LinearSVC()))),
    "ovo_linear_svc": lambda: _p(tfidf_word(), _cal(OneVsOneClassifier(LinearSVC()))),
    # ── bagging / nu-svc ───────────────────────────────────────────
    "bagging_logreg": lambda: _p(tfidf_word(), BaggingClassifier(LogisticRegression(max_iter=1000))),
    "nusvc": lambda: _p(tfidf_word(), NuSVC(probability=True)),
    # ── discriminant analysis (require dense input) ────────────────
    "lda_classifier": lambda: Pipeline([
        ("feat", tfidf_word()),
        ("dense", FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)),
        ("clf", LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")),
    ]),
    "qda_classifier": lambda: Pipeline([
        ("feat", lsa(20)),
        ("clf", QuadraticDiscriminantAnalysis(reg_param=0.5)),
    ]),
    # ── feature-engineering ────────────────────────────────────────
    "text_stats_logreg": lambda: _p(text_stats(), LogisticRegression(max_iter=1000)),
    "union_text_stats_logreg": lambda: _p(
        feature_union(tfidf_word(), text_stats()), LogisticRegression(max_iter=1000),
    ),
}


# Group tags — set in B2 too; declared here so the registry knows about them.
_GROUPS: Dict[str, Set[str]] = {
    "naive_bayes": {"nb_multinomial", "nb_complement", "nb_bernoulli"},
    "linear": {
        "logreg", "logreg_char", "linear_svc", "linear_svc_char",
        "sgd_log", "sgd_hinge", "passive_aggressive", "perceptron", "ridge",
        "logreg_l1", "logreg_elasticnet", "linear_svc_hinge",
        "sgd_modified_huber",
    },
    "kernel": {"rbf_svc", "knn", "nusvc", "lsa_rbf_svc"},
    "tree": {
        "random_forest", "extra_trees", "gradient_boosting",
        "hist_gbm", "decision_tree", "bagging_logreg",
    },
    "neural": {"mlp_shallow"},
    "ensemble": {"voting_soft", "stacking", "union_logreg", "bagging_logreg"},
    "reduced_dim": {
        "lsa_logreg", "lsa_linear_svc", "lsa_rbf_svc",
        "nmf_logreg", "lda_logreg",
    },
    "online": {
        "hashing_sgd_log", "hashing_sgd_hinge",
        "sgd_log", "sgd_hinge", "sgd_modified_huber",
    },
    "strategy": {"ovr_linear_svc", "ovo_linear_svc"},
    "feature_engineering": {"text_stats_logreg", "union_text_stats_logreg"},
    "discriminant": {"lda_classifier", "qda_classifier"},
}
_GROUPS["naive_bayes"].add("complement_nb_count")
_GROUPS["linear"].update({"hashing_sgd_log", "hashing_sgd_hinge"})


class _Registry:
    def __init__(self):
        self._items: Dict[str, Callable[[], Pipeline]] = {}
        self._groups: Dict[str, Set[str]] = {}

    def register(self, name: str, factory: Callable[[], Pipeline], *, group: str = None) -> None:
        self._items[name] = factory
        if group is not None:
            self._groups.setdefault(group, set()).add(name)

    def add_to_group(self, group: str, name: str) -> None:
        self._groups.setdefault(group, set()).add(name)

    def build(self, name: str) -> Pipeline:
        if name not in self._items:
            raise KeyError(f"unknown baseline: {name}")
        return self._items[name]()

    def names(self):
        return list(self._items.keys())

    def groups(self) -> Dict[str, Set[str]]:
        """Return group → baseline names mapping."""
        # filter out any names that no longer exist in the registry
        return {g: {n for n in members if n in self._items} for g, members in self._groups.items()}

    def in_group(self, name: str) -> Set[str]:
        """Return all groups a baseline belongs to."""
        return {g for g, members in self._groups.items() if name in members}

    def resolve(self, selector: str) -> list:
        """Resolve a name or ``@group`` selector to a list of baseline names."""
        if selector.startswith("@"):
            grp = selector[1:]
            if grp == "all":
                return self.names()
            if grp not in self._groups:
                raise KeyError(f"unknown group: {grp}")
            return sorted(n for n in self._groups[grp] if n in self._items)
        if selector not in self._items:
            raise KeyError(f"unknown baseline: {selector}")
        return [selector]

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


BASELINES = _Registry()

for _name, _factory in BASELINE_SPECS.items():
    BASELINES.register(_name, _factory)

for _group, _members in _GROUPS.items():
    for _m in _members:
        BASELINES.add_to_group(_group, _m)


def default() -> Pipeline:
    return BASELINES.build("linear_svc")
