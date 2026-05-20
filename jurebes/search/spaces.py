"""Predefined search spaces for the baseline registry.

Each pipeline in :data:`jurebes.baselines.BASELINES` uses the ``feat``
+ ``clf`` step naming convention from :func:`jurebes.baselines._p`, so
parameter keys here follow ``feat__<param>`` and ``clf__<param>``.
"""

from __future__ import annotations

from typing import Any, Dict


_TFIDF_FEAT = {
    "feat__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "feat__min_df": [1, 2, 3],
    "feat__max_df": [0.95, 1.0],
}

_TFIDF_CHAR_FEAT = {
    "feat__ngram_range": [(2, 4), (3, 5), (3, 6), (4, 6)],
    "feat__min_df": [1, 2],
}

_COUNT_FEAT = {
    "feat__ngram_range": [(1, 1), (1, 2)],
    "feat__min_df": [1, 2],
}

_HASHING_FEAT = {
    "feat__n_features": [2 ** 16, 2 ** 18, 2 ** 20],
    "feat__ngram_range": [(1, 1), (1, 2)],
}


def _merge(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in dicts:
        out.update(d)
    return out


_SPACES: Dict[str, Dict[str, Any]] = {
    # ── naive bayes ────────────────────────────────────────────────
    "nb_multinomial": _merge(_TFIDF_FEAT, {"clf__alpha": [0.01, 0.1, 0.5, 1.0]}),
    "nb_complement":  _merge(_TFIDF_FEAT, {"clf__alpha": [0.01, 0.1, 0.5, 1.0],
                                            "clf__norm": [True, False]}),
    "nb_bernoulli":   _merge(_COUNT_FEAT, {"clf__alpha": [0.01, 0.1, 0.5, 1.0],
                                            "clf__binarize": [0.0, 0.5]}),
    "complement_nb_count": _merge(_COUNT_FEAT, {"clf__alpha": [0.01, 0.1, 0.5, 1.0]}),

    # ── linear: logreg variants ────────────────────────────────────
    "logreg":             _merge(_TFIDF_FEAT, {"clf__C": [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]}),
    "logreg_char":        _merge(_TFIDF_CHAR_FEAT, {"clf__C": [0.1, 0.5, 1.0, 2.0, 4.0]}),
    "logreg_l1":          _merge(_TFIDF_FEAT, {"clf__C": [0.1, 0.5, 1.0, 2.0, 4.0]}),
    "logreg_elasticnet":  _merge(_TFIDF_FEAT, {"clf__C": [0.1, 0.5, 1.0, 2.0],
                                                "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}),

    # ── linear: SVC and friends (CalibratedClassifierCV wraps clf) ─
    "linear_svc":       _merge(_TFIDF_FEAT, {"clf__estimator__C": [0.1, 0.5, 1.0, 2.0, 4.0]}),
    "linear_svc_char":  _merge(_TFIDF_CHAR_FEAT, {"clf__estimator__C": [0.1, 0.5, 1.0, 2.0]}),
    "linear_svc_hinge": _merge(_TFIDF_FEAT, {"clf__estimator__C": [0.1, 0.5, 1.0, 2.0]}),
    "ridge":            _merge(_TFIDF_FEAT, {"clf__estimator__alpha": [0.1, 1.0, 10.0]}),
    "perceptron":       _merge(_TFIDF_FEAT, {"clf__estimator__alpha": [1e-5, 1e-4, 1e-3]}),
    "passive_aggressive": _merge(_TFIDF_FEAT, {"clf__estimator__C": [0.1, 0.5, 1.0, 2.0]}),

    # ── online ─────────────────────────────────────────────────────
    "sgd_log":            _merge(_TFIDF_FEAT, {"clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2]}),
    "sgd_hinge":          _merge(_TFIDF_FEAT, {"clf__estimator__alpha": [1e-5, 1e-4, 1e-3, 1e-2]}),
    "sgd_modified_huber": _merge(_TFIDF_FEAT, {"clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2]}),
    "hashing_sgd_log":    _merge(_HASHING_FEAT, {"clf__alpha": [1e-5, 1e-4, 1e-3]}),
    "hashing_sgd_hinge":  _merge(_HASHING_FEAT, {"clf__estimator__alpha": [1e-5, 1e-4, 1e-3]}),

    # ── kernel ─────────────────────────────────────────────────────
    "rbf_svc": _merge(_TFIDF_FEAT, {"clf__C": [0.5, 1.0, 2.0, 4.0],
                                     "clf__gamma": ["scale", 0.1, 1.0]}),
    "nusvc":   _merge(_TFIDF_FEAT, {"clf__nu": [0.1, 0.3, 0.5, 0.7],
                                     "clf__gamma": ["scale", 0.1, 1.0]}),

    # ── knn ────────────────────────────────────────────────────────
    "knn": _merge(_TFIDF_FEAT, {"clf__n_neighbors": [3, 5, 7, 11],
                                 "clf__weights": ["uniform", "distance"],
                                 "clf__metric": ["cosine", "euclidean"]}),

    # ── tree ensembles ─────────────────────────────────────────────
    "random_forest":     _merge(_TFIDF_FEAT, {"clf__n_estimators": [100, 200, 400],
                                               "clf__max_depth": [None, 10, 30],
                                               "clf__min_samples_leaf": [1, 2, 4]}),
    "extra_trees":       _merge(_TFIDF_FEAT, {"clf__n_estimators": [100, 200, 400],
                                               "clf__max_depth": [None, 10, 30]}),
    "gradient_boosting": _merge(_TFIDF_FEAT, {"clf__n_estimators": [100, 200],
                                               "clf__learning_rate": [0.05, 0.1, 0.2],
                                               "clf__max_depth": [3, 5, 7]}),
    "decision_tree":     _merge(_TFIDF_FEAT, {"clf__max_depth": [None, 10, 30],
                                               "clf__min_samples_leaf": [1, 2, 4]}),
    "bagging_logreg":    _merge(_TFIDF_FEAT, {"clf__n_estimators": [10, 25, 50]}),

    # ── multiclass strategies ──────────────────────────────────────
    "ovr_linear_svc": _merge(_TFIDF_FEAT,
                              {"clf__estimator__estimator__C": [0.1, 1.0, 2.0]}),
    "ovo_linear_svc": _merge(_TFIDF_FEAT,
                              {"clf__estimator__estimator__C": [0.1, 1.0, 2.0]}),

    # ── neural ─────────────────────────────────────────────────────
    "mlp_shallow": _merge(_TFIDF_FEAT, {
        "clf__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32)],
        "clf__alpha": [1e-5, 1e-4, 1e-3],
    }),

    # ── reduced-dim (downstream clf only; feat steps are nested) ──
    "lsa_logreg": {"clf__C": [0.5, 1.0, 2.0, 4.0]},
    "nmf_logreg": {"clf__C": [0.5, 1.0, 2.0, 4.0]},

    # ── feature engineering ────────────────────────────────────────
    "text_stats_logreg":       {"clf__C": [0.1, 0.5, 1.0, 2.0, 4.0]},
    "union_logreg":            {"clf__C": [0.1, 0.5, 1.0, 2.0, 4.0]},
    "union_text_stats_logreg": {"clf__C": [0.1, 0.5, 1.0, 2.0, 4.0]},
}


def for_baseline(name: str) -> Dict[str, Any]:
    """Return the canonical search space for a baseline.

    Raises ``KeyError`` for baselines without a predefined space; the
    common case is one of the discriminant / categorical / autoencoder
    variants where parameter coupling makes a curated space brittle.
    """
    if name not in _SPACES:
        raise KeyError(f"no predefined search space for baseline {name!r}; "
                       f"available: {sorted(_SPACES.keys())}")
    out: Dict[str, Any] = {}
    for k, v in _SPACES[name].items():
        out[k] = list(v) if isinstance(v, list) else v
    return out


def available() -> list:
    """Return the list of baseline names with predefined spaces."""
    return sorted(_SPACES.keys())
