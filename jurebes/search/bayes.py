"""Bayesian search backend — scikit-optimize, optional extra."""

from __future__ import annotations

from typing import Any, Dict


def run(*, factory, param_space, X, y, backend, scoring, cv, n_iter, seed, n_jobs, verbose) -> Dict[str, Any]:
    try:
        from skopt import BayesSearchCV  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "install jurebes[search-bayes] to use the bayes backend"
        ) from e
    from skopt import BayesSearchCV

    estimator = factory()
    searcher = BayesSearchCV(
        estimator, param_space, scoring=scoring, cv=cv,
        random_state=seed, n_iter=n_iter, n_jobs=n_jobs, verbose=verbose,
    )
    searcher.fit(X, y)
    return {
        "best_params": dict(searcher.best_params_),
        "best_score": searcher.best_score_,
        "best_estimator": searcher.best_estimator_,
        "cv_results": dict(searcher.cv_results_),
        "n_evaluations": len(searcher.cv_results_.get("params", [])),
    }
