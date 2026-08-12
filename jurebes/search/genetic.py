"""Genetic-algorithm search backend — sklearn-genetic-opt, optional extra."""

from __future__ import annotations

from typing import Any, Dict


def run(*, factory, param_space, X, y, backend, scoring, cv, n_iter, seed, n_jobs, verbose) -> Dict[str, Any]:
    try:
        from sklearn_genetic import GASearchCV  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "install jurebes[search-genetic] to use the genetic backend"
        ) from e
    from sklearn_genetic import GASearchCV

    estimator = factory()
    # n_iter maps to generations*population_size; expose generations directly
    # and let population scale with cv folds.
    searcher = GASearchCV(
        estimator=estimator,
        param_grid=param_space,
        scoring=scoring,
        cv=cv,
        population_size=max(4, n_iter // 5),
        generations=max(2, n_iter // max(4, n_iter // 5)),
        n_jobs=n_jobs,
        verbose=bool(verbose),
    )
    searcher.fit(X, y)
    return {
        "best_params": dict(searcher.best_params_),
        "best_score": searcher.best_score_,
        "best_estimator": searcher.best_estimator_,
        "cv_results": dict(getattr(searcher, "cv_results_", {})),
        "n_evaluations": int(getattr(searcher, "n_iterations_", 0)),
    }
