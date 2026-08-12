"""Random + halving-random backends."""

from __future__ import annotations

from typing import Any, Dict


def run(*, factory, param_space, X, y, backend, scoring, cv, n_iter, seed, n_jobs, verbose) -> Dict[str, Any]:
    from sklearn.model_selection import RandomizedSearchCV

    estimator = factory()
    if backend == "halving_random":
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.model_selection import HalvingRandomSearchCV
        searcher = HalvingRandomSearchCV(
            estimator, param_space, scoring=scoring, cv=cv,
            random_state=seed, n_candidates=n_iter, n_jobs=n_jobs, verbose=verbose,
        )
    else:
        searcher = RandomizedSearchCV(
            estimator, param_space, scoring=scoring, cv=cv,
            random_state=seed, n_iter=n_iter, n_jobs=n_jobs, verbose=verbose,
        )
    searcher.fit(X, y)
    return {
        "best_params": searcher.best_params_,
        "best_score": searcher.best_score_,
        "best_estimator": searcher.best_estimator_,
        "cv_results": dict(searcher.cv_results_),
        "n_evaluations": len(searcher.cv_results_.get("params", [])),
    }
