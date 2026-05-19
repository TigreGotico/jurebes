"""Grid + halving-grid backends."""

from __future__ import annotations

from typing import Any, Dict


def run(*, factory, param_space, X, y, backend, scoring, cv, n_iter, seed, n_jobs, verbose) -> Dict[str, Any]:
    from sklearn.model_selection import GridSearchCV

    estimator = factory()
    if backend == "halving_grid":
        # Successive halving is still gated as experimental in sklearn.
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.model_selection import HalvingGridSearchCV
        searcher = HalvingGridSearchCV(
            estimator, param_space, scoring=scoring, cv=cv,
            random_state=seed, n_jobs=n_jobs, verbose=verbose,
        )
    else:
        searcher = GridSearchCV(
            estimator, param_space, scoring=scoring, cv=cv,
            n_jobs=n_jobs, verbose=verbose,
        )
    searcher.fit(X, y)
    return {
        "best_params": searcher.best_params_,
        "best_score": searcher.best_score_,
        "best_estimator": searcher.best_estimator_,
        "cv_results": dict(searcher.cv_results_),
        "n_evaluations": len(searcher.cv_results_.get("params", [])),
    }
