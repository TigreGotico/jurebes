"""Public ``search()`` entry point + SearchResult dataclass."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Union

from jurebes.baselines import BASELINES
from jurebes.core import IntentClassifier


@dataclass
class SearchResult:
    """Outcome of a hyperparameter search.

    Attributes:
        best_params: dict of best-found hyperparameter values.
        best_score: cross-validated score of the best estimator.
        best_estimator: a fitted :class:`IntentClassifier` wrapping the
            best sklearn pipeline.
        cv_results: the raw sklearn ``cv_results_`` dict (or backend
            equivalent).
        wall_time_seconds: total search wall-clock time.
        n_evaluations: number of candidate evaluations performed.
        backend: the backend name used.
    """

    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    best_estimator: Any = None
    cv_results: Dict[str, Any] = field(default_factory=dict)
    wall_time_seconds: float = 0.0
    n_evaluations: int = 0
    backend: str = ""


_BACKENDS = ("grid", "halving_grid", "random", "halving_random", "bayes", "genetic")


def _resolve_factory(estimator_or_factory: Union[str, Callable]) -> Callable:
    if isinstance(estimator_or_factory, str):
        name = estimator_or_factory
        return lambda: BASELINES.build(name)
    if callable(estimator_or_factory):
        return estimator_or_factory
    # assume already-constructed estimator
    est = estimator_or_factory
    return lambda: est


def search(
    estimator_or_factory: Union[str, Callable, Any],
    param_space: Dict[str, Any],
    X: List[str],
    y: List[str],
    *,
    backend: str = "random",
    scoring: str = "f1_macro",
    cv: int = 5,
    n_iter: int = 50,
    seed: int = 0,
    n_jobs: int = -1,
    verbose: int = 0,
) -> SearchResult:
    """Run a hyperparameter search.

    Args:
        estimator_or_factory: baseline name (string), zero-arg factory,
            or an already-constructed sklearn estimator/pipeline.
        param_space: search space — a dict whose keys are pipeline
            ``step__param`` strings. The value space format depends on
            the backend (grids/lists for grid/random; skopt Dimensions
            for bayes; sklearn-genetic spaces for genetic).
        X, y: training data.
        backend: one of ``grid``, ``halving_grid``, ``random``,
            ``halving_random``, ``bayes``, ``genetic``.
        scoring: sklearn scoring string.
        cv: cross-validation fold count.
        n_iter: number of candidates evaluated by random/bayes/genetic
            backends.
        seed: random seed.
        n_jobs: sklearn n_jobs.
        verbose: sklearn verbosity.

    Returns:
        :class:`SearchResult` with the best estimator wrapped as a
        fitted :class:`IntentClassifier`.

    Raises:
        ValueError: if the backend is unknown.
        ImportError: if an optional backend's dependency is missing.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose one of {_BACKENDS}")
    factory = _resolve_factory(estimator_or_factory)

    t0 = time.perf_counter()
    if backend in ("grid", "halving_grid"):
        from jurebes.search.grid import run as _run
    elif backend in ("random", "halving_random"):
        from jurebes.search.random import run as _run
    elif backend == "bayes":
        from jurebes.search.bayes import run as _run
    elif backend == "genetic":
        from jurebes.search.genetic import run as _run
    else:  # unreachable
        raise ValueError(backend)

    raw = _run(
        factory=factory, param_space=param_space, X=X, y=y,
        backend=backend, scoring=scoring, cv=cv, n_iter=n_iter,
        seed=seed, n_jobs=n_jobs, verbose=verbose,
    )
    wall = time.perf_counter() - t0

    # Wrap the best sklearn estimator into an IntentClassifier.
    best_sklearn = raw["best_estimator"]
    wrapped = IntentClassifier.__new__(IntentClassifier)
    wrapped.estimator = best_sklearn
    wrapped.tagger = None
    wrapped._samples = {}
    wrapped._entity_samples = {}
    wrapped._fitted = True
    from threading import RLock
    wrapped._lock = RLock()
    # Reconstruct samples per class so save/load round-trip is meaningful.
    for x, lbl in zip(X, y):
        wrapped._samples.setdefault(lbl, []).append(x)

    return SearchResult(
        best_params=dict(raw["best_params"]),
        best_score=float(raw["best_score"]),
        best_estimator=wrapped,
        cv_results=raw["cv_results"],
        wall_time_seconds=wall,
        n_evaluations=int(raw.get("n_evaluations", 0)),
        backend=backend,
    )
