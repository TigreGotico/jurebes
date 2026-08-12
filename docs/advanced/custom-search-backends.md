# Custom search backends

The `jurebes.search` subsystem dispatches `search(..., backend=...)` to one of six bundled backends: `grid`, `halving_grid`, `random`, `halving_random`, `bayes`, `genetic`. Adding a seventh requires three pieces: a module, an entry in the dispatch ladder, and an entry in the backends tuple.

## Anatomy of a backend module

Each `jurebes/search/<name>.py` exposes a single function:

```python
def run(*, factory, param_space, X, y, backend,
        scoring, cv, n_iter, seed, n_jobs, verbose) -> dict:
    ...
```

The return dict must contain:

| key | type | meaning |
| --- | --- | --- |
| `best_params` | `dict[str, Any]` | best-found hyperparameter values |
| `best_score` | `float` | cross-validated score under `scoring` |
| `best_estimator` | fitted sklearn pipeline | the chosen pipeline already refit on the full training set |
| `cv_results` | `dict` | the raw cv_results_ (or backend equivalent) for caller introspection |
| `n_evaluations` | `int` | number of candidate evaluations performed |

`factory` is a zero-argument callable returning a fresh pipeline; call it once per candidate (or rely on the backend's internal cloning).

## Reference: the random backend (paraphrased)

A typical implementation looks like:

```python
def run(*, factory, param_space, X, y, backend,
        scoring, cv, n_iter, seed, n_jobs, verbose):
    from sklearn.model_selection import RandomizedSearchCV

    base = factory()
    search = RandomizedSearchCV(
        base, param_distributions=param_space,
        n_iter=n_iter, scoring=scoring, cv=cv,
        random_state=seed, n_jobs=n_jobs, verbose=verbose,
        refit=True,
    )
    search.fit(X, y)
    return {
        "best_params": dict(search.best_params_),
        "best_score": float(search.best_score_),
        "best_estimator": search.best_estimator_,
        "cv_results": dict(search.cv_results_),
        "n_evaluations": len(search.cv_results_["params"]),
    }
```

## Adding a new backend

1. Write `jurebes/search/<my_backend>.py` exposing `run(...)` with the signature above.
2. Add `"my_backend"` to the `_BACKENDS` tuple at the top of `jurebes/search/api.py`.
3. Add a dispatch branch:

```
elif backend == "my_backend":
    from jurebes.search.my_backend import run as _run
```

4. If the backend depends on an optional library, lazy-import inside `run` and raise `ImportError("install jurebes[<extra>] to use the <name> backend")`.

5. Add the optional dependency under `[project.optional-dependencies]` in `pyproject.toml`.

## Optional-dependency pattern

```python
# jurebes/search/my_backend.py
def run(*, factory, param_space, X, y, backend,
        scoring, cv, n_iter, seed, n_jobs, verbose):
    try:
        import my_optional_lib
    except ImportError as e:
        raise ImportError(
            "install jurebes[search-my-backend] to use the my_backend backend"
        ) from e
    # … rest of impl
```

Defer the import until inside `run` so importing `jurebes.search.api` does not fail when the optional dep is missing.

## Best-estimator wrapping

`search()` itself wraps the returned `best_estimator` into a fitted `IntentClassifier`. Your backend's job is to return the *sklearn* pipeline; the IntentClassifier wrap happens once at the api layer regardless of backend.

## Testing a custom backend

Smoke test:

```python
from jurebes.search import search

space = {"clf__C": [0.1, 1.0, 10.0]}
r = search("logreg", space, X, y, backend="my_backend", cv=3, n_iter=5)
assert r.best_estimator.predict("hello").intent in set(y)
```

For more rigorous testing, include a unit test pinning the seed and checking `best_params` and `best_score` against expected values.

## Architectural notes

- The dispatcher is intentionally simple (an if-ladder) to keep the import graph thin. A registry-based design is possible but adds an import-time side effect that complicates lazy loading of optional deps.
- Backends are *not* expected to handle data validation — the caller is responsible for ensuring X/y shapes are correct.

---
- Back to [docs index](../index.md)
