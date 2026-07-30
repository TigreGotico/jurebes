# Search subsystem

`jurebes.search` provides a unified `search()` entry point over multiple
hyperparameter-search backends. Every backend returns a
`SearchResult` whose `best_estimator` is a fitted
`IntentClassifier`, ready to call `predict()` on or save with
`IntentClassifier.save(path)`.

## Quickstart

```python
from jurebes.search import search, spaces
from jurebes.datasets import load_csv

X, y = load_csv("data.csv")
r = search("logreg", spaces.for_baseline("logreg"), X, y,
           backend="random", n_iter=40, cv=5)
print(r.best_score, r.best_params)
```

## Backends

| backend          | dep                              | notes                                                       |
| ---------------- | -------------------------------- | ----------------------------------------------------------- |
| `grid`           | core                             | exhaustive `GridSearchCV`, cost equals the product of grid sizes. |
| `halving_grid`   | core (experimental)              | successive-halving over the grid, cheapest on big spaces.   |
| `random`         | core                             | `RandomizedSearchCV`, the default. Trade `n_iter` for cost. |
| `halving_random` | core (experimental)              | successive-halving sampling.                                |
| `bayes`          | `jurebes[search-bayes]` (skopt)  | Bayesian optimization, needs Real/Categorical Dimensions.   |
| `genetic`        | `jurebes[search-genetic]`        | genetic algorithm through sklearn-genetic-opt.              |

Optional backends emit a clear `ImportError("install jurebes[<extra>] to use the <name> backend")` if the dependency is missing.

## Per-backend snippets

```python
from jurebes.search import search

# grid - exhaustive over a discrete space
search("logreg", {"clf__C": [0.1, 1.0, 10.0]}, X, y, backend="grid", cv=5)

# halving_grid - successive-halving over a grid; n_iter is treated as n_candidates
search("logreg", {"clf__C": [0.1, 1.0, 10.0]}, X, y,
       backend="halving_grid", n_iter=3, cv=5)

# random - sampled n_iter draws
search("logreg", {"clf__C": [0.01, 0.1, 1.0, 10.0]}, X, y,
       backend="random", n_iter=40, cv=5)

# halving_random - successive-halving sampling; n_iter -> n_candidates
search("logreg", {"clf__C": [0.01, 0.1, 1.0, 10.0]}, X, y,
       backend="halving_random", n_iter=20, cv=5)

# bayes - uses skopt dimensions
from skopt.space import Real, Integer, Categorical
search("logreg",
       {"clf__C": Real(1e-3, 1e2, prior="log-uniform"),
        "feat__ngram_range": Categorical([(1, 1), (1, 2)])},
       X, y, backend="bayes", n_iter=30, cv=5)

# genetic - uses sklearn-genetic-opt dimensions
from sklearn_genetic.space import Continuous, Integer, Categorical
search("logreg",
       {"clf__C": Continuous(1e-3, 1e2, distribution="log-uniform"),
        "feat__ngram_range": Categorical([(1, 1), (1, 2)])},
       X, y, backend="genetic", n_iter=30, cv=5)
```

### `n_iter` vs `n_candidates`

The halving backends (`halving_grid`, `halving_random`) wrap sklearn's `Halving*SearchCV`, which accepts `n_candidates` rather than `n_iter`. Jurebes accepts a unified `n_iter` argument across every backend and forwards it as `n_candidates` for the halving variants. For `grid`, Jurebes ignores `n_iter` because the grid is exhaustive. For `random`, `bayes`, and `genetic`, `n_iter` is the number of candidate evaluations.

## Predefined spaces

`jurebes.search.spaces.for_baseline(name)` returns a known-good search space for a registered baseline:

```python
from jurebes.search import spaces
spaces.available()                # ['linear_svc', 'logreg', 'nb_multinomial', 'sgd_log']
spaces.for_baseline("nb_multinomial")
# {'feat__ngram_range': [(1, 1), (1, 2)],
#  'feat__min_df': [1, 2],
#  'clf__alpha': [0.01, 0.1, 0.5, 1.0]}
```

For backends like `bayes`, translate categorical lists into
`skopt.space.{Real, Integer, Categorical}` dimensions. For `genetic`, use
`sklearn_genetic.space.{Continuous, Integer, Categorical}`.

`spaces.for_baseline()` covers only a curated subset of registered baselines (see `spaces.available()`). Calling it with a baseline outside that set raises `KeyError`. Provide your own space dict in that case.

## CLI

```bash
jurebes search --dataset data.csv --baseline logreg \
  --backend random --n-iter 40 --cv 5 \
  --out best_logreg.joblib
```

`--space` accepts a Python literal dict overriding the predefined space:

```bash
jurebes search --dataset data.csv --baseline nb_multinomial \
  --backend grid --space "{'clf__alpha': [0.1, 0.5, 1.0]}"
```

---
[Home](index.md)
