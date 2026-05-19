# Search subsystem

`jurebes.search` provides a unified `search()` entry point over multiple
hyperparameter-search backends. Every backend returns a
:class:`SearchResult` whose ``best_estimator`` is a fitted
:class:`IntentClassifier` — ready to call ``predict()`` on or save with
``IntentClassifier.save(path)``.

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
| `grid`           | core                             | exhaustive `GridSearchCV`; cost = product of grid sizes.    |
| `halving_grid`   | core (experimental)              | successive-halving over the grid; cheapest on big spaces.   |
| `random`         | core                             | `RandomizedSearchCV` — default; trade `n_iter` for cost.    |
| `halving_random` | core (experimental)              | successive-halving sampling.                                |
| `bayes`          | `jurebes[search-bayes]` (skopt)  | Bayesian optimisation; needs Real/Categorical Dimensions.   |
| `genetic`        | `jurebes[search-genetic]`        | Genetic algorithm via sklearn-genetic-opt.                  |

Optional backends emit a clear `ImportError("install jurebes[<extra>] to use the <name> backend")` if the dependency is missing.

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

For backends like `bayes` you need to translate categorical lists into
`skopt.space.{Real, Categorical}` dimensions; for `genetic` use
`sklearn_genetic.space.{Continuous, Categorical}`.

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
