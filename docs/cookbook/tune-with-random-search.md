# Cookbook: tune a baseline with random search

A reasonable winner from a baseline comparison can usually be improved by 0.5–2 macro-F1 points with hyperparameter tuning. Random search is the right default — cheap, embarrassingly parallel, and surprisingly competitive with grid search.

## Prerequisites

```bash
pip install jurebes[hf]
```

(`hf` is only needed if you want the canonical-dataset CLI selectors; not required for local CSV.)

## Script

```python
from jurebes.search import search, spaces
from jurebes.datasets.canonical import load_banking77
from sklearn.model_selection import train_test_split

X, y = load_banking77()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0,
)

# Default predefined space for logreg
space = spaces.for_baseline("logreg")
print("search space:", space)
# {'feat__ngram_range': [(1, 1), (1, 2), (1, 3)],
#  'feat__min_df': [1, 2, 3],
#  'clf__C': [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]}

r = search(
    "logreg", space, X_train, y_train,
    backend="random", n_iter=40, cv=5,
    scoring="f1_macro", seed=0,
)

print(f"best macro-F1 (CV): {r.best_score:.4f}")
print(f"best params: {r.best_params}")
print(f"wall time: {r.wall_time_seconds:.1f}s ({r.n_evaluations} evaluations)")

# Evaluate the tuned model on the held-out test set.
preds = [r.best_estimator.predict(x).intent for x in X_test]
from sklearn.metrics import f1_score
test_macro = f1_score(y_test, preds, average="macro", zero_division=0)
print(f"held-out macro-F1: {test_macro:.4f}")

# Persist.
r.best_estimator.save("banking_logreg_tuned.joblib")
```

## Reading the output

- `best_score` is the CV-averaged macro-F1 of the best-found hyperparameter combination during the search.
- `best_params` are the chosen hyperparameter values (sklearn pipeline `step__param` keys).
- `wall_time_seconds` and `n_evaluations` track cost.
- The held-out test F1 is the honest unbiased estimate. Compare to the untuned baseline's held-out F1 to quantify uplift.

If `best_score` is much higher than `held-out test`, the search overfit to the CV folds. Reduce `n_iter`, increase `cv`, or expand the test split.

## Picking a backend

| backend | when |
| --- | --- |
| `random` | default, almost always the right choice |
| `grid` | small discrete space (≤ 50 combinations); want exhaustive |
| `halving_random` | large space; successive-halving prunes cheaply |
| `bayes` | continuous space, expensive evaluations, `jurebes[search-bayes]` installed |
| `genetic` | mixed continuous/discrete, `jurebes[search-genetic]` installed |

See [../search.md](../search.md) for the full backend matrix.

## Custom search space

`spaces.for_baseline()` covers only a curated subset of baselines. For others, supply your own:

```python
my_space = {
    "feat__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "feat__sublinear_tf": [False, True],
    "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
}
r = search("logreg", my_space, X_train, y_train,
           backend="random", n_iter=40, cv=5)
```

The keys are sklearn `step__param` strings — `feat__*` addresses the featurizer step, `clf__*` the classifier step. For non-trivial pipelines, inspect available parameters:

```python
from jurebes.baselines import BASELINES
print(BASELINES.build("logreg").get_params())
```

## CLI equivalent

```bash
jurebes search --dataset @banking77 --baseline logreg \
  --backend random --n-iter 40 --cv 5 --scoring f1_macro \
  --out banking_logreg_tuned.joblib
```

## Caveats

- Tuning on the same CV folds you will report on inflates the reported score. Use a held-out test set (as in the script above) or nested CV ([../theory/cross-validation.md](../theory/cross-validation.md)).
- `n_iter=40` is a starting point. Expand the search budget when the space is high-dimensional; reduce when each fit is expensive.
- The Bayes and genetic backends need different dimension types (skopt `Real`/`Integer`/`Categorical`, sklearn-genetic `Continuous`/`Integer`/`Categorical`). See [../search.md](../search.md).

---
- Back to [docs index](../index.md)
