# Custom scoring metrics

The benchmark harness exposes a registry of scoring callables at `jurebes.benchmark.scoring.SCORERS`. Built-ins: `f1_macro`, `f1_micro`, `accuracy`, `balanced_accuracy`, `log_loss`, `top_k_accuracy`.

## Callable signature

A scorer is a callable:

```python
def scorer(y_true, y_pred, y_proba_or_None, classes) -> float:
    ...
```

| argument | type | notes |
| --- | --- | --- |
| `y_true` | `list[str]` | gold labels |
| `y_pred` | `list[str]` | predicted labels |
| `y_proba_or_None` | `np.ndarray` of shape `(N, K)` or `None` | predicted probabilities; `None` when the estimator lacks `predict_proba` |
| `classes` | `list[str]` | column ordering for `y_proba_or_None` |

Return a single float. Higher should mean "better" (the harness uses the value for ranking).

## Registering a custom scorer

```python
from jurebes.benchmark.scoring import SCORERS

def per_class_min_recall(y_true, y_pred, _probs=None, classes=None):
    """Worst per-class recall — emphasises the most-confused class."""
    from sklearn.metrics import recall_score
    recs = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    return float(recs.min())

SCORERS["min_recall"] = per_class_min_recall
```

Use it via the `scoring=` argument:

```python
from jurebes.benchmark import compare
result = compare(["logreg", "linear_svc"], X, y, k=5,
                 scoring=("f1_macro", "min_recall"))
```

The metric appears as `min_recall` in `RunResult.extra_scores` and `RunResult.fold_scores`.

## Using probabilities

Scorers that consume `y_proba_or_None` must gracefully handle `None`:

```python
def expected_log_score(y_true, _y_pred, probs, classes):
    if probs is None or classes is None:
        return float("nan")
    import numpy as np
    cls_idx = {c: i for i, c in enumerate(classes)}
    return float(np.mean([
        np.log(max(probs[i, cls_idx[y]], 1e-12))
        for i, y in enumerate(y_true)
    ]))
SCORERS["mean_log_prob"] = expected_log_score
```

`compare()` returns `nan` for that metric on non-probabilistic estimators, so reports still sort correctly when mixed.

## Where the registry is consulted

- `cross_validate(..., scoring=(...))` resolves each name via `jurebes.benchmark.scoring.get(name)` per fold.
- `train_test(..., scoring=(...))` does the same on the single held-out set.
- `compare(..., scoring=(...))` delegates to `cross_validate`.

There is no path through the CLI that injects custom scorers — register them in a Python wrapper script around the harness:

```python
from jurebes.benchmark.scoring import SCORERS
SCORERS["min_recall"] = per_class_min_recall
# ... then call compare(..., scoring=("min_recall",))
```

## Listing available metrics

```python
from jurebes.benchmark.scoring import available
print(available())
# ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_micro', 'log_loss', 'top_k_accuracy']
```

After custom registration, the new name appears in the list automatically.

## Why "higher is better"

Many sklearn metrics use the opposite convention (`log_loss` lower is better). The jurebes scoring contract reports the raw value — `log_loss` is reported as positive numbers, with lower being better. Downstream sort helpers in `to_markdown(result, sort_by="log_loss")` honour ascending order for known "lower is better" metrics. For custom metrics, sort by descending value or invert the metric (`1 / loss`) if needed.

---
- Back to [docs index](../index.md)
