# `jurebes.semi_supervised` API reference

Pure-sklearn / pure-jurebes semi-supervised learning primitives.

## Pseudo-labeling

### `pseudo_label(clf, X_unlabeled) -> list[tuple[str, float]]`

Return `(top_label, top_confidence)` per utterance, parallel to
`X_unlabeled`. `clf` must be a fitted `jurebes.IntentClassifier`.
Reuses `jurebes.active_learning.uncertainty_scores`.

### `select_high_confidence(scored, *, strategy="global_top_k", k=10, threshold=0.0) -> list[int]`

Return indices into `scored` (output of `pseudo_label`) that should be
promoted into the labeled pool.

- `strategy`: key into `SELECTION_STRATEGIES`. Raises `KeyError` on
  unknown keys.
- `k`: budget. With `global_top_k` this is the total cap; with
  `per_class_quota` it is the cap per class.
- `threshold`: minimum confidence to consider.

### `SELECTION_STRATEGIES`

Registry mapping strategy name to a callable
`(scored, *, k, threshold) -> list[int]`.

| Key | Behaviour |
| --- | --- |
| `global_top_k` | Picks up to `k` highest-confidence indices ≥ threshold |
| `per_class_quota` | Picks up to `k` per predicted class ≥ threshold |

## `self_train`

```python
self_train(
    clf, labeled_X, labeled_y, unlabeled_X,
    *,
    confidence_threshold=0.8,
    selection="global_top_k",
    k_per_round=10,
    max_rounds=10,
    threshold_schedule=None,
    eval_X=None,
    eval_y=None,
    early_stop_patience=3,
) -> SelfTrainResult
```

`clf` is a template `IntentClassifier`; each round clones its
estimator via `sklearn.base.clone`, re-adds intents from the current
labeled pool, and fits. `threshold_schedule(round_idx, current) ->
new_threshold` is invoked once per round (0-based) before pseudo-labeling.

Returns `SelfTrainResult` with:

- `classifier`: the final fitted classifier.
- `labeled_X`, `labeled_y`: final pool (seed + promoted).
- `added_per_round`: ints.
- `eval_f1_per_round`: floats (empty if no eval set).
- `stopped_early`: bool.
- `wall_time_s`: float.

## `co_train`

```python
co_train(
    view_a_factory, view_b_factory,
    labeled_X, labeled_y, unlabeled_X,
    *,
    confidence_threshold=0.8,
    selection="global_top_k",
    k_per_round=10,
    max_rounds=10,
    eval_X=None,
    eval_y=None,
) -> CoTrainResult
```

`view_a_factory` / `view_b_factory` are zero-arg callables that
return a fresh, unfitted `IntentClassifier`. Each round both views
pseudo-label the pool; the union of high-confidence picks
(de-duplicated by global index, view A wins ties) is promoted into
the shared labeled pool that both views refit against.

Returns `CoTrainResult` with `view_a`, `view_b`, `labeled_X`,
`labeled_y`, `added_per_round_view_a`, `added_per_round_view_b`,
`eval_f1_per_round_view_a`, `eval_f1_per_round_view_b`, `wall_time_s`.

## `label_propagation`

```python
label_propagation(
    labeled_X, labeled_y, unlabeled_X,
    *,
    featurizer=None,
    method="propagation",
    **kwargs,
) -> LabelPropResult
```

Thin wrapper over `sklearn.semi_supervised.LabelPropagation` /
`LabelSpreading`. The featurizer defaults to
`jurebes.featurizers.tfidf_word()`; the matrix is densified before
being passed to sklearn. `**kwargs` forward to the estimator
constructor (e.g. `kernel="knn"`, `n_neighbors=7`).

Returns `LabelPropResult` with `predicted_labels`,
`predicted_confidences`, `method`, `wall_time_s`, `model`.
