# Performance tuning

Two axes of performance matter for intent classifiers: training cost (offline, infrequent) and prediction latency (online, every utterance). This guide focuses on the prediction-latency axis since it is usually the binding constraint for voice-assistant deployments.

## Measure first

The benchmark harness reports pooled tail-latency percentiles:

```python
from jurebes.benchmark import compare, to_markdown

result = compare(
    ["logreg", "linear_svc", "nb_multinomial", "hashing_sgd_log", "rbf_svc"],
    X, y, k=5,
)
print(to_markdown(result, sort_by="predict_ms_p95_pooled", precision=3))
```

`predict_ms_p95_pooled` is the 95th-percentile latency over the flat pool of every per-prediction time across all folds — statistically meaningful for tail-latency. Prefer it over the per-fold `predict_ms_p95` (which is a mean of per-fold percentiles).

## Latency tactics

### Drop calibration when probabilities are not needed

`CalibratedClassifierCV(cv=3)` internally maintains and averages predictions across 3 folds. If `predict_proba` is not actually consumed downstream (e.g. you only use `predict`), use a baseline whose underlying estimator is natively probabilistic and skip the wrap:

```python
from sklearn.linear_model import LogisticRegression
from jurebes import IntentClassifier
clf = IntentClassifier(LogisticRegression(max_iter=1000), calibrate=False)
```

This roughly halves prediction time and model size compared to a calibrated `LinearSVC`.

### Use linear models

`logreg`, `linear_svc`, `nb_multinomial`, and the hashing-SGD variants predict in microseconds per utterance on CPU. Avoid `rbf_svc`, `mlp_shallow`, `voting_soft`, and `stacking` when latency matters — those can be 10–100x slower.

### Hashing featurizer for memory

`HashingVectorizer(n_features=2**18)` allocates a fixed-size hash table regardless of vocabulary size. The bundled `hashing_sgd_log` and `hashing_sgd_hinge` baselines use this. They are also online-friendly — no need to refit on vocabulary expansion.

### Reduce featurizer cost

The default `TfidfVectorizer` does Python-level tokenisation. For very high throughput, build a custom transformer over precomputed integer token IDs and skip the regex step entirely — but only after the rest of the pipeline has been verified as not the bottleneck.

## Memory tactics

### Sparse formats

TF-IDF matrices are sparse by default (scipy's CSR format). Most jurebes baselines preserve sparsity through the entire pipeline. Avoid the densifying `FunctionTransformer` step unless the downstream classifier requires it (`HistGBM`, MLP, discriminant-analysis baselines).

### Compact model size

joblib persists by default without compression. To halve the on-disk size:

```python
import joblib
joblib.dump(payload, "model.joblib", compress=3)
```

The `IntentClassifier.save` shortcut does not expose `compress`; serialise manually if size matters.

For an estimate of model size during a benchmark, `RunResult.model_size_bytes` is the uncompressed joblib pickle size.

## Training-cost tactics

### Parallel CV

`compare()` runs each baseline sequentially; within a baseline, sklearn's `cross_val_score` would parallelise — but `compare()` does its own loop. To parallelise across baselines, drive from a `concurrent.futures.ProcessPoolExecutor`:

```python
from concurrent.futures import ProcessPoolExecutor
from jurebes.benchmark import cross_validate

def _run(name):
    return cross_validate(name, X, y, k=5)

with ProcessPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(_run, ["logreg", "linear_svc", "nb_multinomial"]))
```

Inside the search subsystem, `search(..., n_jobs=-1)` parallelises candidate evaluations via the underlying sklearn search.

### HistGBM for tabular-style features

When working with mixed text + structured features (text plus a few dozen numeric columns), `HistGradientBoostingClassifier` is dramatically faster than vanilla gradient boosting and handles missing values natively. The `hist_gbm` baseline densifies via `FunctionTransformer` before the classifier.

### Choose `cv` thoughtfully

`StratifiedKFold(n_splits=5)` is a reasonable default but doubles training time vs 3-fold. For the search loop, 3-fold often suffices since the absolute score matters less than the relative ranking of candidates; reserve 5-fold for the final benchmark.

## When to give up

If after these tactics the latency target is still missed:

- Reduce intent inventory.
- Move to a lighter featurization (`hashing_word(2**16)` instead of `tfidf_word()`).
- Pre-compute exact-match short-circuits for the most common utterances. The OVOS pipeline plugin already does this.

---
- Back to [docs index](../index.md)
