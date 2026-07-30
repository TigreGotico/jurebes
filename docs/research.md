# Research guide

## Adding a featurizer

Featurizers are zero-arg builders that return a fresh sklearn transformer or `Pipeline`. Add them to `jurebes/featurizers.py`:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_word_bigrams():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
```

Featurizers compose freely with the existing ones. `feature_union(builder_a, builder_b, ...)` returns a `FeatureUnion` over their outputs.

## Adding a baseline

```python
from jurebes import BASELINES
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASELINES.register(
    "my_logreg",
    lambda: Pipeline([
        ("feat", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(C=4.0, max_iter=2000)),
    ]),
)
```

Jurebes calls each factory fresh per run, so cross-validation gets clean estimators.

The bundled registry uses a dict literal at `jurebes/baselines.py::BASELINE_SPECS`. Add new entries there directly, then tag them into a group through `_GROUPS` so the CLI's `@<group>` selector works.

## Adding a search backend

The search subsystem (`jurebes/search/`) dispatches on a `backend` string to a per-backend `run()` function. To add a backend, write `jurebes/search/<name>.py` that exposes a `run(*, factory, param_space, X, y, backend, scoring, cv, n_iter, seed, n_jobs, verbose) -> dict` function, then add it to the `_BACKENDS` tuple and the dispatch ladder in `search/api.py`. Lazy-import optional dependencies with a clear `ImportError("install jurebes[<extra>] to use the <name> backend")`.

For non-probabilistic estimators (`LinearSVC`, `RidgeClassifier`, `Perceptron`, `PassiveAggressiveClassifier`, `SGDClassifier(loss="hinge")`), wrap with `sklearn.calibration.CalibratedClassifierCV(cv=3)` so `predict_proba` works.

## Running a benchmark

```python
from jurebes.benchmark import compare, to_markdown
from jurebes.datasets import load_csv

X, y = load_csv("data.csv")
report = compare(["logreg", "linear_svc", "nb_multinomial", "my_logreg"], X, y, k=5)
print(to_markdown(report))
```

Or from the CLI:

```bash
jurebes benchmark --dataset data.csv --baselines logreg,linear_svc --cv 5 --out report.md
jurebes benchmark --dataset data.csv --baselines logreg --format json --out report.json
```

## Reported metric caveats

- `model_size_bytes` is the uncompressed joblib pickle size. The real on-disk size is smaller when joblib's default zlib compression is enabled at save time.
- Per-fold `p50_ms`, `p95_ms`, and `p99_ms` are computed within each fold, then averaged across folds. This is a per-fold percentile, not a pooled percentile across all predictions. The pooled fields `p50_ms_pooled`, `p95_ms_pooled`, `p99_ms_pooled`, and `mean_ms` are computed over the flat list of every per-prediction latency observed across all folds. Prefer them for tail-latency reporting.

## Cookbook

### 1. Compare all linear models

```python
from jurebes.benchmark import compare, to_markdown
from jurebes.datasets import load_csv
from jurebes.baselines import BASELINES

X, y = load_csv("data.csv")
result = compare(BASELINES.resolve("@linear"), X, y, k=5,
                 scoring=("accuracy", "f1_macro"))
print(to_markdown(result, sort_by="macro_f1"))
```

### 2. Tune logreg with random search

```python
from jurebes.search import search, spaces
from jurebes.datasets import load_csv

X, y = load_csv("data.csv")
r = search("logreg", spaces.for_baseline("logreg"), X, y,
           backend="random", n_iter=40, cv=5)
print("best params:", r.best_params)
r.best_estimator.save("best_logreg.joblib")
```

### 3. Find the best reduced-dim representation

```python
from jurebes.benchmark import compare, to_markdown
from jurebes.baselines import BASELINES
from jurebes.datasets import load_csv

X, y = load_csv("data.csv")
result = compare(BASELINES.resolve("@reduced_dim"), X, y, k=5)
print(to_markdown(result, sort_by="macro_f1"))
```

### 4. Find the best NB variant for short utterances

```python
from jurebes.benchmark import compare, to_markdown
from jurebes.baselines import BASELINES
from jurebes.datasets import load_csv

X, y = load_csv("short_utts.csv")
result = compare(BASELINES.resolve("@naive_bayes"), X, y, k=5,
                 scoring=("accuracy", "f1_macro", "log_loss"))
print(to_markdown(result, sort_by="log_loss"))
```

### 5. Benchmark CPU latency

```python
from jurebes.benchmark import compare, to_markdown
from jurebes.datasets import load_csv

X, y = load_csv("data.csv")
result = compare(
    ["logreg", "linear_svc", "nb_multinomial", "hashing_sgd_log"],
    X, y, k=5,
)
print(to_markdown(result, sort_by="p95_ms", precision=3))
```

## Interpreting the report

| column | meaning |
| --- | --- |
| accuracy | mean across CV folds |
| macro_f1 | unweighted mean F1 per class; penalizes poor minority-class performance |
| micro_f1 | global F1; equals accuracy in multi-class single-label setups |
| train_s | mean training seconds per fold |
| p50_ms / p95_ms | per-fold per-utterance inference latency percentiles, averaged across folds |
| predict_ms_p95_pooled | pooled p95 over every per-prediction latency across all folds |
| predict_ms_p99_pooled | pooled p99 over every per-prediction latency across all folds |
| size_kb | serialized model size via `joblib.dump` to an in-memory buffer |
| extra_scores | dict of any extra scoring metrics requested through `scoring=` |
| group | the `BASELINES` group tag this baseline belongs to (or `None` for ad-hoc estimators) |

Use `to_json(comparison)` for downstream plotting or aggregation. Every value is JSON-serializable.

### Performance notes

The numbers below are illustrative orders of magnitude, not measured benchmarks. Actual values depend on dataset size, vocabulary, hardware, and joblib compression settings.

| baseline family | train (relative) | p95 latency (relative) | model size (relative) |
| --- | --- | --- | --- |
| linear (logreg / linear_svc) | fast | low | small |
| naive bayes | fastest | low | small |
| hashing + SGD | fastest | low | tiny |
| reduced_dim (LSA/NMF) | medium | low | small |
| tree ensembles | medium-slow | medium | medium-large |
| RBF SVM | slow | medium-high | medium |
| MLP shallow | slow | low | medium |
| stacking / voting | slowest | highest | largest |

## Datasets

- `load_csv(path, text="text", label="intent")`
- `load_jsonl(path, text="text", label="intent")`
- `load_ovos_intents(directory)`: recurses for `.intent` / `.voc` / `.entity` files.
- `load_hf(name, split)`: optional `jurebes[hf]` extra; lazy-imports `datasets`.

## Canonical datasets

`jurebes.datasets.canonical` provides six canonical intent-benchmark loaders,
also reachable from the CLI as `--dataset @<name>`:

- `load_snips`: SNIPS NLU benchmark (HF `benayas/snips`, 7 intents).
- `load_clinc`: CLINC150 OOS (HF `clinc_oos`, config `plus`, 151 intents incl. `oos`).
- `load_banking77`: fine-grained banking intents (HF `banking77`, 77 intents).
- `load_hwu64`: home-assistant intents (HF `DeepPavlov/hwu64`, 64 intents).
- `load_atis`: Air Travel Information System (HF `tuetschek/atis`).
- `load_massive`: multilingual SLU benchmark (HF `AmazonScience/massive`, 60 intents x 51 locales).

Each loader returns `(X, y)` and raises `ImportError("install jurebes[hf] ...")`
if the `datasets` extra is missing. HF handles caching.

## Statistical comparison

`jurebes.benchmark.stats` provides the tests recommended by Demšar (2006),
["Statistical Comparisons of Classifiers over Multiple Data Sets"](https://www.jmlr.org/papers/v7/demsar06a.html) (JMLR 7):

- `paired_t_test_cv(a, b)`: a paired Student's t-test on per-fold CV scores.
  It assumes scores are roughly normal. Use it to compare two models on a
  single dataset with k-fold CV.
- `wilcoxon_signed_rank_cv(a, b)`: a non-parametric drop-in for paired-t
  when normality is suspect or k is small.
- `mcnemar_test(preds_a, preds_b, y_true)`: a paired test on per-sample
  correctness (contingency table), with mid-p continuity correction. Use it
  on a held-out test set with two models.
- `friedman_nemenyi(fold_scores)`: a Friedman omnibus test across three or
  more baselines, followed by a post-hoc Nemenyi pairwise comparison. Use it
  when comparing many baselines across multiple datasets or folds.
- `critical_difference(fold_scores)`: average ranks plus the
  critical-difference threshold. It renders as an ASCII table by default,
  with `to_matplotlib(ax)` available when `jurebes[bench-plot]` is installed.

The `to_markdown(result, with_significance=True)` report adds either a
Critical Difference subsection (three or more baselines) or paired-t and
Wilcoxon rows (exactly two baselines) for the chosen metric (default
`f1_macro`). The CLI exposes the same feature through
`jurebes benchmark --with-significance` and the `jurebes stats` subcommand
for offline analysis of saved runs.

---
[Home](index.md)
