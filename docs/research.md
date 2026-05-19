# Research guide

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

Factories are called fresh per run so cross-validation gets clean estimators.

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

- `model_size_bytes` is the uncompressed joblib pickle size. Real on-disk size will be smaller when joblib's default zlib compression is enabled at save time.
- Per-fold `p50_ms`, `p95_ms`, `p99_ms` are computed *within* each fold and then averaged across folds. This is a per-fold percentile, not a pooled percentile across all predictions. Pooled percentiles (statistically more meaningful for tail latency) are planned in a follow-up sprint.

## Interpreting the report

| column | meaning |
| --- | --- |
| accuracy | mean across CV folds |
| macro_f1 | unweighted mean F1 per class — penalises poor-minority performance |
| micro_f1 | global F1 — equals accuracy in multi-class single-label setups |
| train_s | mean training seconds per fold |
| p50_ms / p95_ms | per-utterance inference latency percentiles |
| size_kb | serialised model size via `joblib.dump` to an in-memory buffer |

Use `to_json(comparison)` for downstream plotting / aggregation; every value is JSON-serialisable.

## Datasets

- `load_csv(path, text="text", label="intent")`
- `load_jsonl(path, text="text", label="intent")`
- `load_ovos_intents(directory)` — recurses for `.intent` / `.voc` / `.entity` files.
- `load_hf(name, split)` — optional `jurebes[hf]` extra; lazy-imports `datasets`.
