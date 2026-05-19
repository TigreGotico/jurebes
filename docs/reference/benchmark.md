# `jurebes.benchmark`

Module: `jurebes.benchmark`. Re-exports `compare`, `cross_validate`, `train_test`, `RunResult`, `ComparisonResult`, `to_markdown`, `to_json`, `pooled_percentiles`, and `SCORING`.

## Public functions

### `train_test(classifier_factory, X, y, *, test_size=0.2, seed=0, scoring=("accuracy", "f1_macro"))`

Single train/test holdout evaluation. `classifier_factory` is a baseline name (`str`) or a zero-arg factory callable returning a fresh sklearn pipeline.

Returns a `RunResult`. The `fold_scores` field is empty (CV-only).

### `cross_validate(classifier_factory, X, y, *, k=5, seed=0, scoring=("accuracy", "f1_macro"))`

Stratified k-fold CV. Returns a `RunResult` whose scalar fields are means over the folds and whose `fold_scores` holds per-fold raw scores per scoring metric (plus `f1_macro` and `f1_micro` automatically).

### `compare(baselines, X, y, *, k=5, seed=0, scoring=("accuracy", "f1_macro"))`

Run `cross_validate` for each baseline. Returns a `ComparisonResult`.

```python
from jurebes.benchmark import compare, to_markdown
result = compare(["logreg", "linear_svc", "nb_complement"], X, y, k=5)
print(to_markdown(result))
```

### `to_markdown(result, *, sort_by=None, precision=4, with_significance=False)`

Render a `ComparisonResult` as a Markdown table. `sort_by` is a column name; `with_significance` appends a Friedman+Nemenyi block (≥3 baselines) or paired-t + Wilcoxon block (=2 baselines).

### `to_json(result)`

Return a JSON string of the comparison. Every value is JSON-serialisable.

### `pooled_percentiles(samples)`

Returns `{"p50", "p95", "p99", "mean"}` over a flat list of per-prediction latencies (milliseconds).

## Dataclasses

### `RunResult`

```python
@dataclass
class RunResult:
    name: str
    accuracy: float
    macro_f1: float
    micro_f1: float
    per_class_f1: Dict[str, float]
    train_seconds: float
    predict_ms_p50: float                  # per-fold mean of within-fold p50
    predict_ms_p95: float                  # per-fold mean of within-fold p95
    predict_ms_p99: float                  # per-fold mean of within-fold p99
    predict_ms_p50_pooled: float           # pooled p50 across all predictions
    predict_ms_p95_pooled: float           # pooled p95
    predict_ms_p99_pooled: float           # pooled p99
    predict_ms_mean: float                 # pooled mean
    model_size_bytes: int                  # uncompressed joblib pickle size
    confusion_matrix: List[List[int]]
    labels: List[str]
    extra_scores: Dict[str, float]         # mean per metric
    group: str                             # group memberships (comma-separated)
    fold_scores: Dict[str, List[float]]    # per-fold raw scores
```

### `ComparisonResult`

```python
@dataclass
class ComparisonResult:
    rows: List[RunResult]
    scoring: Tuple[str, ...]

    @property
    def fold_scores_by_baseline(self) -> dict:
        """{baseline_name → {metric → [per-fold scores]}}"""
```

`compare()` populates `rows` and `scoring`; `fold_scores_by_baseline` is derived.

## The `SCORING` registry

`from jurebes.benchmark import SCORING` is an alias for `jurebes.benchmark.scoring.SCORERS`. Built-in keys:

| key | meaning |
| --- | --- |
| `f1_macro` | macro-averaged F1 |
| `f1_micro` | micro-averaged F1 (= accuracy in single-label multi-class) |
| `accuracy` | classification accuracy |
| `balanced_accuracy` | recall macro-averaged over classes |
| `log_loss` | cross-entropy; needs `predict_proba` |
| `top_k_accuracy` | top-3 accuracy by default; needs `predict_proba` |

Register custom scorers: see [../advanced/custom-scoring-metrics.md](../advanced/custom-scoring-metrics.md).

## Caveats

- `predict_ms_p50`, `predict_ms_p95`, `predict_ms_p99` are per-fold averages — useful for variance signalling. For tail-latency reporting use the pooled fields.
- `model_size_bytes` is the *uncompressed* joblib pickle size. On-disk size with `compress=3` is typically half.
- `extra_scores` reports the mean across folds for each requested scoring metric; the raw per-fold values are in `fold_scores`.

## Example — full workflow

```python
from jurebes.benchmark import compare, to_markdown
from jurebes.datasets import load_csv

X, y = load_csv("data.csv")

result = compare(
    ["logreg", "linear_svc", "nb_complement", "linear_svc_char"],
    X, y, k=5,
    scoring=("accuracy", "f1_macro", "log_loss"),
)

print(to_markdown(result, sort_by="macro_f1", with_significance=True))
```

---
- Back to [reference index](index.md)
