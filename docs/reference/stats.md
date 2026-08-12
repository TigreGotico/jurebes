# `jurebes.benchmark.stats`

Statistical comparison tests on benchmark results. See [../theory/statistical-comparison.md](../theory/statistical-comparison.md) for the theoretical framing.

## Dataclasses

### `TestResult`

```python
@dataclass
class TestResult:
    statistic: float
    pvalue: float
    reject_null: bool
    method: str        # "paired_t" | "wilcoxon" | "mcnemar"
    n: int
```

### `FriedmanResult`

```python
@dataclass
class FriedmanResult:
    statistic: float
    pvalue: float
    reject_null: bool
    mean_ranks: Dict[str, float]
    pairwise: Dict[Tuple[str, str], float]   # approximate two-sided Nemenyi p-values
```

### `CDDiagram`

```python
@dataclass
class CDDiagram:
    mean_ranks: Dict[str, float]
    cd_threshold: float
    groups: List[Set[str]]                  # maximal cliques of indistinguishable baselines
    n_datasets: int

    def to_ascii(self) -> str: ...
    def to_matplotlib(self, ax=None): ...   # requires jurebes[bench-plot]
```

## Functions

### `paired_t_test_cv(scores_a, scores_b, alpha=0.05) -> TestResult`

Paired Student's t-test on per-fold CV scores. Both arrays must have the same length. Returns a `TestResult` with `method="paired_t"`.

```python
from jurebes.benchmark import cross_validate
from jurebes.benchmark.stats import paired_t_test_cv

a = cross_validate("logreg",    X, y, k=5, seed=0)
b = cross_validate("linear_svc", X, y, k=5, seed=0)
r = paired_t_test_cv(a.fold_scores["f1_macro"], b.fold_scores["f1_macro"])
print(r.statistic, r.pvalue, r.reject_null)
```

### `wilcoxon_signed_rank_cv(scores_a, scores_b, alpha=0.05) -> TestResult`

Non-parametric paired test on per-fold CV scores. Drop-in for paired-t when normality is suspect.

### `mcnemar_test(preds_a, preds_b, y_true, alpha=0.05) -> TestResult`

McNemar's test on per-sample correctness with mid-p continuity correction. Operates on a single held-out test set, not on fold scores.

```python
from jurebes.benchmark.stats import mcnemar_test
preds_a = [clf_a.predict(x).intent for x in X_test]
preds_b = [clf_b.predict(x).intent for x in X_test]
r = mcnemar_test(preds_a, preds_b, y_test)
```

### `friedman_nemenyi(fold_scores, alpha=0.05) -> FriedmanResult`

Friedman omnibus test followed by post-hoc Nemenyi pairwise comparison. `fold_scores` is `dict[name → list[per-fold scores]]`. Requires at least 2 baselines; supports up to 20 (Nemenyi q-table limit).

`alpha` must be one of `{0.05, 0.10}` (q-table values).

```python
from jurebes.benchmark.stats import friedman_nemenyi
fr = friedman_nemenyi({
    "logreg":     result.fold_scores_by_baseline["logreg"]["f1_macro"],
    "linear_svc": result.fold_scores_by_baseline["linear_svc"]["f1_macro"],
    "rbf_svc":    result.fold_scores_by_baseline["rbf_svc"]["f1_macro"],
})
print(fr.mean_ranks, fr.reject_null)
```

### `critical_difference(fold_scores, alpha=0.05) -> CDDiagram`

Builds a critical-difference diagram. `fold_scores` and `alpha` have the same constraints as `friedman_nemenyi`.

```python
from jurebes.benchmark.stats import critical_difference
cd = critical_difference(fold_scores_by_baseline_for_metric)
print(cd.to_ascii())

# Plotting requires jurebes[bench-plot]
ax = cd.to_matplotlib()
```

## Significance integration in reports

`to_markdown(result, with_significance=True)` (in `jurebes.benchmark.report`) appends:

- A **Critical Difference** subsection when comparing ≥3 baselines (Friedman + Nemenyi on the chosen metric, default `f1_macro`, rendered as ASCII).
- A **Paired-t** + **Wilcoxon** block when comparing exactly 2 baselines.

CLI counterparts:

```bash
jurebes benchmark --dataset data.csv --baselines @linear --with-significance
jurebes stats --runs run_a.json run_b.json --metric f1_macro
jurebes stats --pair run_a.json run_b.json --metric f1_macro
```

## Caveats

- Tests assume per-fold scores are exchangeable under $H_0$. Severe class imbalance or fold leakage violates the assumption.
- p-values do not measure effect size — a "significant" 0.001 difference in macro-F1 can be operationally meaningless.
- Nemenyi's q-table covers up to k=20 baselines and alpha in {0.05, 0.10}. For other configurations, fall back to pairwise paired-t with Bonferroni correction.

---
- Back to [reference index](index.md)
