# Cross-validation

A single train/test split estimates model quality with high variance — the result depends on which samples ended up where. Cross-validation averages over multiple splits to reduce that variance.

## k-fold

Partition the dataset into $k$ disjoint folds. For each fold $i \in \{1, \ldots, k\}$:

1. Train on the union of the other $k - 1$ folds.
2. Evaluate on fold $i$.

Average the $k$ scores. Each sample is held out exactly once. Standard $k$ is 5 or 10.

The variance of the $k$-fold estimator decreases with $k$ but the folds become smaller and noisier; 5 is a reasonable default for intent corpora.

## Stratified k-fold

Standard k-fold draws folds uniformly at random. With imbalanced classes, a fold may end up with zero samples of a minority class, breaking per-class metrics.

*Stratified* k-fold preserves class proportions in every fold. `sklearn.model_selection.StratifiedKFold` is the implementation; `jurebes.benchmark.cross_validate` uses it by default. There is no good reason not to stratify for multi-class classification.

## Leave-one-out (LOOCV)

The extreme case $k = N$: every fold holds out one sample. Low bias, high variance, and expensive — $N$ model fits.

Useful when $N$ is tiny (a few dozen samples) and any other choice of $k$ would yield uselessly small folds. Rarely the right choice for intent corpora.

## Holdout vs CV

A single 80/20 holdout is simpler and faster but provides only one score with no error bar. Use:

- **Holdout** for fast iteration during model development.
- **5-fold CV** for the final benchmark and statistical comparisons.

`jurebes.benchmark.train_test` does the holdout; `jurebes.benchmark.cross_validate` and `compare` do k-fold.

## Hyperparameter selection: nested CV

If hyperparameters are chosen on the same folds used for final evaluation, the reported score is optimistically biased. Proper protocol:

1. **Outer loop** (e.g. 5-fold) for evaluation.
2. **Inner loop** (e.g. 3-fold within each outer-fold's training data) for hyperparameter search.
3. Each outer-fold trains on its train data using the best inner-loop hyperparameters; outer-fold score is the unbiased estimate.

This is expensive ($k_\text{outer} \cdot k_\text{inner}$ fits per candidate). A faster and acceptable shortcut: tune on a single inner CV; report the chosen-baseline-on-outer-CV score *plus* an honest acknowledgement of the small optimistic bias.

jurebes' `search()` runs an inner CV. To compose nested CV, drive `search()` from inside a manual outer loop.

## Variance and the paired tests

When two baselines are compared on the *same* CV folds (same train/test partitioning), the per-fold scores are paired. Paired tests (paired t-test, Wilcoxon signed-rank) exploit the pairing to detect smaller effects than unpaired tests would.

`jurebes.benchmark.compare(..., seed=0)` uses the same seed for every baseline, ensuring identical folds across baselines and enabling the paired tests in [statistical-comparison.md](statistical-comparison.md).

## Fold-score variance

`RunResult.fold_scores` exposes the per-fold scores for every scoring metric:

```python
from jurebes.benchmark import cross_validate
import numpy as np

r = cross_validate("logreg", X, y, k=5, scoring=("f1_macro",))
scores = r.fold_scores["f1_macro"]
print(f"{np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

A large std relative to the mean indicates the dataset is small enough that fold composition matters a lot — bigger $k$ or more data needed for stable estimates.

---
- Back to [docs index](../index.md)
