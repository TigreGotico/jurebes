# Statistical comparison of classifiers

Reporting "model A scored 0.87 macro-F1; model B scored 0.86" without uncertainty obscures whether the difference is meaningful. The standard remedies are paired statistical tests on per-fold scores, plus the Friedman + Nemenyi protocol when comparing more than two baselines. See Demšar (2006), "Statistical Comparisons of Classifiers over Multiple Data Sets" (JMLR 7).

## Null-hypothesis framing

For each test:

- **Null hypothesis $H_0$.** The two (or more) models have the same expected performance.
- **Alternative $H_1$.** They differ.
- **Test statistic.** A function of the observed scores.
- **p-value.** The probability of seeing a statistic at least as extreme under $H_0$.
- **Decision.** Reject $H_0$ when $p < \alpha$ (typically $\alpha = 0.05$).

## Paired t-test

For two baselines evaluated on the same $k$ CV folds, the per-fold score differences $d_i = a_i - b_i$ should have zero mean under $H_0$:

$$t = \frac{\bar{d}}{s_d / \sqrt{k}}, \quad H_0 : \mathbb{E}[d] = 0$$

The statistic is t-distributed with $k - 1$ degrees of freedom *under the assumption that $d$ is normally distributed*.

`jurebes.benchmark.stats.paired_t_test_cv(a, b, alpha=0.05)` returns a `TestResult`.

Caveats:

- Assumes $d_i$ are roughly normal. With $k = 5$ folds, normality is hard to verify; the test is robust to mild deviations.
- The folds are not truly independent (samples overlap across training sets); the test slightly over-estimates significance.

## Wilcoxon signed-rank

A non-parametric drop-in for paired t. Operates on the ranks of $|d_i|$, signed by the sign of $d_i$. Robust when normality is suspect or $k$ is small.

`jurebes.benchmark.stats.wilcoxon_signed_rank_cv(a, b, alpha=0.05)` returns a `TestResult`.

Preferred over the paired t-test when:

- $k$ is small.
- The score distributions are heavy-tailed.
- You want a single conservative test without assumption-checking.

## McNemar's test

Operates on per-*sample* correctness from a single held-out test set, not per-fold scores. Build the $2 \times 2$ contingency table:

|  | B correct | B wrong |
| --- | --- | --- |
| A correct | $a$ | $c$ |
| A wrong | $b$ | $d$ |

Under $H_0$ that A and B err equivalently, $b$ and $c$ should be equal. The test statistic is:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

(with continuity correction). `jurebes.benchmark.stats.mcnemar_test(preds_a, preds_b, y_true)` uses the more accurate mid-p exact binomial. Returns a `TestResult`.

Use when you have a single held-out test set and want sample-level rather than fold-level pairing.

## Friedman + Nemenyi

For comparing $k \geq 3$ baselines across $n$ folds (or datasets):

1. Rank the baselines within each fold (1 = best).
2. Compute mean ranks $\bar{R}_j$ per baseline.
3. The Friedman statistic under $H_0$ (equal mean ranks) is:

$$\chi^2_F = \frac{12n}{k(k+1)} \left[ \sum_j \bar{R}_j^2 - \frac{k(k+1)^2}{4} \right]$$

4. If Friedman rejects, run the **Nemenyi** post-hoc: pairs $(j, j')$ differ significantly when

$$|\bar{R}_j - \bar{R}_{j'}| > \mathrm{CD}, \quad \mathrm{CD} = q_\alpha \sqrt{\frac{k(k+1)}{6n}}$$

where $q_\alpha$ is the studentised-range critical value (tabulated by Demšar 2006).

`jurebes.benchmark.stats.friedman_nemenyi(fold_scores, alpha=0.05)` returns a `FriedmanResult` with Friedman statistic, p-value, mean ranks, and pairwise approximate p-values.

## Critical-difference diagrams

A visual summary of the Friedman + Nemenyi outcome: plot mean ranks on a horizontal axis, mark the critical difference CD, and group statistically-indistinguishable baselines into cliques.

`jurebes.benchmark.stats.critical_difference(fold_scores, alpha=0.05)` returns a `CDDiagram` with `to_ascii()` and `to_matplotlib(ax=None)` renderers. The matplotlib renderer requires `jurebes[bench-plot]`.

## Reading a CD diagram

```
Critical Difference = 0.846  (n=5)

rank  baseline
1.200  logreg
1.800  linear_svc
4.100  random_forest

statistically indistinguishable groups:
  {logreg, linear_svc}
  {random_forest}
```

`logreg` and `linear_svc` are within CD of each other; `random_forest` is not. Reporting "logreg beats random_forest" is justified; reporting "logreg beats linear_svc" is not.

## Integration with benchmark reports

`to_markdown(result, with_significance=True)` automatically appends:

- A **Critical Difference** section when comparing $\geq 3$ baselines on the chosen metric (default `f1_macro`).
- A paired-t + Wilcoxon block when comparing exactly 2 baselines.

The CLI exposes the same via `jurebes benchmark --with-significance` and offline analysis of saved runs via `jurebes stats --runs run_a.json run_b.json --metric f1_macro`.

## Caveats

- All these tests assume per-fold scores are exchangeable under $H_0$. Severe class imbalance, fold leakage, or non-iid sampling violates the assumption.
- p-values do not measure effect size. A 0.001 difference in macro-F1 can be "significant" with enough folds and still be operationally meaningless.
- Multiple comparisons inflate the false-discovery rate. Nemenyi already handles this for $k$ baselines, but if you also run paired t-tests on every pair, apply Bonferroni or Benjamini-Hochberg correction.

---
- See [reference/stats.md](../reference/stats.md) for the full function-level API.
- Back to [docs index](../index.md)
