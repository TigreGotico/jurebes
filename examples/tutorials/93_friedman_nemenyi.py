"""Friedman test with Nemenyi post-hoc.

For comparing more than two classifiers across multiple folds /
datasets. Returns mean ranks and pairwise p-values.
"""

# %%
from jurebes.benchmark.stats import friedman_nemenyi

fold_scores = {
    "logreg":     [0.84, 0.86, 0.82, 0.88, 0.85],
    "linear_svc": [0.83, 0.85, 0.80, 0.87, 0.84],
    "nb":         [0.78, 0.79, 0.76, 0.80, 0.78],
}
res = friedman_nemenyi(fold_scores, alpha=0.05)
print(f"Friedman stat={res.statistic:.3f} p={res.pvalue:.4f} reject_null={res.reject_null}")
print(f"mean ranks: {res.mean_ranks}")
print(f"pairwise p-values: {res.pairwise}")
