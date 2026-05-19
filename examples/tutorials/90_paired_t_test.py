"""Paired Student's t-test on per-fold cross-validation scores.

Tests whether two classifiers differ significantly across the same CV
folds.
"""

# %%
from jurebes.benchmark.stats import paired_t_test_cv

scores_a = [0.84, 0.86, 0.82, 0.88, 0.85]
scores_b = [0.81, 0.80, 0.79, 0.83, 0.82]
res = paired_t_test_cv(scores_a, scores_b, alpha=0.05)
print(f"method={res.method}")
print(f"statistic={res.statistic:.3f} pvalue={res.pvalue:.4f}")
print(f"n={res.n} reject_null={res.reject_null}")
