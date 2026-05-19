"""Wilcoxon signed-rank test on per-fold CV scores.

Non-parametric alternative to the paired t-test — does not assume the
score differences are normally distributed.
"""

# %%
from jurebes.benchmark.stats import wilcoxon_signed_rank_cv

scores_a = [0.84, 0.86, 0.82, 0.88, 0.85, 0.87]
scores_b = [0.81, 0.80, 0.79, 0.83, 0.82, 0.80]
res = wilcoxon_signed_rank_cv(scores_a, scores_b, alpha=0.05)
print(f"{res.method}: statistic={res.statistic:.3f} p={res.pvalue:.4f}")
print(f"reject_null={res.reject_null}")
