"""McNemar's test on paired predictions.

Compares two classifiers' predictions on the same held-out set — tests
disagreement asymmetry.
"""

# %%
from jurebes.benchmark.stats import mcnemar_test

y_true = ["a", "b", "a", "b", "a", "b", "a", "b"]
preds_a = ["a", "b", "a", "a", "a", "b", "a", "b"]
preds_b = ["a", "a", "b", "b", "a", "b", "a", "a"]
res = mcnemar_test(preds_a, preds_b, y_true, alpha=0.05)
print(f"{res.method}: statistic={res.statistic:.3f} p={res.pvalue:.4f}")
print(f"reject_null={res.reject_null} n={res.n}")
