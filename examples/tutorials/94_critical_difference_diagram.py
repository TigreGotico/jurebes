"""Critical-difference diagram (Demsar 2006).

Visualises which classifiers are statistically indistinguishable
according to the Nemenyi post-hoc test.
"""

# %%
from jurebes.benchmark.stats import critical_difference

fold_scores = {
    "logreg":     [0.84, 0.86, 0.82, 0.88, 0.85],
    "linear_svc": [0.83, 0.85, 0.80, 0.87, 0.84],
    "nb":         [0.78, 0.79, 0.76, 0.80, 0.78],
    "rf":         [0.80, 0.82, 0.78, 0.83, 0.81],
}
cd = critical_difference(fold_scores, alpha=0.05)
print(cd.to_ascii())
