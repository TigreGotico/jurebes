"""Find the best naive Bayes variant + paired t-test.

Compare the three NB baselines via cross-validation, then run a paired
t-test on the per-fold macro_f1 scores of the top two.
"""

# %%
from jurebes.benchmark import compare
from jurebes.benchmark.stats import paired_t_test_cv

X = (
    ["hello", "hi", "hey there", "good morning"] * 5
    + ["goodbye", "bye", "see you", "later"] * 5
    + ["thanks", "thank you", "much appreciated", "ta"] * 5
)
y = ["greet"] * 20 + ["bye"] * 20 + ["thanks"] * 20

result = compare(["nb_multinomial", "nb_complement", "nb_bernoulli"], X, y, k=3)
ranked = sorted(result.rows, key=lambda r: r.macro_f1, reverse=True)
for r in ranked:
    print(f"{r.name:15s} macro_f1={r.macro_f1:.3f}")

top, second = ranked[0], ranked[1]
a = top.fold_scores.get("f1_macro", [])
b = second.fold_scores.get("f1_macro", [])
test = paired_t_test_cv(a, b, alpha=0.05)
print(f"{top.name} vs {second.name}: p={test.pvalue:.4f} reject_null={test.reject_null}")
