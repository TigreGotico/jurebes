"""Stability of cross-validation across fold counts.

k=3 vs k=5 on the same data — variance in macro_f1 across folds is a
useful stability signal.
"""

# %%
from jurebes.benchmark import cross_validate

X = (
    ["hello", "hi", "hey there", "good morning"] * 5
    + ["goodbye", "bye", "see you", "later"] * 5
    + ["thanks", "thank you", "much appreciated", "ta"] * 5
)
y = ["greet"] * 20 + ["bye"] * 20 + ["thanks"] * 20

for k in (3, 5):
    r = cross_validate("logreg", X, y, k=k)
    folds = r.fold_scores.get("f1_macro", [])
    print(f"k={k}: macro_f1={r.macro_f1:.3f} per-fold={[round(v, 3) for v in folds]}")
