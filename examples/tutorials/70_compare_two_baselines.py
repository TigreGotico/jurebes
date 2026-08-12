"""Compare two baselines via cross-validation.

compare() runs cross_validate over each factory and returns a
ComparisonResult.
"""

# %%
from jurebes.benchmark import compare

X = (
    ["hello", "hi", "hey there", "good morning", "howdy"] * 4
    + ["goodbye", "bye", "see you", "later", "farewell"] * 4
    + ["thank you", "thanks", "much appreciated", "thanks a lot", "ta"] * 4
)
y = ["greet"] * 20 + ["bye"] * 20 + ["thanks"] * 20

result = compare(["logreg", "linear_svc"], X, y, k=3)
for r in result.rows:
    print(f"{r.name:12s} macro_f1={r.macro_f1:.3f} train={r.train_seconds:.3f}s")
