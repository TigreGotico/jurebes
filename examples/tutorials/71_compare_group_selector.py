"""Compare an entire baseline group.

Use BASELINES.resolve('@linear') to expand a group selector into the
list of factories compare() consumes.
"""

# %%
from jurebes.baselines import BASELINES
from jurebes.benchmark import compare

X = (
    ["hello", "hi", "hey there", "good morning"] * 4
    + ["goodbye", "bye", "see you", "later"] * 4
    + ["thanks", "thank you", "much appreciated", "ta"] * 4
)
y = ["greet"] * 16 + ["bye"] * 16 + ["thanks"] * 16

names = BASELINES.resolve("@linear")[:4]  # subset to keep the demo quick
print(f"comparing: {names}")
result = compare(names, X, y, k=3)
for r in result.rows:
    print(f"{r.name:22s} macro_f1={r.macro_f1:.3f}")
