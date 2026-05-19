"""Multi-metric scoring.

Pass several scoring metric names to compare(); each row's
extra_scores will populate accordingly.
"""

# %%
from jurebes.benchmark import compare

X = (
    ["hello", "hi", "hey there", "good morning"] * 5
    + ["goodbye", "bye", "see you", "later"] * 5
    + ["thanks", "thank you", "much appreciated", "ta"] * 5
)
y = ["greet"] * 20 + ["bye"] * 20 + ["thanks"] * 20

result = compare(
    ["logreg", "nb_multinomial"],
    X, y, k=3,
    scoring=("accuracy", "f1_macro", "log_loss"),
)
for r in result.rows:
    extras = {k: round(v, 3) for k, v in r.extra_scores.items()}
    print(f"{r.name:15s} {extras}")
