"""Pick the winner of a compare(), then tune it.

Run a small compare() to find the strongest baseline, then random-search
its canonical hyperparameter space and report the uplift.
"""

# %%
from jurebes.benchmark import compare
from jurebes.search import search, spaces

X = (
    ["hello", "hi", "hey there", "good morning"] * 5
    + ["goodbye", "bye", "see you", "later"] * 5
    + ["thanks", "thank you", "much appreciated", "ta"] * 5
)
y = ["greet"] * 20 + ["bye"] * 20 + ["thanks"] * 20

candidates = [n for n in ("logreg", "linear_svc", "nb_multinomial")
              if n in spaces.available()]
result = compare(candidates, X, y, k=2)
winner = max(result.rows, key=lambda r: r.macro_f1)
print(f"winner: {winner.name} baseline macro_f1={winner.macro_f1:.3f}")

space = spaces.for_baseline(winner.name)
tuned = search(winner.name, space, X, y, backend="random", cv=2, n_iter=5, seed=0)
print(f"tuned best_score={tuned.best_score:.3f} params={tuned.best_params}")
print(f"uplift={tuned.best_score - winner.macro_f1:+.3f}")
