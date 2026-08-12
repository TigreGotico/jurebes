"""Halving grid search.

Successive halving prunes poor candidates early — faster than full grid
when the space is large.
"""

# %%
from jurebes.search import search

X = ["hello", "hi", "bye", "goodbye", "thanks", "thank you"] * 6
y = (["greet"] * 2 + ["bye"] * 2 + ["thanks"] * 2) * 6

space = {"clf__C": [0.1, 0.5, 1.0, 2.0, 4.0]}
res = search("logreg", space, X, y, backend="halving_grid", cv=2)
print(f"best_score={res.best_score:.3f} best_params={res.best_params}")
