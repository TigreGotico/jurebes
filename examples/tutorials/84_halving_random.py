"""Halving random search.

Sample candidates randomly then progressively prune via halving — a
good speed/quality compromise.
"""

# %%
from jurebes.search import search

X = ["hello", "hi", "bye", "goodbye", "thanks", "thank you"] * 6
y = (["greet"] * 2 + ["bye"] * 2 + ["thanks"] * 2) * 6

space = {"clf__C": [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]}
res = search("logreg", space, X, y, backend="halving_random", cv=2, n_iter=6, seed=0)
print(f"best_score={res.best_score:.3f} best_params={res.best_params}")
