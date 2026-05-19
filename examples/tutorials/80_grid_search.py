"""Grid search over a baseline's hyperparameter space."""

# %%
from jurebes.search import search

X = ["hello", "hi", "bye", "goodbye", "thanks", "thank you"] * 4
y = (["greet"] * 2 + ["bye"] * 2 + ["thanks"] * 2) * 4

space = {"clf__C": [0.5, 1.0, 2.0], "feat__ngram_range": [(1, 1), (1, 2)]}
res = search("logreg", space, X, y, backend="grid", cv=2)
print(f"best_score={res.best_score:.3f}  best_params={res.best_params}")
print(f"backend={res.backend} n_evaluations={res.n_evaluations}")
