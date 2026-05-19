"""Genetic-algorithm hyperparameter search via sklearn-genetic-opt.

Optional extra: install jurebes[search-genetic] to enable.
"""

# %%
try:
    from sklearn_genetic.space import Continuous
except ImportError:
    print("install jurebes[search-genetic]")
    raise SystemExit(0)

from jurebes.search import search

X = ["hello", "hi", "bye", "goodbye", "thanks", "thank you"] * 5
y = (["greet"] * 2 + ["bye"] * 2 + ["thanks"] * 2) * 5

space = {"clf__C": Continuous(0.01, 100.0, distribution="log-uniform")}
res = search("logreg", space, X, y, backend="genetic", cv=2, n_iter=6, seed=0)
print(f"best_score={res.best_score:.3f} best_params={res.best_params}")
