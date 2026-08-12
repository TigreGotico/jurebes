"""Bayesian hyperparameter search via skopt.

Optional extra: install jurebes[search-bayes] to enable.
"""

# %%
try:
    from skopt.space import Real
except ImportError:
    print("install jurebes[search-bayes]")
    raise SystemExit(0)

from jurebes.search import search

X = ["hello", "hi", "bye", "goodbye", "thanks", "thank you"] * 5
y = (["greet"] * 2 + ["bye"] * 2 + ["thanks"] * 2) * 5

space = {"clf__C": Real(1e-2, 1e2, prior="log-uniform")}
res = search("logreg", space, X, y, backend="bayes", cv=2, n_iter=6, seed=0)
print(f"best_score={res.best_score:.3f} best_params={res.best_params}")
