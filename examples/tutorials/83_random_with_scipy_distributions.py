"""Random search with scipy.stats distributions.

scipy.stats.loguniform and randint give continuous / integer sampling
rather than enumerated lists.
"""

# %%
from scipy.stats import loguniform, randint

from jurebes.search import search

X = ["hello", "hi", "bye", "goodbye", "thanks", "thank you"] * 4
y = (["greet"] * 2 + ["bye"] * 2 + ["thanks"] * 2) * 4

space = {
    "clf__C": loguniform(1e-2, 1e2),
    "feat__min_df": randint(1, 4),
}
res = search("logreg", space, X, y, backend="random", cv=2, n_iter=6, seed=0)
print(f"best_score={res.best_score:.3f}")
print(f"best_params={res.best_params}")
