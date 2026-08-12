"""Customise a predefined search space.

spaces.for_baseline() returns the canonical space for a baseline as a
plain dict; mutate freely before passing it to search().
"""

# %%
from jurebes.search import search, spaces

space = spaces.for_baseline("logreg")
print(f"default space: {space}")

# Drop a hyperparameter and add a tighter range for another.
space.pop("feat__min_df", None)
space["clf__C"] = [0.5, 1.0, 2.0]
print(f"customised   : {space}")

X = ["hello", "hi", "bye", "goodbye", "thanks", "thank you"] * 4
y = (["greet"] * 2 + ["bye"] * 2 + ["thanks"] * 2) * 4
res = search("logreg", space, X, y, backend="grid", cv=2)
print(f"best={res.best_params} score={res.best_score:.3f}")
