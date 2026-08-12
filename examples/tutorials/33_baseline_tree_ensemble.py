"""Tree and ensemble baselines.

Random forest, extra trees, gradient boosting, hist GBM, bagging.
"""

# %%
from jurebes import IntentClassifier
from jurebes.baselines import BASELINES

samples = {
    "greet": ["hello", "hi", "hey there", "good morning", "howdy"],
    "bye": ["goodbye", "bye", "see you", "later", "farewell"],
}
names = (
    "random_forest", "extra_trees", "gradient_boosting",
    "hist_gbm", "bagging_logreg",
)
for name in names:
    clf = IntentClassifier(BASELINES.build(name))
    for k, v in samples.items():
        clf.add_intent(k, v)
    clf.fit()
    r = clf.predict("see you later")
    print(f"{name:20s} -> {r.intent} ({r.confidence:.3f})")
