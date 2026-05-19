"""Voting and stacking ensemble baselines.

Both ensemble three base learners (logreg, linear_svc, multinomial NB);
voting averages probabilities, stacking learns a meta-classifier.
"""

# %%
from jurebes import IntentClassifier
from jurebes.baselines import BASELINES

samples = {
    "greet": ["hello", "hi", "hey there", "good morning"],
    "bye": ["goodbye", "bye", "see you", "later"],
}

for name in ("voting_soft", "stacking"):
    clf = IntentClassifier(BASELINES.build(name))
    for k, v in samples.items():
        clf.add_intent(k, v)
    clf.fit()
    r = clf.predict("hi friend")
    print(f"{name:12s} -> {r.intent} ({r.confidence:.3f})")
