"""Voting and stacking ensemble baselines.

Both ensemble three base learners (logreg, linear_svc, multinomial NB);
voting averages probabilities, stacking learns a meta-classifier.
"""

# %%
from jurebes import IntentClassifier
from jurebes.baselines import BASELINES

samples = {
    "greet": ["hello", "hi there", "hey friend", "good morning",
              "howdy", "salutations", "morning everyone", "good day"],
    "bye": ["goodbye", "see you later", "farewell", "catch you soon",
            "take care", "until next time", "so long", "ciao"],
}

for name in ("voting_soft", "stacking"):
    clf = IntentClassifier(BASELINES.build(name))
    for k, v in samples.items():
        clf.add_intent(k, v)
    clf.fit()
    r = clf.predict("hi friend")
    print(f"{name:12s} -> {r.intent} ({r.confidence:.3f})")
