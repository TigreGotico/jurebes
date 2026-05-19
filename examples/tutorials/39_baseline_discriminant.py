"""Discriminant analysis baselines: LDA and QDA.

LDA assumes shared covariance, QDA per-class covariance. Both need
dense input — handled inside the BASELINES factories.
"""

# %%
from jurebes import IntentClassifier
from jurebes.baselines import BASELINES

samples = {
    "greet": ["hello", "hi", "hey there", "good morning"],
    "bye": ["goodbye", "bye", "see you", "later"],
}
for name in ("lda_classifier", "qda_classifier"):
    clf = IntentClassifier(BASELINES.build(name))
    for k, v in samples.items():
        clf.add_intent(k, v)
    clf.fit()
    r = clf.predict("good morning")
    print(f"{name:16s} -> {r.intent} ({r.confidence:.3f})")
