"""Reduced-dimensionality baselines.

LSA, NMF and autoencoder bottlenecks feeding into LogisticRegression.
"""

# %%
from jurebes import IntentClassifier
from jurebes.baselines import BASELINES

samples = {
    "greet": ["hello", "hi", "hey there", "good morning", "howdy"],
    "bye": ["goodbye", "bye", "see you", "later", "farewell"],
}
for name in ("lsa_logreg", "nmf_logreg", "autoencoder_logreg"):
    clf = IntentClassifier(BASELINES.build(name))
    for k, v in samples.items():
        clf.add_intent(k, v)
    clf.fit()
    r = clf.predict("see you")
    print(f"{name:22s} -> {r.intent} ({r.confidence:.3f})")
