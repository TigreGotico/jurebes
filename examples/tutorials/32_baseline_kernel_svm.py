"""Kernel SVM baselines: rbf_svc, nusvc.

Non-linear SVMs on TF-IDF features.
"""

# %%
from jurebes import IntentClassifier
from jurebes.baselines import BASELINES

samples = {
    "greet": ["hello", "hi", "hey there", "good morning"],
    "bye": ["goodbye", "bye", "see you", "later"],
}

for name in ("rbf_svc", "nusvc"):
    clf = IntentClassifier(BASELINES.build(name))
    for k, v in samples.items():
        clf.add_intent(k, v)
    clf.fit()
    r = clf.predict("hi friend")
    print(f"{name:8s} -> {r.intent} ({r.confidence:.3f})")
