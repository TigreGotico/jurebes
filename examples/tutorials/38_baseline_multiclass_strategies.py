"""Multiclass strategies: one-vs-rest vs one-vs-one LinearSVC.

OvR trains one binary classifier per class; OvO trains one per pair.
"""

# %%
from jurebes import IntentClassifier
from jurebes.baselines import BASELINES

samples = {
    "greet": ["hello", "hi", "hey there"],
    "bye": ["goodbye", "bye", "see you"],
    "thanks": ["thank you", "thanks", "much appreciated"],
}
for name in ("ovr_linear_svc", "ovo_linear_svc"):
    clf = IntentClassifier(BASELINES.build(name))
    for k, v in samples.items():
        clf.add_intent(k, v)
    clf.fit()
    r = clf.predict("thanks a lot")
    print(f"{name:18s} -> {r.intent} ({r.confidence:.3f})")
