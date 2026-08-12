"""Linear baselines side by side: LogReg, LinearSVC, Ridge.

All three use the same TF-IDF features under the hood — the only thing
that changes is the linear classifier head.
"""

# %%
from jurebes import IntentClassifier
from jurebes.baselines import BASELINES

samples = {
    "greet": ["hello", "hi", "hey there"],
    "bye": ["goodbye", "bye", "see you"],
    "thanks": ["thank you", "thanks", "much appreciated"],
}

for name in ("logreg", "linear_svc", "ridge"):
    clf = IntentClassifier(BASELINES.build(name))
    for k, v in samples.items():
        clf.add_intent(k, v)
    clf.fit()
    r = clf.predict("thanks a lot")
    print(f"{name:12s} -> {r.intent} ({r.confidence:.3f})")
