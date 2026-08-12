"""Naive Bayes variants: multinomial, complement, bernoulli.

Train each on the same toy data and print top-1 prediction.
"""

# %%
from jurebes import IntentClassifier
from jurebes.baselines import BASELINES

samples = {
    "greet": ["hello", "hi", "hey there", "good morning"],
    "bye": ["goodbye", "bye", "see you", "later"],
}

for name in ("nb_multinomial", "nb_complement", "nb_bernoulli"):
    clf = IntentClassifier(BASELINES.build(name))
    for intent, utts in samples.items():
        clf.add_intent(intent, utts)
    clf.fit()
    r = clf.predict("hello friend")
    print(f"{name:15s} -> {r.intent} ({r.confidence:.3f})")
