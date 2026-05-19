"""Online linear baselines.

SGDClassifier with three loss functions plus perceptron and passive
aggressive.
"""

# %%
from jurebes import IntentClassifier
from jurebes.baselines import BASELINES

samples = {
    "greet": ["hello", "hi", "hey there", "good morning"],
    "bye": ["goodbye", "bye", "see you", "later"],
}
names = (
    "sgd_log", "sgd_hinge", "sgd_modified_huber",
    "perceptron", "passive_aggressive",
)
for name in names:
    clf = IntentClassifier(BASELINES.build(name))
    for k, v in samples.items():
        clf.add_intent(k, v)
    clf.fit()
    r = clf.predict("hi friend")
    print(f"{name:22s} -> {r.intent} ({r.confidence:.3f})")
