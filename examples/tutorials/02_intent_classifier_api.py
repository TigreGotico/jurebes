"""IntentClassifier API tour.

Exercises add_intent, remove_intent, predict, predict_proba,
predict_batch and the _samples introspection attribute.
"""

# %% setup
from jurebes import IntentClassifier

clf = IntentClassifier()
clf.add_intent("greet", ["hello", "hi", "hey there"])
clf.add_intent("bye", ["goodbye", "bye", "see ya"])
clf.add_intent("thanks", ["thank you", "thanks a lot", "much appreciated"])

# %% introspect registered samples
print(f"intents={list(clf._samples)}")
print(f"greet samples={clf._samples['greet']}")

# %% remove then refit
clf.remove_intent("thanks")
clf.fit()

# %% single predict + ranked predict_proba
top = clf.predict("hello there")
print(f"top: {top.intent} ({top.confidence:.3f})")
for r in clf.predict_proba("see you tomorrow"):
    print(f"  {r.intent}: {r.confidence:.3f}")

# %% batch predict
batch = clf.predict_batch(["hi", "bye now"])
for r in batch:
    print(f"batch -> {r.utterance!r}: {r.intent}")
