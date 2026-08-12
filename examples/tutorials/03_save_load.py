"""Save and load a fitted classifier.

Trains a classifier, persists it via joblib, reloads it, and asserts the
prediction matches.
"""

# %% setup + fit
import tempfile
from pathlib import Path

from jurebes import IntentClassifier

clf = IntentClassifier()
clf.add_intent("weather", ["what is the weather", "is it raining", "weather today"])
clf.add_intent("time", ["what time is it", "tell me the time", "current time"])
clf.fit()

# %% save + load
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "clf.joblib"
    clf.save(path)
    reloaded = IntentClassifier.load(path)

    a = clf.predict("is it raining")
    b = reloaded.predict("is it raining")
    print(f"original: {a.intent} ({a.confidence:.3f})")
    print(f"reloaded: {b.intent} ({b.confidence:.3f})")
    assert a.intent == b.intent
    print("prediction parity confirmed")
