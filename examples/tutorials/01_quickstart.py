"""Quickstart: minimal IntentClassifier workflow.

Create a classifier, register two intents with a few samples each, fit,
then predict on a new utterance. Demonstrates the simplest possible
end-to-end use of jurebes.
"""

# %% imports
from jurebes import IntentClassifier

# %% define data
clf = IntentClassifier()
clf.add_intent("greet", ["hello", "hi there", "hey", "good morning"])
clf.add_intent("bye", ["goodbye", "see you", "bye", "later"])

# %% train
clf.fit()

# %% predict
result = clf.predict("hi friend")
print(f"intent={result.intent} confidence={result.confidence:.3f}")
print(f"utterance={result.utterance!r}")
