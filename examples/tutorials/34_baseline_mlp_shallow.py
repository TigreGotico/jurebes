"""Shallow MLP baseline.

A single 64-unit hidden layer on top of TF-IDF features.
"""

# %%
from jurebes import IntentClassifier
from jurebes.baselines import BASELINES

clf = IntentClassifier(BASELINES.build("mlp_shallow"))
clf.add_intent("greet", ["hello", "hi", "hey there", "good morning"])
clf.add_intent("bye", ["goodbye", "bye", "see you", "later"])
clf.fit()

r = clf.predict("hello there")
print(f"mlp_shallow -> {r.intent} ({r.confidence:.3f})")
