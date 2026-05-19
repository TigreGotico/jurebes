"""IntentClassifier with a slot tagger.

Pass an SklearnIOBTagger to IntentClassifier; entities populate the
IntentResult.entities dict alongside the intent label.
"""

# %%
from jurebes import IntentClassifier
from jurebes.slots import SklearnIOBTagger

tagger = SklearnIOBTagger()
clf = IntentClassifier(tagger=tagger)
clf.add_entity("city", ["paris", "lisbon", "berlin"])

clf.add_intent("weather", [
    "weather in {city}",
    "what is the weather in {city}",
])
clf.add_intent("bye", ["goodbye", "bye", "see you"])
clf.fit()

r = clf.predict("weather in lisbon")
print(f"intent={r.intent} confidence={r.confidence:.3f}")
print(f"entities={r.entities}")
