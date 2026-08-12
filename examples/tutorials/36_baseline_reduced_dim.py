"""Reduced-dimensionality baselines.

LSA, NMF and autoencoder bottlenecks feeding into LogisticRegression.
"""

# %%
from jurebes import IntentClassifier
from jurebes.baselines import BASELINES

_greet = ["hello world", "hi there", "hey friend", "good morning sunshine",
          "howdy partner", "hey buddy", "morning everyone", "greetings folks",
          "salutations dear", "good day", "hello again", "hey you",
          "morning star", "afternoon greetings", "evening hello", "hi mate"]
_bye = ["goodbye now", "see you later", "farewell my friend", "catch you soon",
        "bye now", "until next time", "take care", "so long buddy",
        "ciao bella", "adieu", "later alligator", "talk soon",
        "in a while", "goodbye forever", "see ya", "off i go"]
_thanks = ["thank you kindly", "thanks a lot", "much appreciated indeed", "cheers mate",
           "thanks so much", "ta indeed", "grateful for it", "kind of you",
           "appreciated greatly", "many thanks", "obliged sir", "thanks again",
           "very grateful", "thank you so", "much obliged", "with gratitude"]
_weather = ["how is weather", "is sunny outside", "is raining today",
            "weather forecast please", "temperature now", "current conditions",
            "humidity level high", "wind speed strong", "is cold today", "warm or chilly",
            "snowing now outside", "cloudy skies above", "foggy morning here",
            "is nice outside", "hot summer day", "cool autumn breeze"]
samples = {"greet": _greet, "bye": _bye, "thanks": _thanks, "weather": _weather}
for name in ("lsa_logreg", "nmf_logreg", "autoencoder_logreg"):
    clf = IntentClassifier(BASELINES.build(name))
    for k, v in samples.items():
        clf.add_intent(k, v)
    clf.fit()
    r = clf.predict("see you")
    print(f"{name:22s} -> {r.intent} ({r.confidence:.3f})")
