"""Discriminant analysis baselines: LDA and QDA.

LDA assumes shared covariance, QDA per-class covariance. Both need
dense input — handled inside the BASELINES factories.
"""

# %%
from jurebes import IntentClassifier
from jurebes.baselines import BASELINES

_greet = ["hello world", "hi there", "hey friend", "good morning sunshine",
          "howdy partner", "salutations dear", "morning everyone", "good day folks",
          "hey buddy", "greetings traveler", "afternoon all", "evening hello",
          "hello again", "hi mate", "morning star", "good day sir",
          "hey there pal", "hi everyone here", "greetings folks here", "hello friend dear",
          "morning sunshine bright", "evening folks all", "good afternoon there", "hi there friend"]
_bye = ["goodbye now", "see you later", "farewell my friend", "catch you soon",
        "take care now", "until next time friend", "so long buddy", "ciao for now",
        "later alligator", "adieu friend", "bye now folks", "talk to you soon",
        "in a while crocodile", "goodbye forever then", "off i go now", "see ya",
        "bye bye friend", "farewell now dear", "until we meet", "take it easy",
        "have a good one", "be well friend", "stay safe out", "catch up soon"]
_thanks = ["thank you kindly", "thanks a lot", "much appreciated indeed", "cheers mate",
           "grateful for help", "kind of you sir", "many thanks friend", "ta indeed",
           "appreciated greatly", "obliged sir kindly", "very grateful indeed", "thanks again friend",
           "with gratitude all", "thank you so much", "much obliged sir", "great help thanks",
           "you are kind", "thanks for help", "appreciate the help", "grateful indeed friend",
           "kindly appreciated all", "many many thanks", "thanks so kindly", "warm thanks here"]
_weather = ["how is the weather", "is it sunny outside", "raining today outside",
            "weather forecast please tell", "current temperature here now", "humidity level today",
            "wind speed report please", "warm or cold outside", "snowing now outside",
            "cloudy skies above", "foggy this morning here", "is nice outside today",
            "hot summer day", "cool autumn breeze here", "winter chill outside", "spring warmth",
            "is freezing today", "muggy weather here", "windy day outside", "rain forecast",
            "sunny skies all", "overcast today here", "drizzle this morning", "storm coming"]
samples = {"greet": _greet, "bye": _bye, "thanks": _thanks, "weather": _weather}
for name in ("lda_classifier", "qda_classifier"):
    clf = IntentClassifier(BASELINES.build(name))
    for k, v in samples.items():
        clf.add_intent(k, v)
    clf.fit()
    r = clf.predict("good morning")
    print(f"{name:16s} -> {r.intent} ({r.confidence:.3f})")
