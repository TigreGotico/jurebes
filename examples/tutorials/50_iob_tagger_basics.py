"""SklearnIOBTagger basics.

Train a per-token IOB tagger on a toy intent dataset with one entity
type, then predict on a fresh utterance.
"""

# %%
from jurebes.slots import SklearnIOBTagger

tagger = SklearnIOBTagger()
tagger.add_entity("city", ["paris", "lisbon", "berlin"])

intent_samples = {
    "weather": [
        "weather in {city}",
        "what is the weather in {city}",
        "tell me the forecast for {city}",
    ],
}
tagger.fit(intent_samples)

print(f"fitted={tagger.fitted}")
print(f"tag(paris query)={tagger.tag('weather in lisbon')}")
print(f"predict={tagger.predict('weather in lisbon')}")
