"""Save and load a fitted SklearnIOBTagger.

Round-trips the tagger to disk and confirms predictions match.
"""

# %%
import tempfile
from pathlib import Path

from jurebes.slots import SklearnIOBTagger

tagger = SklearnIOBTagger()
tagger.add_entity("city", ["paris", "lisbon", "berlin"])
tagger.fit({"weather": ["weather in {city}", "forecast for {city}"]})

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "tagger.joblib"
    tagger.save(path)

    reloaded = SklearnIOBTagger.load(path)

    a = tagger.predict("weather in berlin")
    b = reloaded.predict("weather in berlin")
    print(f"original={a}")
    print(f"reloaded={b}")
    assert a == b
    print("round-trip OK")
