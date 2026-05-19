"""Load an OVOS-style intent directory.

.intent / .voc / .entity files in a directory; (a|b) alternation and
[optional] constructs are expanded; {entity} placeholders are collected
into the entities dict.
"""

# %%
import tempfile
from pathlib import Path

from jurebes.datasets import load_ovos_intents

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "greet.intent").write_text("(hi|hello) [there] friend\n", encoding="utf-8")
    (root / "weather.intent").write_text("weather in {city}\n", encoding="utf-8")
    (root / "city.entity").write_text("paris\nlisbon\n", encoding="utf-8")
    X, y, entities = load_ovos_intents(root)

print(f"samples expanded to: {X}")
print(f"labels: {y}")
print(f"entities: {entities}")
