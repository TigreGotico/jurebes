"""Export a comparison to JSON and round-trip the dict.

to_json returns a string suitable for archiving benchmark runs.
"""

# %%
import json

from jurebes.benchmark import compare, to_json

X = ["hello", "hi", "bye", "goodbye", "thanks", "thank you"] * 4
y = (["greet"] * 2 + ["bye"] * 2 + ["thanks"] * 2) * 4

result = compare(["logreg", "nb_multinomial"], X, y, k=2)
payload = to_json(result)
data = json.loads(payload)
print(f"top-level keys: {sorted(data)}")
print(f"baseline names in JSON: {[r['name'] for r in data['rows']]}")
