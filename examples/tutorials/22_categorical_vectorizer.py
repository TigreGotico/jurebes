"""CategoricalVectorizer: dict-of-string features with JSON round-trip.

One-hot over key=value pairs. Vocabulary persists to JSON (no pickle).
"""

# %%
import tempfile
from pathlib import Path

from jurebes.featurizers import CategoricalVectorizer

rows = [
    {"colour": "red", "size": "small"},
    {"colour": "blue", "size": "large"},
    {"colour": "red", "size": "large"},
]
vec = CategoricalVectorizer()
X = vec.fit_transform(rows)
print(f"shape={X.shape} vocab={vec.vocabulary_}")

# %% JSON save/load
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "vocab.json"
    vec.save(str(p))
    loaded = CategoricalVectorizer()
    loaded.load(str(p))
    print(f"loaded vocab matches: {loaded.vocabulary_ == vec.vocabulary_}")
    print(f"inverse_transform[0]={vec.inverse_transform(X)[0]}")
