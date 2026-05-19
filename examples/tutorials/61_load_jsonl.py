"""Load a JSONL dataset.

One JSON object per line with text and label keys.
"""

# %%
import json
import tempfile
from pathlib import Path

from jurebes.datasets import load_jsonl

rows = [
    {"text": "hello there", "intent": "greet"},
    {"text": "goodbye now", "intent": "bye"},
    {"text": "thanks a lot", "intent": "thanks"},
]
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "demo.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    X, y = load_jsonl(path)

print(f"loaded {len(X)} samples")
print(f"X={X}")
print(f"y={y}")
