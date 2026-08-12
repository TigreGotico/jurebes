"""Load a CSV dataset.

Writes a tiny CSV via tempfile and reads it with load_csv.
"""

# %%
import tempfile
from pathlib import Path

from jurebes.datasets import load_csv

csv_data = """text,intent
hello there,greet
hi friend,greet
goodbye now,bye
see you later,bye
"""
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "demo.csv"
    path.write_text(csv_data, encoding="utf-8")
    X, y = load_csv(path)

print(f"loaded {len(X)} samples")
print(f"labels={sorted(set(y))}")
print(f"X[0]={X[0]!r} y[0]={y[0]!r}")
