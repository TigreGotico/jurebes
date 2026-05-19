"""Inspect every public field on ComparisonResult and RunResult.

Walks the public surface area so you know what each row contains.
"""

# %%
from dataclasses import fields

from jurebes.benchmark import compare
from jurebes.benchmark.metrics import RunResult

X = ["hello", "hi", "bye", "goodbye", "thanks", "thank you"] * 4
y = (["greet"] * 2 + ["bye"] * 2 + ["thanks"] * 2) * 4

result = compare(["logreg", "linear_svc"], X, y, k=2)
print(f"ComparisonResult.scoring = {result.scoring}")
print(f"fold_scores_by_baseline keys: {list(result.fold_scores_by_baseline)}")

print("\nRunResult fields:")
for f in fields(RunResult):
    print(f"  {f.name}: {f.type}")

r0 = result.rows[0]
print(f"\nfirst row name={r0.name} accuracy={r0.accuracy:.3f}")
print(f"  per_class_f1={r0.per_class_f1}")
print(f"  pooled p95={r0.predict_ms_p95_pooled:.2f}ms group={r0.group}")
