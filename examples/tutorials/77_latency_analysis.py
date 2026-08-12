"""Pooled p50/p95/p99 latency inspection.

RunResult exposes pooled percentiles across every individual prediction
made across all CV folds — meaningful for tail latency.
"""

# %%
from jurebes.benchmark import compare

X = ["hello", "hi", "bye", "goodbye", "thanks", "thank you"] * 6
y = (["greet"] * 2 + ["bye"] * 2 + ["thanks"] * 2) * 6

result = compare(["logreg", "linear_svc", "random_forest"], X, y, k=2)
for r in result.rows:
    print(
        f"{r.name:15s} p50={r.predict_ms_p50_pooled:.2f}ms "
        f"p95={r.predict_ms_p95_pooled:.2f}ms "
        f"p99={r.predict_ms_p99_pooled:.2f}ms"
    )
