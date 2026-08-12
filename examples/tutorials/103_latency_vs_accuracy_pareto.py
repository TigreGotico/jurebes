"""Scatter accuracy vs p95 latency to surface the Pareto frontier.

Plotting requires matplotlib (extra: bench-plot). The script always
prints the (accuracy, latency) pairs first; plotting is gated behind a
try/except.
"""

# %%
from jurebes.benchmark import compare

X = ["hello", "hi", "bye", "goodbye", "thanks", "thank you"] * 6
y = (["greet"] * 2 + ["bye"] * 2 + ["thanks"] * 2) * 6

result = compare(["logreg", "linear_svc", "random_forest", "nb_multinomial"], X, y, k=2)
points = [(r.name, r.macro_f1, r.predict_ms_p95_pooled) for r in result.rows]
for name, f1, p95 in points:
    print(f"{name:15s} macro_f1={f1:.3f} p95={p95:.2f}ms")

# %% optional matplotlib scatter
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for name, f1, p95 in points:
        ax.scatter(p95, f1)
        ax.annotate(name, (p95, f1))
    ax.set_xlabel("p95 latency (ms)")
    ax.set_ylabel("macro F1")
    ax.set_title("accuracy vs p95 latency")
    print("matplotlib figure prepared (not shown in non-interactive mode)")
except ImportError:
    print("install jurebes[bench-plot] to render the Pareto chart")
