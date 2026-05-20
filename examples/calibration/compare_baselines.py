"""Compare calibration quality across several baselines on the same data.

Runs ``compare()`` with ECE and Brier among the scoring metrics, then
prints a markdown table sorted by ECE. Lower ECE = predicted confidence
better matches empirical accuracy.

This is the diagnostic to run before tuning ``hard_conf_max`` in the
active-learning loop (see ``examples/llm_augmentation/``) or before
setting OPM ``conf_high/med/low`` thresholds in production.
"""

from __future__ import annotations

from jurebes.benchmark import compare, to_markdown


# ── small but-not-toy dataset ───────────────────────────────────────
X = (
    ["play africa", "put on hey jude", "queue up bohemian rhapsody",
     "start smells like teen spirit", "spin africa", "throw on hey jude",
     "stream africa", "blast hey jude", "play bohemian rhapsody",
     "put on smells like teen spirit", "play africa now", "play hey jude please"]
    + ["set a timer for five minutes", "wake me in ten minutes",
       "remind me in twenty minutes", "timer for half an hour",
       "set timer for three minutes", "wake me up in an hour",
       "set alarm for fifteen minutes", "remind me in five minutes",
       "set a timer for two hours", "alarm in ten minutes",
       "remind me in seven minutes", "set a timer for one minute"]
    + ["how is the weather", "is it raining outside", "weather forecast today",
       "current temperature", "humidity level today", "is it sunny",
       "weather in lisbon", "is it snowing", "rain forecast",
       "current weather conditions", "is it cold outside", "weather forecast please"]
)
y = (["play_song"] * 12 + ["set_timer"] * 12 + ["weather"] * 12)

# ── compare with calibration metrics inline ─────────────────────────
result = compare(
    ["nb_multinomial", "logreg", "linear_svc", "linear_svc_char",
     "random_forest", "mlp_shallow"],
    X, y,
    k=3,
    scoring=("accuracy", "f1_macro", "ece", "brier", "log_loss"),
)

print(to_markdown(result, sort_by="ece", precision=4))

# ── interpretation ──────────────────────────────────────────────────
best = min(result.rows, key=lambda r: r.fold_scores["ece"][-1])
worst = max(result.rows, key=lambda r: r.fold_scores["ece"][-1])
best_ece = sum(best.fold_scores["ece"]) / len(best.fold_scores["ece"])
worst_ece = sum(worst.fold_scores["ece"]) / len(worst.fold_scores["ece"])

print()
print(f"best-calibrated: {best.name} (mean ECE {best_ece:.4f})")
print(f"worst-calibrated: {worst.name} (mean ECE {worst_ece:.4f})")
print()
print("Action: for active-learning hard_conf_max threshold tuning, use the")
print(f"best-calibrated baseline ({best.name}); the worst-calibrated one will")
print("produce HARD-bucket boundaries that don't match the actual confusion.")
