# Cookbook: full research pipeline

Soup-to-nuts: dataset → portfolio compare → Friedman+Nemenyi → tune the winner → test evaluation → save → load → predict. The canonical jurebes research workflow.

## Prerequisites

```bash
pip install jurebes[hf,bench-plot,search-bayes]
```

## Script

```python
from pathlib import Path

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from jurebes import IntentClassifier
from jurebes.baselines import BASELINES
from jurebes.benchmark import compare, to_markdown
from jurebes.benchmark.stats import critical_difference, friedman_nemenyi
from jurebes.datasets.canonical import load_banking77
from jurebes.search import search, spaces


# 1. Dataset
X, y = load_banking77()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0,
)
print(f"train {len(X_train)}, test {len(X_test)}, classes {len(set(y))}")


# 2. Portfolio compare across families
portfolio = (
    BASELINES.resolve("@linear")
    + BASELINES.resolve("@naive_bayes")
    + ["random_forest", "hist_gbm"]
)
print(f"comparing {len(portfolio)} baselines")

result = compare(portfolio, X_train, y_train, k=5,
                 scoring=("accuracy", "f1_macro", "log_loss"), seed=0)

Path("portfolio_report.md").write_text(
    to_markdown(result, sort_by="macro_f1", with_significance=True, precision=4),
    encoding="utf-8",
)


# 3. Friedman + Nemenyi
fold_macro = {
    name: result.fold_scores_by_baseline[name]["f1_macro"]
    for name in portfolio
}
fr = friedman_nemenyi(fold_macro)
cd = critical_difference(fold_macro)
print(cd.to_ascii())
print(f"Friedman stat={fr.statistic:.3f} p={fr.pvalue:.4f} reject_null={fr.reject_null}")


# 4. Pick the winner (highest mean rank, lowest fold-rank value)
winner = min(cd.mean_ranks.items(), key=lambda kv: kv[1])[0]
print(f"winner: {winner}")


# 5. Tune the winner with random search
space = spaces.for_baseline(winner) if winner in spaces.available() else None
if space is None:
    print(f"no predefined space for {winner}; skipping tune")
    tuned = IntentClassifier(BASELINES.build(winner))
    for label in sorted(set(y_train)):
        tuned.add_intent(label, [x for x, yy in zip(X_train, y_train) if yy == label])
    tuned.fit()
else:
    r = search(winner, space, X_train, y_train,
               backend="random", n_iter=40, cv=5,
               scoring="f1_macro", seed=0)
    print(f"tuned CV macro-F1: {r.best_score:.4f}  params={r.best_params}")
    tuned = r.best_estimator


# 6. Held-out test evaluation
preds = [tuned.predict(x).intent for x in X_test]
test_macro = f1_score(y_test, preds, average="macro", zero_division=0)
print(f"held-out test macro-F1: {test_macro:.4f}")


# 7. Save the deployable model
tuned.save("banking_winner.joblib")


# 8. Load and predict
clf2 = IntentClassifier.load("banking_winner.joblib")
demo = clf2.predict("how do I report a lost card")
print(demo.intent, round(demo.confidence, 3))
```

## Reading a critical-difference diagram

```
Critical Difference = 0.823  (n=5)

rank  baseline
1.800  linear_svc
2.200  logreg
2.400  union_logreg
8.600  random_forest

statistically indistinguishable groups:
  {linear_svc, logreg, union_logreg}
  {random_forest}
```

The first group contains baselines whose mean ranks are within CD of each other — any of them is a defensible winner on macro-F1; tie-break with latency or model size.

## Production handoff

The saved `banking_winner.joblib` file is the entire deployable artefact. Production wiring needs:

- The pinned `scikit-learn` version that produced the file.
- The pinned `jurebes` version (recorded in `_loaded_from_version`).
- A loader call: `IntentClassifier.load(path)`.

See [production-deployment.md](production-deployment.md) for the OVOS pipeline-plugin packaging story.

## What "research pipeline" means here

The protocol above is the lighter shadow of what is typically called "ML research workflow":

1. **Hypothesise** which baseline family fits the data shape.
2. **Compare** the candidates with a controlled metric and stratified CV.
3. **Statistically validate** via Friedman+Nemenyi to avoid reporting noise as signal.
4. **Tune** the validated winner.
5. **Evaluate** on a held-out test set — the only unbiased number.
6. **Persist** the artefact along with provenance (git SHA, library versions).

Skipping steps 3 and 5 is the most common research-report sin. Both are cheap with jurebes; there is no reason to skip them.

## Variations

- Replace `load_banking77` with `load_csv("my_data.csv")`.
- Swap `portfolio` for a single family: `BASELINES.resolve("@reduced_dim")`.
- Add a Bayesian search step for continuous hyperparameters:

  ```python
  from skopt.space import Real, Categorical
  bayes_space = {
      "clf__C": Real(0.01, 100, prior="log-uniform"),
      "feat__ngram_range": Categorical([(1, 1), (1, 2), (1, 3)]),
  }
  r = search("logreg", bayes_space, X_train, y_train,
             backend="bayes", n_iter=30, cv=5)
  ```

## Related cookbooks

- [compare-all-linear.md](compare-all-linear.md) — narrower portfolio.
- [tune-with-random-search.md](tune-with-random-search.md) — focused tuning step.
- [production-deployment.md](production-deployment.md) — deploying the artefact.

---
- Back to [docs index](../index.md)
