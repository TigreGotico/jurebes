# Your first classifier

This walkthrough takes the simplest possible jurebes pipeline from definition to prediction, swaps the underlying baseline, then saves and reloads a trained model.

## A three-intent classifier

```python
from jurebes import IntentClassifier, BASELINES

clf = IntentClassifier(BASELINES.build("logreg"))

clf.add_intent("greet", [
    "hello", "hi", "hey", "good morning", "good evening",
])
clf.add_intent("farewell", [
    "bye", "goodbye", "see you later", "see you tomorrow", "take care",
])
clf.add_intent("thanks", [
    "thanks", "thank you", "thanks a lot", "cheers", "much appreciated",
])
clf.fit()

result = clf.predict("hey there")
print(result.intent, round(result.confidence, 3))
```

`IntentClassifier.predict(utt)` returns an `IntentResult` with four fields: `intent`, `confidence`, `entities` (empty here — no slot tagger), and `utterance`.

## Inspecting all class probabilities

```python
ranked = clf.predict_proba("see you later")
for r in ranked:
    print(f"  {r.intent:10s} {r.confidence:.3f}")
```

`predict_proba` returns a list of `IntentResult` sorted by descending confidence — one entry per known intent. The top entry equals what `predict()` returns.

## Swapping the baseline

Every baseline name in the registry can be substituted with no other code change:

```python
from jurebes import IntentClassifier, BASELINES

for name in ("logreg", "linear_svc", "nb_complement", "rbf_svc"):
    clf = IntentClassifier(BASELINES.build(name))
    clf.add_intent("greet",    ["hello", "hi", "hey there"])
    clf.add_intent("farewell", ["bye", "goodbye", "see ya"])
    clf.add_intent("thanks",   ["thanks", "thank you", "cheers"])
    clf.fit()
    r = clf.predict("hey")
    print(f"{name:14s} -> {r.intent} ({r.confidence:.3f})")
```

`BASELINES.names()` lists every registered baseline. See [reference/baselines.md](../reference/baselines.md) for the full table grouped by family.

## Saving and loading

```python
clf.save("greet_clf.joblib")

# … later, in a fresh process:
from jurebes import IntentClassifier
clf2 = IntentClassifier.load("greet_clf.joblib")
print(clf2.predict("good evening").intent)
```

The joblib payload includes the fitted estimator, the slot tagger (if any), the original sample bank, and a `_jurebes_version` tag for cross-version diagnostics. See [guides/saving-and-loading.md](../guides/saving-and-loading.md) for security caveats and file-size expectations per baseline family.

## Next steps

- Add a slot tagger: [guides/adding-slots.md](../guides/adding-slots.md)
- Pick the right baseline for your dataset: [guides/choosing-a-baseline.md](../guides/choosing-a-baseline.md)
- Run a CV benchmark across many baselines: [cookbook/compare-all-linear.md](../cookbook/compare-all-linear.md)

---
- Previous: [Installation](02-installation.md)
- Next: [Core concepts](04-core-concepts.md)
