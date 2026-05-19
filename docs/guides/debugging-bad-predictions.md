# Debugging bad predictions

When a trained classifier misclassifies, the diagnosis follows a fixed sequence: confusion matrix → misclassified samples → training-data sanity → per-class F1.

## Inspect the confusion matrix

```python
from jurebes.benchmark import cross_validate
r = cross_validate("logreg", X, y, k=5)
labels = r.labels
cm     = r.confusion_matrix

print(f"{'':20s} " + " ".join(f"{l[:8]:>8s}" for l in labels))
for i, true_lab in enumerate(labels):
    row = " ".join(f"{cm[i][j]:8d}" for j in range(len(labels)))
    print(f"{true_lab[:20]:20s} {row}")
```

Heavy off-diagonal mass between two labels means the model conflates them. Common causes:

- Lexical overlap between samples of the two intents.
- One intent's training samples are subsets of the other's.
- Labelling noise — some samples are arguably either class.

## Dump misclassified samples

```python
from sklearn.model_selection import train_test_split
from jurebes import IntentClassifier, BASELINES

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

clf = IntentClassifier(BASELINES.build("logreg"))
for lbl in sorted(set(y_train)):
    clf.add_intent(lbl, [x for x, yy in zip(X_train, y_train) if yy == lbl])
clf.fit()

for x, y_true in zip(X_test, y_test):
    pred = clf.predict(x)
    if pred.intent != y_true:
        print(f"true={y_true:20s} pred={pred.intent:20s} conf={pred.confidence:.3f}  | {x}")
```

Look for patterns: a specific phrase that always fools the model, a vocabulary item missing from the training set, capitalisation artefacts.

## Sanity-check the training data

Common gotchas:

- **Duplicate samples** across intents. Search for them:

  ```python
  from collections import defaultdict
  by_text = defaultdict(set)
  for x, lbl in zip(X, y):
      by_text[x].add(lbl)
  conflicts = {x: lbls for x, lbls in by_text.items() if len(lbls) > 1}
  for x, lbls in conflicts.items():
      print(f"{sorted(lbls)} | {x}")
  ```

  Identical text under two different labels is unresolvable noise.

- **Sample count imbalance** — see [handling-class-imbalance.md](handling-class-imbalance.md).

- **Vocabulary leakage** between the OVOS-conventional `{entity}` placeholders and the surface samples. A sample like `play {song}` expands at training time; verify the entity sample list is what you expect.

## Per-class F1

`RunResult.per_class_f1` exposes a dict `{label → f1}`:

```python
for label, f1 in sorted(r.per_class_f1.items(), key=lambda kv: kv[1]):
    print(f"  {label:30s} {f1:.3f}")
```

The bottom of the list points at the weakest class. Decide:

- If the class has too few samples — add more.
- If the class lexically overlaps another — rephrase samples to differentiate.
- If the class is genuinely hard — accept and route low-confidence predictions to clarification.

## Compare baselines

A weak class on one baseline may be fine on another. Run the standard compare:

```python
from jurebes.benchmark import compare, to_markdown
from jurebes.baselines import BASELINES

result = compare(BASELINES.resolve("@linear") + ["nb_complement"], X, y, k=5)
print(to_markdown(result, sort_by="macro_f1"))
```

When per-class F1 differs meaningfully between baselines, the *kind* of model matters more than the hyperparameters.

## When all baselines fail

If every baseline scores poorly on the same class:

- The class is too small (add data).
- The class is semantically diffuse (split into sub-intents or merge with a sibling).
- The featurization is wrong for the data (try `linear_svc_char` or `union_logreg` for short or typo-heavy inputs).

If the problem is a single confusing pair, consider:

- An IOB-tagged slot in place of one of the intents (the difference becomes a slot value).
- A two-stage classifier: an initial coarse model routes to a fine-grained sub-classifier on the ambiguous pair.

See [theory/intent-classification.md](../theory/intent-classification.md) for the broader framing.

---
- Back to [docs index](../index.md)
