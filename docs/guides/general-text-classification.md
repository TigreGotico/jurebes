# jurebes for general text classification

Intent classification is one application of text classification. The
jurebes core — the classifier, the baseline registry, the benchmark
harness, the search subsystem, calibration, active learning and
semi-supervised learning — is domain-agnostic. Spam filtering,
sentiment analysis, topic labelling, language identification,
toxic-comment detection and any other single-label text-classification
task run on the same API with no code changes.

## The vocabulary mapping

| intent-classification term | general text-classification term |
| --- | --- |
| intent | class / label |
| utterance | document / text |
| `add_intent(name, samples)` | `add_class(name, samples)` |
| `IntentResult.intent` | `.label` |
| `IntentResult.utterance` | `.text` |
| `IntentClassifier` | `TextClassifier` (same class) |

`TextClassifier` is `IntentClassifier` under a second name; `add_class`
is `add_intent`; `.label` and `.text` are read-only aliases on the
result object. Use whichever vocabulary fits your task — they are
interchangeable and you can mix them.

## What carries over unchanged

Everything except the NLU application layer:

- **All 53 baselines** in `BASELINES` — Naive Bayes, linear models,
  kernel SVM, trees, ensembles, reduced-dim, label-guided embeddings.
- **The benchmark harness** — `compare`, `cross_validate`, `train_test`,
  Friedman + Nemenyi statistical comparison, calibration diagnostics
  (ECE / Brier / reliability curves).
- **The search subsystem** — grid / random / halving / Bayesian /
  genetic hyperparameter search.
- **`active_learning`** — uncertainty sampling, query-by-committee,
  confusion-pair mining.
- **`semi_supervised`** — self-training, co-training, label propagation.
- **Dataset loaders** — `load_csv`, `load_jsonl`, `load_hf`.

## What is intent/NLU-specific

These sit as an application layer on top of the generic core; ignore
them for non-intent tasks:

- **Slot extraction** (`jurebes.slots`) — the IOB / CRF / dictionary
  taggers. Text classification has no slots.
- **The OVOS pipeline plugin** (`jurebes.opm`) — deploys a classifier
  inside an OpenVoiceOS voice assistant.
- **OVOS dataset loaders** — `load_ovos_intents`, `load_intents_for_eval`,
  `load_massive_templates`.

## Example: spam filtering

```python
from jurebes import TextClassifier, BASELINES

clf = TextClassifier(BASELINES.build("linear_svc_char"))
clf.add_class("spam", [
    "win a free prize now", "claim your reward click here",
    "cheap meds no prescription", "you have won the lottery",
])
clf.add_class("ham", [
    "are we still on for lunch", "the meeting moved to 3pm",
    "can you send me the report", "see you at the weekend",
])
clf.fit()

r = clf.predict("free prize claim now")
print(r.label, round(r.confidence, 3))   # spam 0.9...
```

## Example: sentiment analysis

```python
from jurebes import TextClassifier, BASELINES

clf = TextClassifier(BASELINES.build("logreg"))
clf.add_class("positive", ["loved it", "fantastic experience", "highly recommend"])
clf.add_class("negative", ["terrible service", "would not return", "very disappointed"])
clf.add_class("neutral",  ["it was okay", "nothing special", "average at best"])
clf.fit()

print(clf.predict("fantastic, highly recommend").label)
```

## Example: benchmarking baselines on a CSV dataset

The benchmark harness does not care whether the labels are intents or
topic categories:

```python
from jurebes.datasets import load_csv
from jurebes.benchmark import compare, to_markdown

X, y = load_csv("reviews.csv", text="review", label="sentiment")
result = compare(
    ["nb_multinomial", "logreg", "linear_svc_char", "random_forest"],
    X, y, k=5, scoring=("accuracy", "f1_macro", "ece"),
)
print(to_markdown(result, sort_by="f1_macro"))
```

Run statistical comparison, calibration analysis, hyperparameter search
and the active-learning / semi-supervised loops exactly as documented
for intent classification — the guides apply verbatim, just read
"class" for "intent".

## When to reach for the intent layer

Use `add_intent` / `IntentClassifier` / the slot tagger / the OPM
plugin when you are building a voice-assistant NLU component. Use
`add_class` / `TextClassifier` for everything else. The underlying
model, training, evaluation and tuning are identical.

---
[← back to docs index](../index.md)
