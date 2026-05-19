# jurebes 🐾

**J**ust-sklearn **U**tility for **R**eproducible **E**valuation of **B**aselines, **E**stimators and **S**olvers.

A classical-ML intent classification research framework — pure scikit-learn, no NLTK, no padacioso.

Jurebes lets you wire **any sklearn featurizer + any sklearn classifier** behind a small intent-classification API, plus a registry of ready-to-use baselines, a benchmark harness, dataset loaders, a CLI, and an OVOS pipeline plugin.

> *Named in memory of Jurebes, the author's dog.*

## Install

```bash
pip install jurebes
# optional extras
pip install jurebes[hf]    # HuggingFace dataset loader
pip install jurebes[test]  # pytest stack
```

## Quickstart

```python
from jurebes import IntentClassifier, BASELINES

clf = IntentClassifier(BASELINES.build("logreg"))
clf.add_intent("hello", ["hello", "hi", "hey there"])
clf.add_intent("joke",  ["tell me a joke", "say a joke", "make me laugh"])
clf.fit()

result = clf.predict("hi there")
print(result.intent, result.confidence, result.entities)
```

Pass any sklearn `Pipeline` / estimator instead of a baseline name; Jurebes auto-wraps non-probabilistic estimators with `CalibratedClassifierCV` so `predict_proba` always works.

## Baselines

`BASELINES` is a registry of ~23 named factories covering naive Bayes, logistic regression, linear/RBF SVMs, kNN, online learners (SGD, perceptron, passive-aggressive, ridge), tree ensembles (random forest, extra trees, gradient boosting, HistGBM), shallow MLP, voting, stacking, and a word+char union. List them:

```bash
jurebes list-baselines
```

Register your own:

```python
from jurebes import BASELINES
BASELINES.register("my_pipeline", lambda: build_my_sklearn_pipeline())
```

## Benchmark

```bash
jurebes benchmark --dataset data.csv \
    --baselines logreg,nb_multinomial,linear_svc --cv 5 \
    --out report.md
```

Or programmatically:

```python
from jurebes.benchmark import compare, to_markdown
from jurebes.datasets import load_csv

X, y = load_csv("data.csv")
result = compare(["logreg", "linear_svc", "nb_multinomial"], X, y, k=5)
print(to_markdown(result))
```

`RunResult` captures accuracy, macro/micro F1, per-class F1, training time, predict-latency p50/p95/p99, model size, and confusion matrix.

## Slots

Optional per-token IOB tagger:

```python
from jurebes import IntentClassifier, BASELINES
from jurebes.slots import SklearnIOBTagger

clf = IntentClassifier(BASELINES.build("logreg"), tagger=SklearnIOBTagger())
clf.add_entity("name", ["bob", "alice"])
clf.add_intent("name", ["my name is {name}", "call me {name}"])
clf.add_intent("hello", ["hello", "hi"])
clf.fit()
clf.predict("my name is bob").entities  # -> {"name": "bob"}
```

See [`docs/`](docs/) for the research guide, slot tagger details, and the OVOS deployment notes.
