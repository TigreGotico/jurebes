# jurebes 🐾

**J**ust-sklearn **U**tility for **R**eproducible **E**valuation of **B**aselines, **E**stimators and **S**olvers.

A classical-ML intent classification research framework — pure scikit-learn, no NLTK, no padacioso.

Jurebes lets you wire **any sklearn featurizer + any sklearn classifier** behind a small intent-classification API, plus a registry of ready-to-use baselines, a benchmark harness, dataset loaders, a CLI, and an OVOS pipeline plugin.

> *Named in memory of Jurebes, the author's dog.*

## Install

```bash
pip install jurebes
# optional extras
pip install jurebes[hf]              # HuggingFace dataset loader
pip install jurebes[search-bayes]    # Bayesian hyperparameter search (skopt)
pip install jurebes[search-genetic]  # Genetic-algorithm search (sklearn-genetic-opt)
pip install jurebes[search-all]      # both bayes and genetic
pip install jurebes[bench-plot]      # matplotlib-based comparison plots
pip install jurebes[test]            # pytest stack
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

## Research

Jurebes is built as a research framework:

- 43 named baselines tagged into groups (`linear`, `kernel`, `tree`, `naive_bayes`, `neural`, `reduced_dim`, `online`, `strategy`, `feature_engineering`, `ensemble`).
- Hyperparameter search subsystem (`jurebes.search`) supporting grid, random, successive-halving, Bayesian (optional), and genetic (optional) backends.
- Benchmark harness with multi-metric scoring (`f1_macro`, `accuracy`, `log_loss`, `top_k_accuracy`, ...) and pooled tail-latency percentiles.
- See [`docs/research.md`](docs/research.md) and [`docs/search.md`](docs/search.md).

## Baselines

`BASELINES` is a registry of 43 named factories covering naive Bayes, logistic regression, linear/RBF SVMs, kNN, online learners (SGD, perceptron, passive-aggressive, ridge), tree ensembles (random forest, extra trees, gradient boosting, HistGBM, bagging), shallow MLP, voting, stacking, reduced-dim (LSA/NMF/LDA), multi-class strategy wrappers (OvR/OvO), discriminant analysis, and text-statistics composites. List them:

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
