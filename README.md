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

- 43 named baselines tagged into groups (`linear`, `kernel`, `tree`, `naive_bayes`, `neural`, `reduced_dim`, `online`, `strategy`, `feature_engineering`, `ensemble`, `discriminant`).
- Hyperparameter search subsystem (`jurebes.search`) supporting grid, random, successive-halving, Bayesian (optional), and genetic (optional) backends.
- Benchmark harness with multi-metric scoring (`f1_macro`, `accuracy`, `log_loss`, `top_k_accuracy`, ...) and pooled tail-latency percentiles.
- See [`docs/research.md`](docs/research.md) and [`docs/search.md`](docs/search.md).

## Baselines

`BASELINES` is a registry of 43 named factories covering naive Bayes, logistic regression, linear/RBF SVMs, kNN, online learners (SGD, perceptron, passive-aggressive, ridge), tree ensembles (random forest, extra trees, gradient boosting, HistGBM, bagging), shallow MLP, voting, stacking, reduced-dim (LSA/NMF/LDA), multi-class strategy wrappers (OvR/OvO), discriminant analysis, and text-statistics composites. List them:

```bash
jurebes list-baselines
```

### Registry table

| group | baseline |
|---|---|
| linear | `hashing_sgd_hinge` |
| linear | `hashing_sgd_log` |
| linear | `linear_svc` |
| linear | `linear_svc_char` |
| linear | `linear_svc_hinge` |
| linear | `logreg` |
| linear | `logreg_char` |
| linear | `logreg_elasticnet` |
| linear | `logreg_l1` |
| linear | `passive_aggressive` |
| linear | `perceptron` |
| linear | `ridge` |
| linear | `sgd_hinge` |
| linear | `sgd_log` |
| linear | `sgd_modified_huber` |
| kernel | `knn` |
| kernel | `lsa_rbf_svc` |
| kernel | `nusvc` |
| kernel | `rbf_svc` |
| tree | `bagging_logreg` |
| tree | `decision_tree` |
| tree | `extra_trees` |
| tree | `gradient_boosting` |
| tree | `hist_gbm` |
| tree | `random_forest` |
| naive_bayes | `complement_nb_count` |
| naive_bayes | `nb_bernoulli` |
| naive_bayes | `nb_complement` |
| naive_bayes | `nb_multinomial` |
| neural | `mlp_shallow` |
| reduced_dim | `lda_logreg` |
| reduced_dim | `lsa_linear_svc` |
| reduced_dim | `lsa_logreg` |
| reduced_dim | `lsa_rbf_svc` |
| reduced_dim | `nmf_logreg` |
| online | `hashing_sgd_hinge` |
| online | `hashing_sgd_log` |
| online | `sgd_hinge` |
| online | `sgd_log` |
| online | `sgd_modified_huber` |
| strategy | `ovo_linear_svc` |
| strategy | `ovr_linear_svc` |
| feature_engineering | `text_stats_logreg` |
| feature_engineering | `union_text_stats_logreg` |
| ensemble | `bagging_logreg` |
| ensemble | `stacking` |
| ensemble | `union_logreg` |
| ensemble | `voting_soft` |
| discriminant | `lda_classifier` |
| discriminant | `qda_classifier` |

Rows total 50 because some baselines (e.g. `bagging_logreg`, `sgd_log`, `hashing_sgd_hinge`) belong to multiple groups. The registry itself contains 43 unique factories.

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

See [`docs/`](docs/) for the research guide, slot tagger details, and the OVOS deployment notes:

- [`docs/research.md`](docs/research.md) — adding baselines, benchmarks, reports.
- [`docs/search.md`](docs/search.md) — hyperparameter search backends.
- [`docs/slots.md`](docs/slots.md) — IOB slot tagger and token features.
- [`docs/opm.md`](docs/opm.md) — OVOS pipeline plugin configuration.
- [`MIGRATION.md`](MIGRATION.md) — porting from prior `JurebesIntentContainer` API.
