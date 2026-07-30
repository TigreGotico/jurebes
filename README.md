# jurebes 🐾

**J**ust-sklearn **U**tility for **R**eproducible **E**valuation of **B**aselines, **E**stimators and **S**olvers.

A classical-ML text classification research framework, built on plain scikit-learn. It uses no NLTK and no padacioso.

Jurebes wires **any sklearn featurizer** to **any sklearn classifier** behind a small text-classification API. It also includes a registry of ready-to-use baselines, a benchmark harness, dataset loaders, a CLI, and an OVOS pipeline plugin.

Intent classification is the main use case, but the core works with any single-label text task. Spam, sentiment, topic, and language-ID tasks run on the same API. See [the general text classification guide](docs/guides/general-text-classification.md).

> *Named in memory of Jurebes, the best dog.*

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

Pass any sklearn `Pipeline` or estimator instead of a baseline name. Jurebes wraps non-probabilistic estimators with `CalibratedClassifierCV` automatically, so `predict_proba` always works.

## Documentation

Full documentation lives under [`docs/`](docs/):

- [Getting started](docs/getting-started/01-what-is-jurebes.md): install, first classifier, core concepts, troubleshooting.
- [Guides](docs/guides/choosing-a-baseline.md): choosing a baseline, slots, calibration, reproducibility, debugging.
- [Theory](docs/theory/intent-classification.md): classical-ML foundations: featurization, linear models, kernels, ensembles, statistical comparison.
- [API reference](docs/reference/index.md): every public module, function, dataclass, and CLI flag.

## Research

Jurebes is built as a research framework:

- 48 named baselines tagged into groups (`linear`, `kernel`, `tree`, `naive_bayes`, `neural`, `reduced_dim`, `online`, `strategy`, `feature_engineering`, `ensemble`, `discriminant`, `categorical`). This includes neural-bottleneck autoencoder pipelines (`autoencoder_*`) and dict-input categorical pipelines (`categorical_*`).
- A hyperparameter search subsystem (`jurebes.search`) with grid, random, successive-halving, Bayesian (optional), and genetic (optional) backends.
- A benchmark harness with multi-metric scoring (`f1_macro`, `accuracy`, `log_loss`, `top_k_accuracy`, ...) and pooled tail-latency percentiles.
- See [`docs/research.md`](docs/research.md) and [`docs/search.md`](docs/search.md).

## Baselines

`BASELINES` is a registry of 48 named factories. It covers naive Bayes, logistic regression, linear/RBF SVMs, kNN, online learners (SGD, perceptron, passive-aggressive, ridge), tree ensembles (random forest, extra trees, gradient boosting, HistGBM, bagging), a shallow MLP, voting, stacking, reduced-dim methods (LSA/NMF/LDA/autoencoder), multi-class strategy wrappers (OvR/OvO), discriminant analysis, text-statistics composites, and dict-input categorical pipelines. List them:

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
| reduced_dim | `autoencoder_linear_svc` |
| reduced_dim | `autoencoder_logreg` |
| reduced_dim | `autoencoder_rbf_svc` |
| discriminant | `lda_classifier` |
| discriminant | `qda_classifier` |
| categorical | `categorical_logreg` |
| categorical | `categorical_random_forest` |

Rows total more than 48 because some baselines (for example `bagging_logreg`, `sgd_log`, `hashing_sgd_hinge`) belong to multiple groups. The registry itself holds 48 unique factories.

The `categorical_*` pipelines accept `list[dict[str, str]]` instead of raw text. They are available through the programmatic API only, and text-fixture benchmarks skip them.

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

`RunResult` captures accuracy, macro/micro F1, per-class F1, training time, predict-latency p50/p95/p99, model size, and the confusion matrix.

## Slots

Five pluggable taggers cover dictionary, template, sklearn IOB,
hybrid cascade, and CRF (optional extra). All are available through
the `TAGGERS` registry and share the same protocol.

```python
from jurebes import IntentClassifier, BASELINES
from jurebes.slots import TAGGERS, SklearnIOBTagger

clf = IntentClassifier(BASELINES.build("logreg"), tagger=SklearnIOBTagger())
# or by name:
clf = IntentClassifier(BASELINES.build("logreg"), tagger="hybrid")

clf.add_entity("name", ["bob", "alice"])
clf.add_intent("name", ["my name is {name}", "call me {name}"])
clf.add_intent("hello", ["hello", "hi"])
clf.fit()
clf.predict("my name is bob").entities  # -> {"name": "bob"}

TAGGERS.names()  # ['dictionary', 'template', 'sklearn_iob', 'hybrid', 'crf']
```

See [`docs/`](docs/) for the research guide, slot tagger details, and the OVOS deployment notes:

- [`docs/research.md`](docs/research.md): adding baselines, benchmarks, reports.
- [`docs/search.md`](docs/search.md): hyperparameter search backends.
- [`docs/slots.md`](docs/slots.md): IOB slot tagger and token features.
- [`docs/opm.md`](docs/opm.md): OVOS pipeline plugin configuration.
- [`MIGRATION.md`](MIGRATION.md): porting from the prior `JurebesIntentContainer` API.

## Examples

Runnable, cell-marked scripts live in [`examples/notebooks/`](examples/notebooks/):

- [`snips_quickstart.py`](examples/notebooks/snips_quickstart.py): fetch SNIPS, train, evaluate.
- [`banking77_full_research_flow.py`](examples/notebooks/banking77_full_research_flow.py): compare baselines, run Friedman+Nemenyi, tune the winner, evaluate on a holdout set.

Both examples need `pip install jurebes[hf,bench-plot]`.
