# Core concepts

A combined glossary and conceptual map for jurebes.

## Data terms

**Utterance.** A single natural-language input string. "play some jazz", "what time is it".

**Sample.** A training utterance paired with a target label. The training set is a list of samples per intent.

**Label.** The discrete class assigned to a sample. Synonymous with *intent name* in this codebase.

**Intent.** A semantic category of user requests. In jurebes intents are flat string names; nested hierarchies are out of scope.

**Slot.** A named span extracted from an utterance ("artist" = "Miles Davis"). Slots are populated by an IOB tagger attached to the classifier.

**Entity.** The same idea as a slot, viewed from the tagger's side: the set of valid string values that can fill a slot. In `clf.add_entity("name", ["bob", "alice"])`, `name` is the entity name and the list is its sample values.

**IOB tagging.** Inside/Outside/Beginning per-token labelling, a standard scheme for slot filling. A token labelled `B-artist` begins an artist span, `I-artist` continues it, `O` is outside any span.

## Algorithmic terms

**Featurizer.** Anything mapping raw utterance strings to a numeric matrix. In jurebes a featurizer is a scikit-learn `BaseEstimator + TransformerMixin` whose `fit_transform` accepts `list[str]`. See [reference/featurizers.md](../reference/featurizers.md).

**Classifier.** Anything implementing `fit(X, y)` and `predict(X)`. Most jurebes classifiers also expose `predict_proba`.

**Pipeline.** A scikit-learn `Pipeline` chaining a featurizer step with a classifier step. Every baseline in jurebes is a `Pipeline`.

**Baseline.** A *named factory* in the `BASELINES` registry producing a fresh `Pipeline`. Factories are zero-argument callables so each cross-validation fold gets a clean estimator.

**Registry.** The mutable name → factory map at `jurebes.baselines.BASELINES`. Supports `names()`, `build(name)`, `register(name, factory)`, `resolve("@group")`.

**Group.** A tag categorising baselines (`linear`, `kernel`, `tree`, `naive_bayes`, `neural`, `reduced_dim`, `online`, `strategy`, `feature_engineering`, `ensemble`, `discriminant`, `categorical`). A baseline can belong to multiple groups.

**Calibration.** A post-hoc transform that maps a classifier's raw decision function to well-calibrated probabilities. jurebes wraps non-probabilistic estimators with `CalibratedClassifierCV(cv=3)` so `predict_proba` is always available. See [theory/calibration.md](../theory/calibration.md).

**Fold.** One iteration of cross-validation: a train/test split drawn from the full dataset.

**Cross-validation (CV).** A protocol that fits and evaluates on `k` non-overlapping folds, then averages the scores. jurebes uses `StratifiedKFold` so class proportions are preserved.

## Tooling terms

**Search.** Hyperparameter optimisation. `jurebes.search.search()` dispatches to six backends. See [reference/cli.md](../reference/cli.md) and the [search subsystem guide](../search.md).

**Search backend.** One of `grid`, `halving_grid`, `random`, `halving_random`, `bayes`, `genetic`. Each is a small module under `jurebes/search/`.

**Benchmark harness.** `jurebes.benchmark` — `train_test`, `cross_validate`, `compare` returning `RunResult` / `ComparisonResult` dataclasses; serialise via `to_markdown` and `to_json`.

**Scoring metric.** A callable in the `SCORERS` registry (`jurebes.benchmark.scoring`). Built-ins: `f1_macro`, `f1_micro`, `accuracy`, `balanced_accuracy`, `log_loss`, `top_k_accuracy`.

**Statistical comparison.** The functions in `jurebes.benchmark.stats`: `paired_t_test_cv`, `wilcoxon_signed_rank_cv`, `mcnemar_test`, `friedman_nemenyi`, `critical_difference`. See [theory/statistical-comparison.md](../theory/statistical-comparison.md).

**Critical difference (CD).** The Nemenyi rank-difference threshold above which two classifiers' mean ranks are considered significantly different. Demšar (2006).

## OVOS terms

**OVOS.** [OpenVoiceOS](https://www.openvoiceos.org/), an open voice assistant platform.

**OPM.** [ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager). jurebes registers an `opm.pipeline` entry point.

**Pipeline plugin.** An OPM `ConfidenceMatcherPipeline` subclass — the OVOS-side adapter exposing high/med/low confidence buckets. `jurebes.opm.JurebesPipeline`.

**Bus.** The OVOS message bus. jurebes listens for `padatious:register_intent`, `padatious:register_entity`, `detach_intent`, `detach_skill`, and `mycroft.ready`.

**IntentHandlerMatch.** The OPM dataclass emitted to skills when a pipeline matches.

## Conceptual map

```
       utterance (str)
            │
            ▼
   ┌──────────────────┐
   │   Featurizer     │   tfidf_word / tfidf_char / hashing / lsa / nmf / lda / autoencoder
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │   Classifier     │   logreg / SVC / NB / RF / MLP / VotingClassifier / …
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │  Calibration     │   optional CalibratedClassifierCV wrap
   └────────┬─────────┘
            ▼
       IntentResult            ◄── + entities from SklearnIOBTagger (optional)
```

The pieces compose: any featurizer with any classifier, optionally wrapped for calibration, optionally paired with a slot tagger. The 48 baselines are pre-built combinations of those pieces.

---
- Previous: [Your first classifier](03-first-classifier.md)
- Next: [Troubleshooting](05-troubleshooting.md)
