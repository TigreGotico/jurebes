# Architecture

The module graph and dataflow inside jurebes.

## Module graph

```
   ┌───────────────────────────────────────────────────────────────┐
   │                            jurebes                            │
   │                                                               │
   │   core (IntentClassifier, IntentResult)                       │
   │     ▲                                                         │
   │     │                                                         │
   │   baselines (BASELINES registry, factories)                   │
   │     ▲                                                         │
   │     │                                                         │
   │   featurizers (TF-IDF, hashing, LSA, NMF, LDA, autoencoder,   │
   │                text_stats, categorical)                       │
   │                                                               │
   │   slots (SklearnIOBTagger, token_features, tokenize)          │
   │                                                               │
   │   benchmark                                                   │
   │     ├── harness (train_test, cross_validate, compare)         │
   │     ├── metrics (RunResult, pooled_percentiles)               │
   │     ├── scoring (SCORERS registry)                            │
   │     ├── report  (to_markdown, to_json)                        │
   │     └── stats   (paired_t, wilcoxon, mcnemar, friedman_nemenyi│
   │                  critical_difference)                         │
   │                                                               │
   │   search                                                      │
   │     ├── api (search(), SearchResult)                          │
   │     ├── spaces (for_baseline)                                 │
   │     ├── grid / random / bayes / genetic backends              │
   │                                                               │
   │   datasets (csv, jsonl, ovos, huggingface, canonical/*)       │
   │                                                               │
   │   opm (JurebesPipeline — ConfidenceMatcherPipeline)           │
   │                                                               │
   │   cli (argparse entry point — uses every public module)       │
   └───────────────────────────────────────────────────────────────┘
```

## Dataflow at fit

```
training utterances X ──┐
training labels y ──────┤
                        │
                        ▼
              IntentClassifier.add_intent(name, samples)
                        │
                        │  (per-intent sample bank: dict[str, list[str]])
                        ▼
              IntentClassifier.fit()
                        │
                        ▼
                ┌───────────────┐
                │    Pipeline   │
                │  ┌─────────┐  │
                │  │  feat   │  │   featurizer.fit(X).transform(X)
                │  └────┬────┘  │
                │       ▼       │
                │  ┌─────────┐  │
                │  │   clf   │  │   classifier.fit(feat(X), y)
                │  └─────────┘  │
                └───────────────┘
                        │
                        ▼  (optional)
                ┌───────────────┐
                │ SklearnIOB    │   IOB-expanded samples
                │ Tagger.fit    │   token_features per token
                └───────────────┘
```

## Dataflow at predict

```
utterance ───┐
             ▼
   IntentClassifier.predict(utt)
             │
             ├─► estimator.predict_proba([utt])   →  probabilities per class
             │       │
             │       └─► sort, take top
             │
             ├─► tagger.predict(utt)              →  {entity: span}
             │
             └─► IntentResult(intent, confidence, entities, utterance)
```

## State

- `IntentClassifier._samples: Dict[str, List[str]]` — per-intent training sample bank.
- `IntentClassifier._entity_samples: Dict[str, List[str]]` — per-entity training sample bank.
- `IntentClassifier.estimator` — the (possibly calibrated) sklearn pipeline.
- `IntentClassifier.tagger` — the `SklearnIOBTagger` instance or `None`.
- `IntentClassifier._fitted: bool` — guards `predict`/`predict_proba`.
- `IntentClassifier._lock: RLock` — protects mutation during multi-threaded use.

After `save(path)` all of the above are pickled to a single joblib payload along with `_jurebes_version`.

## Extension points

| extension | how |
| --- | --- |
| add a featurizer | implement `sklearn.base.{BaseEstimator, TransformerMixin}` — see [custom-featurizers.md](custom-featurizers.md) |
| add a baseline | `BASELINES.register(name, factory, group=...)` — see [custom-baselines.md](custom-baselines.md) |
| add a search backend | new module under `jurebes/search/` + dispatch in `api.py` — see [custom-search-backends.md](custom-search-backends.md) |
| add a scoring metric | `SCORERS[name] = callable` — see [custom-scoring-metrics.md](custom-scoring-metrics.md) |
| add a dataset loader | `(path, **kwargs) -> tuple[list[str], list[str]]` — see [custom-datasets.md](custom-datasets.md) |

## Dependency policy

- **Hard dependencies** (always installed): scikit-learn, joblib, ovos-utils, ovos-bus-client, ovos-config, ovos-plugin-manager, langcodes.
- **Optional dependencies** (extras): `datasets` (HF), `scikit-optimize` (Bayes search), `sklearn-genetic-opt` (genetic search), `matplotlib` (CD plots), `ovoscope` + `ovos-core` (e2e tests).
- The core module never imports from an optional dependency at module-import time. Optional imports happen *inside* the function that needs them, with a clear `ImportError("install jurebes[<extra>] to use ...")` message.

## Threading model

`IntentClassifier` is thread-safe for read operations after `fit()`. Mutating operations (`add_intent`, `add_entity`, `fit`, `remove_*`) are guarded by `self._lock`.

The OVOS plugin instantiates one `IntentClassifier` per language; bus events from different threads serialise through the lock.

## CLI architecture

`jurebes.cli.main(argv=None)` constructs an argparse tree with five subcommands (`list-baselines`, `benchmark`, `train`, `predict`, `search`, `stats`). Each subcommand resolves its arguments and calls into the public Python API. See [reference/cli.md](../reference/cli.md).

## Versioning

`jurebes.version.__version__` is the single source of truth. The semver is bumped automatically by CI based on conventional commit prefixes; the file is not edited by hand. `IntentClassifier.save` stamps the current version into the payload as `_jurebes_version` for cross-version drift diagnostics.

---
- Back to [docs index](../index.md)
