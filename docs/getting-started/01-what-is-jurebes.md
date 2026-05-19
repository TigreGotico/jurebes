# What is jurebes?

## Intent classification in one paragraph

Intent classification is the task of mapping a short natural-language utterance — typically a single spoken or typed sentence — to one label from a fixed inventory of *intents*. "play some jazz", "put on some jazz", and "I want jazz music" all map to the intent `play_music`. The output is a single label, optionally paired with extracted *slots* (named arguments like a song title or artist). The problem sits at the entry point of voice assistants, chatbots, and command-line natural-language interfaces.

## What jurebes does

jurebes is a research framework for classical-ML approaches to intent classification. It wires any scikit-learn featurizer with any scikit-learn classifier behind a small intent API and ships:

- A registry of 48 named baselines covering naive Bayes, linear models, kernel SVMs, tree ensembles, neural-bottleneck reduced-dim pipelines, multi-class strategy wrappers, discriminant analysis, voting, stacking, and dict-input categorical pipelines.
- Six canonical dataset loaders (SNIPS, CLINC150, BANKING77, HWU64, ATIS, MASSIVE).
- A benchmark harness with multi-metric scoring and pooled tail-latency percentiles.
- A hyperparameter search subsystem with six backends (grid, halving-grid, random, halving-random, Bayesian, genetic).
- A statistical comparison module following Demšar (2006): paired t-test, Wilcoxon signed-rank, McNemar, Friedman with post-hoc Nemenyi, critical-difference diagrams.
- A `SklearnIOBTagger` slot extractor.
- A CLI: `list-baselines`, `benchmark`, `train`, `predict`, `search`, `stats`.
- An OVOS `ConfidenceMatcherPipeline` plugin.

## What jurebes does *not* do

- No deep learning. No transformers, no embeddings beyond scikit-learn's bag-of-words and topic-model derivatives. Use [ovos-padatious](https://github.com/OpenVoiceOS/padatious) or a transformer-based pipeline if you need that.
- No production serving infrastructure beyond the OVOS plugin adapter. There is no REST server, no gRPC interface, no model registry, no A/B framework.
- No data labelling, no active learning, no synthetic-data generation.

## Who should use jurebes

- **Researchers** comparing classical-ML baselines on small intent corpora.
- **OVOS plugin authors** prototyping intent matchers that run on CPU with bounded memory.
- **Anyone benchmarking** small-corpus text classifiers and needing a reproducible harness with proper statistical tests.

## Elevator pitch

```python
from jurebes import IntentClassifier, BASELINES

clf = IntentClassifier(BASELINES.build("logreg"))
clf.add_intent("hello", ["hello", "hi", "hey there"])
clf.add_intent("joke",  ["tell me a joke", "say a joke", "make me laugh"])
clf.fit()
print(clf.predict("hi there").intent)  # -> "hello"
```

That is the entire API surface for the simplest end-to-end use. The rest of this documentation tree extends from there in two directions: deeper API capabilities, and the underlying classical-ML theory.

---
- Next: [Installation](02-installation.md)
- Back to [docs index](../index.md)
