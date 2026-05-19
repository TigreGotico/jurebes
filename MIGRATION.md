# Migration guide

This page maps the prior `JurebesIntentContainer` surface onto the current `IntentClassifier` API.

## Class rename

`jurebes.JurebesIntentContainer` is now `jurebes.IntentClassifier`. Construct it with a sklearn estimator (or a `BASELINES.build(name)` result) and, optionally, a `SklearnIOBTagger`.

```python
from jurebes import IntentClassifier, BASELINES
clf = IntentClassifier(BASELINES.build("linear_svc"))
```

## Method mapping

| old | new |
| --- | --- |
| `add_intent(name, samples)` | `add_intent(name, samples)` — unchanged |
| `add_entity(name, samples)` | `add_entity(name, samples)` — requires a `tagger=` on the classifier |
| `calc_intent(utt)` | `predict(utt)` — returns a single `IntentMatch` |
| `calc_intents(utt)` | `predict_proba(utt)` — returns ranked candidates |
| `enable_fuzzy()` / `disable_fuzzy()` | removed — pick a char-ngram baseline (`logreg_char`, `linear_svc_char`, `union_logreg`) for typo tolerance |
| `detach_intent()` / `reatach_intent()` | removed |
| `set_context()` / `require_context()` / `exclude_context()` / `exclude_keywords()` | removed — handled by OVOS `SessionManager` |

## Slot extraction

Template-with-slot matching is provided exclusively by the trained `SklearnIOBTagger`. Pass a tagger to the classifier and call `add_entity(...)` plus `add_intent("name", ["my name is {name}"])`; slots are returned in the `entities` field of the prediction result.

## Required dependencies

The following dependencies are not used: `ovos-classifiers`, `nltk`, `padacioso`, `quebra-frases`. Only `scikit-learn`, `numpy`, `joblib`, and a minimal OVOS runtime are required.

## Entry-point compatibility

The OVOS pipeline entry point name is unchanged:

```
ovos-jurebes-pipeline-plugin = jurebes.opm:JurebesPipeline
```

Existing `mycroft.conf` pipeline lists keep working; see [`docs/opm.md`](docs/opm.md) for the config block.
