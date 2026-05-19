# jurebes

OVOS intent pipeline plugin built on top of **JurebesIntentContainer** — an
ensemble intent parser that combines:

- **padacioso** for fast exact / regex template matches and entity capture
- **scikit-learn classifiers** (SVC, LogisticRegression, DecisionTree, …) via
  `ovos-classifiers`, with optional soft-voting ensembles
- **entity taggers** (sklearn or nltk n-gram) for IOB-style slot extraction

## Install

```bash
pip install jurebes
```

## OPM entry point

```
opm.pipeline:
  ovos-jurebes-pipeline-plugin = jurebes.opm:JurebesPipeline
```

`JurebesPipeline` implements `ConfidenceMatcherPipeline` and exposes the usual
`match_high`, `match_medium`, `match_low` matchers.

## Configuration

```json
{
  "jurebes": {
    "conf_high": 0.8,
    "conf_med": 0.6,
    "conf_low": 0.4,
    "fuzzy": false
  }
}
```

| key         | default | meaning                                              |
| ----------- | ------- | ---------------------------------------------------- |
| `conf_high` | 0.8     | threshold for `match_high`                           |
| `conf_med`  | 0.6     | threshold for `match_medium`                         |
| `conf_low`  | 0.4     | threshold for `match_low`                            |
| `fuzzy`     | false   | enable fuzzy entity matching in padacioso            |

Secondary languages (`core_config.secondary_langs`) get independent containers.

## Programmatic use

```python
from jurebes import JurebesIntentContainer

engine = JurebesIntentContainer()
engine.add_entity("name", ["jarbas", "bob"])
engine.add_intent("name", ["my name is {name}", "call me {name}"])
engine.add_intent("hello", ["hello", "hi", "hey"])
engine.add_intent("joke", ["tell me a joke", "say a joke"])
engine.train()

print(engine.calc_intent("my name is jarbas"))
# IntentMatch(intent_name='name', confidence=..., entities={'name': 'jarbas'})
```

## Pipeline architecture

Reacts to standard OVOS messagebus events:

- `padatious:register_intent` / `padatious:register_entity` — feed samples into
  the per-language `JurebesIntentContainer`.
- `detach_intent` / `detach_skill` — remove intents on skill unload.
- `mycroft.ready` — trigger ensemble training on all languages.

Predictions go through padacioso first (cheap, exact-match) then through the
sklearn classifier; padacioso hits boost the classifier probability. Entity
slots from regex captures and from the IOB tagger are merged.

## See also

- [Source for `JurebesIntentContainer`](../jurebes/__init__.py)
- [OPM wrapper](../jurebes/opm.py)
