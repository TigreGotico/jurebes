# Slot tagging

Jurebes exposes five pluggable slot-tagging strategies through the
`TAGGERS` registry. They share a common protocol — `add_entity`,
`fit`, `predict`, `tag`, `save`, `load` — and can be swapped under any
`IntentClassifier`.

## Strategy comparison

| strategy        | training | unseen values | typical accuracy | dependency        |
|-----------------|----------|---------------|------------------|-------------------|
| `dictionary`    | none     | no            | high on known    | stdlib only       |
| `template`      | regex compile | yes      | high on matched templates | stdlib only |
| `sklearn_iob`   | sklearn  | yes           | moderate         | scikit-learn      |
| `knn`           | sklearn (NearestNeighbors) | yes | strong on templated patterns | scikit-learn |
| `hybrid`        | per stage | yes          | high             | scikit-learn      |
| `crf`           | sklearn-crfsuite | yes   | highest          | `jurebes[slots-crf]` |

## `DictionaryTagger` — gazetteer regex

```python
from jurebes.slots import DictionaryTagger

t = DictionaryTagger()
t.add_entity("city", ["paris", "lisbon", "new york"])
t.predict("weather in paris")        # {"city": "paris"}
t.predict("weather in new york")     # {"city": "new york"}
```

Single-word entries are word-boundary anchored; multi-token entries are
matched as exact substrings. Case-insensitive by default; flip
`case_sensitive=True` for strict matching. No training required.

## `TemplateTagger` — `{slot}` regex templates

```python
from jurebes.slots import TemplateTagger

t = TemplateTagger()
t.add_intent("weather", ["(weather|temperature) in {city}"])
t.fit()
t.predict("temperature in berlin")   # {"city": "berlin"}
```

`{slot}` placeholders become named regex groups; `(a|b)` becomes a
grouped alternation. The first matching template wins
(longest-template-first for determinism).

## `SklearnIOBTagger` — per-token sklearn classifier

Default pipeline: `DictVectorizer` + `LogisticRegression`. Trains on
token-level IOB tags expanded from `{entity}` placeholders in the
intent samples.

```python
from jurebes.slots import SklearnIOBTagger

t = SklearnIOBTagger()
t.add_entity("name", ["bob", "alice", "tom"])
t.fit({"name": ["my name is {name}", "call me {name}"]})
t.predict("call me bob")             # {"name": "bob"}
```

### Tokenizer

Regex: `re.findall(r"\w+|[^\w\s]", text)`. No nltk.

### Token features

`jurebes.slots.token_features(tokens, i)` returns a dict with: `word`,
`lower`, `suffix2/3`, `prefix2/3`, `is_upper`, `is_title`,
`is_digit`, `has_digit`, `bos`, `eos`, `prev_word`, `next_word`.

## `HybridCascadeTagger` — dict → template → IOB cascade

```python
from jurebes.slots import HybridCascadeTagger

h = HybridCascadeTagger()
h.add_entity("city", ["paris"])
h.add_intent("weather", ["weather in {city}"])
h.fit()
h.predict("weather in paris")        # {"city": "paris"}
h.predict("weather in berlin")       # {"city": "berlin"} via template fallback
```

Constituent taggers run in order; earlier taggers win on key collisions.
Pass a custom list via `HybridCascadeTagger(taggers=[...])`.

## `KNNTagger` — nearest-utterance tag transfer

```python
from jurebes.slots import KNNTagger

t = KNNTagger(k=3)
t.add_entity("city", ["lisbon", "paris", "berlin"])
t.fit({"weather": ["weather in {city}", "forecast for {city}"]})
t.predict("weather in tokyo")        # {"city": "tokyo"}
```

Training utterances are TF-IDF-vectorised (char 3-5 n-grams by default);
at predict time the input is matched against the `k` nearest neighbours
via `sklearn.neighbors.NearestNeighbors`, and their IOB tags are
majority-voted onto the input by token position. The pattern transfers
to unseen entity values whenever the surrounding context matches a
trained template — `tokyo` above was never registered as a `city`.

Tune the surface vectoriser by passing your own `vectorizer=` argument.

## `CRFTagger` — optional sklearn-crfsuite

Install the extra:

```bash
pip install jurebes[slots-crf]
```

```python
from jurebes.slots.crf import CRFTagger

t = CRFTagger()
t.add_entity("name", ["bob", "alice"])
t.fit({"name": ["my name is {name}", "call me {name}"]})
```

Same feature dictionaries as `SklearnIOBTagger`; under the hood trains
`sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100)`.

## Registry

```python
from jurebes.slots import TAGGERS

TAGGERS.names()              # ['dictionary', 'template', 'sklearn_iob', 'knn', 'hybrid', 'crf']
TAGGERS.build("hybrid")      # HybridCascadeTagger instance
```

`IntentClassifier` accepts a registry name directly:

```python
from jurebes import IntentClassifier

clf = IntentClassifier(tagger="dictionary")
```

## When to use which

- **`dictionary`** — closed-set slots (timezones, ISO codes, known
  device names). Zero training cost.
- **`template`** — small command grammars with predictable phrasings.
  Handles unseen slot values cleanly.
- **`sklearn_iob`** — broader coverage with moderate training data;
  generalises beyond seen templates.
- **`hybrid`** — production default. Dictionary catches the easy
  cases; template handles known phrasings; IOB picks up the rest.
- **`crf`** — when sequence structure matters and the extra dependency
  is acceptable.

## Persistence

All taggers expose `save(path)` / `load(path)`. `DictionaryTagger`
and `TemplateTagger` persist as JSON; the sklearn-backed taggers and
`HybridCascadeTagger` use joblib.

## Benchmark harness

`jurebes.benchmark.slots.compare_taggers` evaluates a set of taggers on
a labelled list of `(utterance, gold_slots)` pairs and returns a
`SlotComparisonResult` with markdown rendering.

```python
from jurebes.benchmark.slots import compare_taggers

result = compare_taggers(
    ["dictionary", "template", "sklearn_iob", "hybrid"],
    intent_samples={"weather": ["weather in {city}"], "greet": ["hi"]},
    entity_samples={"city": ["paris", "berlin"]},
    test_utterances=[("weather in paris", {"city": "paris"}), ("hi", {})],
)
print(result.to_markdown())
```

---
[← back to docs index](index.md)
