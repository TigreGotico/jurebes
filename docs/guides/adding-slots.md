# Adding slots

Slot extraction recovers named arguments from an utterance — the "Miles Davis" in "play some Miles Davis". jurebes ships a pure-sklearn IOB tagger that trains on `{entity}` placeholders inside intent samples.

## When slots help

- The intent is "do X with argument Y" and the skill needs Y.
- The argument space is sparse and not enumerable as a separate intent per value.
- You want to keep the intent inventory small even when many arguments are possible.

## When slots do not help

- The argument set is tiny and discrete. Treat each as its own intent.
- The argument is the entire utterance (e.g. a search query). Use `clf.predict(utt).utterance` and parse downstream.

## Minimal wiring

```python
from jurebes import IntentClassifier, BASELINES
from jurebes.slots import SklearnIOBTagger

clf = IntentClassifier(BASELINES.build("logreg"), tagger=SklearnIOBTagger())

clf.add_entity("name", ["bob", "alice", "tom", "anna"])
clf.add_intent("introduce", [
    "my name is {name}",
    "call me {name}",
    "I am {name}",
    "I'm {name}",
])
clf.add_intent("hello", ["hello", "hi", "hey there"])

clf.fit()
print(clf.predict("my name is bob").entities)   # {'name': 'bob'}
print(clf.predict("hello").entities)            # {}
```

## Sample format

- `{entity_name}` placeholders are expanded against the registered entity samples.
- A sample without placeholders contributes only to the intent classifier.
- An entity name is a string. Convention: lowercase, no spaces. Skill-namespaced names (`skill_id:entity_name`) are accepted and used as-is.

## Multiple slots per sample

```python
clf.add_entity("artist", ["miles davis", "coltrane"])
clf.add_entity("genre",  ["jazz", "rock"])
clf.add_intent("play", [
    "play some {genre}",
    "play {artist}",
    "play some {genre} from {artist}",
])
```

Slot spans can be multi-token; the IOB scheme tags `B-artist` on the first token and `I-artist` on each continuation token.

## What the tagger does internally

`SklearnIOBTagger`:

1. Tokenises each sample with the regex `\w+|[^\w\s]`.
2. Expands `{entity}` placeholders against the registered entity samples (one expansion per entity sample).
3. Assigns IOB tags (`B-<entity>`, `I-<entity>`, `O`) per token.
4. Extracts dense per-token features via `jurebes.slots.token_features` — word, suffix2/3, prefix2/3, casing flags, digit flags, BOS/EOS, previous/next words.
5. Fits a `DictVectorizer + LogisticRegression` pipeline on the token-feature → IOB-tag mapping.

Override the estimator by passing `SklearnIOBTagger(estimator=my_pipeline)`. See [docs/slots.md](../slots.md) for the feature-key table and the override pattern.

## Testing slot quality

A quick eyeball check:

```python
for utt, expected in [
    ("my name is bob", {"name": "bob"}),
    ("call me alice", {"name": "alice"}),
    ("hello there", {}),
]:
    got = clf.predict(utt).entities
    print("ok" if got == expected else "fail", utt, got)
```

For systematic evaluation, hold out a slot-annotated test set and compare predicted vs gold spans per utterance. The tagger silently returns an empty dict when it has not been fitted on at least two distinct IOB tag classes.

## Saving and loading

`IntentClassifier.save(path)` persists both the intent classifier and the attached tagger. `IntentClassifier.load(path)` restores both. The tagger also has its own `save`/`load` if you want to ship it independently.

## OVOS integration

In the OVOS pipeline plugin, slots are enabled by default and controlled by the `enable_slots` config key:

```json
{ "jurebes": { "enable_slots": true } }
```

The plugin auto-attaches an `SklearnIOBTagger` per language. See [advanced/ovos-integration-deep-dive.md](../advanced/ovos-integration-deep-dive.md).

---
- Back to [docs index](../index.md)
