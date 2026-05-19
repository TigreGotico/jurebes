# Slot tagging

`SklearnIOBTagger` is a per-token sklearn classifier — by default a `DictVectorizer` + `LogisticRegression` pipeline — trained on token-level IOB tags expanded from `{entity}` placeholders in the intent samples.

## Tokenizer

Regex-based: `re.findall(r"\w+|[^\w\s]", text)`. No nltk, no quebra_frases.

## Token features

`jurebes.slots.token_features(tokens, i)` returns a dict with word, lowercase, prefix/suffix of length 2-3, casing flags, digit flags, BOS/EOS markers, and prev/next words. Override by passing a fully custom sklearn pipeline as `estimator=`.

## Usage

```python
from jurebes import IntentClassifier, BASELINES
from jurebes.slots import SklearnIOBTagger

clf = IntentClassifier(BASELINES.build("logreg"), tagger=SklearnIOBTagger())
clf.add_entity("name", ["bob", "alice", "tom"])
clf.add_intent("name", ["my name is {name}", "call me {name}", "I am {name}"])
clf.add_intent("hello", ["hello", "hi"])
clf.fit()

clf.predict("my name is bob").entities  # -> {"name": "bob"}
```

The tagger and intent classifier share their sample bank; `clf.fit()` trains both. The tagger silently no-ops at prediction time if fewer than two IOB tag classes were seen during training.

## Persistence

`SklearnIOBTagger.save(path)` / `SklearnIOBTagger.load(path)` use joblib. The `IntentClassifier.save/load` round-trip persists the tagger too.
