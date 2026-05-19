# Slot tagging

`SklearnIOBTagger` is a per-token sklearn classifier — by default a `DictVectorizer` + `LogisticRegression` pipeline — trained on token-level IOB tags expanded from `{entity}` placeholders in the intent samples.

## Tokenizer

Regex-based: `re.findall(r"\w+|[^\w\s]", text)`. No nltk, no quebra_frases.

## Token features

`jurebes.slots.token_features(tokens, i)` returns a dict with the following keys:

| key | type | meaning |
|---|---|---|
| `word` | str | the raw token |
| `lower` | str | `tok.lower()` |
| `suffix2` | str | last 2 chars (lowercased) |
| `suffix3` | str | last 3 chars (lowercased) |
| `prefix2` | str | first 2 chars (lowercased) |
| `prefix3` | str | first 3 chars (lowercased) |
| `is_upper` | bool | `tok.isupper()` |
| `is_title` | bool | `tok.istitle()` |
| `is_digit` | bool | `tok.isdigit()` |
| `has_digit` | bool | any character is a digit |
| `bos` | bool | `i == 0` |
| `eos` | bool | last token |
| `prev_word` | str | previous token lowercased, or `<BOS>` |
| `next_word` | str | next token lowercased, or `<EOS>` |

Example — `token_features(["call", "me", "Bob"], 2)`:

```python
{
    "word": "Bob", "lower": "bob",
    "suffix2": "ob", "suffix3": "bob",
    "prefix2": "bo", "prefix3": "bob",
    "is_upper": False, "is_title": True,
    "is_digit": False, "has_digit": False,
    "bos": False, "eos": True,
    "prev_word": "me", "next_word": "<EOS>",
}
```

Override by passing a fully custom sklearn pipeline as `estimator=`.

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

---
[← back to docs index](index.md)
