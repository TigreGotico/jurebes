# Reference — slot tagging

All taggers share the same protocol:

```python
class Tagger:
    fitted: bool
    def add_entity(name: str, samples: list[str]) -> None: ...
    def remove_entity(name: str) -> None: ...
    def fit(intent_samples: dict[str, list[str]] | None = None) -> Self: ...
    def predict(utterance: str) -> dict[str, str]: ...
    def tag(text: str) -> list[tuple[str, str]]: ...  # BIO tokens
    def save(path) -> None: ...
    @classmethod
    def load(cls, path) -> Self: ...
```

## `jurebes.slots.TAGGERS`

The registry maps strategy names to factory callables.

```python
from jurebes.slots import TAGGERS
TAGGERS.names()                  # list of registered names
TAGGERS.build("dictionary")      # construct an instance
TAGGERS.groups()                 # group → {names}
"hybrid" in TAGGERS              # True
len(TAGGERS)                     # 5
```

Built-in entries:

| name           | class                  | group       | requires            |
|----------------|------------------------|-------------|---------------------|
| `dictionary`   | `DictionaryTagger`     | rule_based  | stdlib              |
| `template`     | `TemplateTagger`       | rule_based  | stdlib              |
| `sklearn_iob`  | `SklearnIOBTagger`     | ml          | scikit-learn        |
| `hybrid`       | `HybridCascadeTagger`  | hybrid      | scikit-learn        |
| `crf`          | `CRFTagger`            | ml          | `jurebes[slots-crf]` |

## `DictionaryTagger(case_sensitive=False)`

Gazetteer regex tagger. Always `fitted=True`. `fit()` is a no-op
that recompiles all patterns; accepts `intent_samples` for symmetry.

## `TemplateTagger()`

Regex-template tagger. `add_intent(name, samples)` stores templates;
`fit()` compiles them. Templates use `{slot}` for named groups and
`(a|b)` for alternations.

## `SklearnIOBTagger(estimator=None)`

Per-token IOB classifier. Defaults to
`DictVectorizer + LogisticRegression(max_iter=1000)`. Override
`estimator=` with any sklearn classifier pipeline.

## `HybridCascadeTagger(taggers=None)`

Cascade of constituent taggers (default:
`[DictionaryTagger(), TemplateTagger(), SklearnIOBTagger()]`).
`add_entity`, `add_intent`, `fit` fan out to every constituent.
`predict` merges results in order — earlier taggers win on collisions.

## `CRFTagger()`

`sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100,
all_possible_transitions=True)`. Same feature dicts as
`SklearnIOBTagger`. Raises `ImportError("install jurebes[slots-crf]
to use the CRF tagger")` if the dependency is missing.

## `jurebes.benchmark.slots.compare_taggers`

```python
compare_taggers(
    tagger_names: list[str],
    intent_samples: dict[str, list[str]],
    entity_samples: dict[str, list[str]],
    test_utterances: list[tuple[str, dict[str, str]]],
    *,
    scoring=("slot_f1", "exact_match"),
) -> SlotComparisonResult
```

Returns a dataclass with `rows: list[SlotRunResult]` and a
`to_markdown()` method. Each row carries `slot_precision`,
`slot_recall`, `slot_f1`, `exact_match`, and `n_test`.

---
[← back to reference index](index.md)
