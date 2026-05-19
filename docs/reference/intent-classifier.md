# `IntentClassifier` and `IntentResult`

Module: `jurebes.core`. Both names are re-exported from the package root: `from jurebes import IntentClassifier, IntentResult`.

## `IntentResult`

```python
@dataclass
class IntentResult:
    intent: Optional[str]
    confidence: float
    entities: Dict[str, str] = field(default_factory=dict)
    utterance: str = ""
```

Returned by `IntentClassifier.predict` (single) and `IntentClassifier.predict_proba` (list, sorted by descending confidence).

| field | type | meaning |
| --- | --- | --- |
| `intent` | `Optional[str]` | predicted intent name, or `None` if the classifier has nothing to predict |
| `confidence` | `float` | probability in `[0, 1]` when the underlying estimator is calibrated; `1.0` for non-probabilistic estimators |
| `entities` | `Dict[str, str]` | extracted slots from the attached `SklearnIOBTagger` (empty when no tagger) |
| `utterance` | `str` | the original input utterance |

## `IntentClassifier`

```python
class IntentClassifier:
    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        *,
        tagger: Optional[Any] = None,
        calibrate: Union[Literal["if_missing", "always"], bool] = "if_missing",
    ): ...
```

### Constructor arguments

| arg | type | default | meaning |
| --- | --- | --- | --- |
| `estimator` | `BaseEstimator` or `None` | `None` → `BASELINES.build("linear_svc")` | any sklearn-compatible estimator, typically a `Pipeline` |
| `tagger` | object with `add_entity`/`fit`/`predict` or `None` | `None` | slot tagger; almost always a `SklearnIOBTagger` |
| `calibrate` | `"if_missing"` \| `"always"` \| `False` \| `True` | `"if_missing"` | calibration mode (see below) |

### Calibration modes

- `"if_missing"` (also `True`): wrap with `CalibratedClassifierCV(cv=3)` only if `estimator` lacks `predict_proba`.
- `"always"`: always wrap, even when `predict_proba` exists. Useful for tree models whose native probabilities are uncalibrated.
- `False`: never wrap. Raises `ValueError` at construction time if the estimator lacks `predict_proba`.

### Methods

#### `add_intent(name: str, samples: List[str]) -> None`

Append samples for an intent. Calling multiple times for the same `name` extends the sample bank.

#### `add_entity(name: str, samples: List[str]) -> None`

Append samples for an entity. Raises `ValueError` if no tagger is configured.

#### `remove_intent(name: str) -> None`

Drop an intent and its samples. No-op when the name is unknown.

#### `remove_entity(name: str) -> None`

Drop an entity from both the sample bank and the tagger.

#### `fit() -> IntentClassifier`

Fit the estimator on the accumulated samples. Also fits the tagger when present. Raises `ValueError` when fewer than 2 intent classes are registered. Returns `self` for chaining.

#### `predict(utterance: str) -> IntentResult`

Predict the top intent. Falls back to `estimator.predict([utt])[0]` with `confidence=1.0` when `predict_proba` is unavailable.

#### `predict_proba(utterance: str) -> List[IntentResult]`

Return one `IntentResult` per class, sorted by descending confidence. Raises `AttributeError` / `NotImplementedError` when the estimator has no `predict_proba`.

#### `predict_batch(utterances: List[str]) -> List[IntentResult]`

Vectorised prediction over a list of utterances. Single forward pass through `predict_proba` when available; otherwise calls `predict` once on the batch.

#### `save(path: str | Path) -> None`

Pickle the estimator, tagger, sample banks, fitted flag, and `_jurebes_version` to a joblib file.

#### `load(path: str | Path) -> IntentClassifier` (classmethod)

Restore an `IntentClassifier` from a joblib file. The loaded instance has `_loaded_from_version` set to the version string stamped at save time.

### Joblib payload schema

```python
{
    "_jurebes_version": "<version-string>",
    "estimator":        <fitted sklearn estimator>,
    "tagger":           <SklearnIOBTagger or None>,
    "samples":          {intent_name: [sample, ...]},
    "entity_samples":   {entity_name: [sample, ...]},
    "fitted":           True,
}
```

## Minimal end-to-end example

```python
from jurebes import IntentClassifier, BASELINES

clf = IntentClassifier(BASELINES.build("logreg"))
clf.add_intent("hello", ["hello", "hi", "hey there"])
clf.add_intent("joke",  ["tell me a joke", "say a joke"])
clf.fit()

print(clf.predict("hey").intent)         # -> "hello"
for r in clf.predict_proba("hello"):
    print(r.intent, r.confidence)

clf.save("model.joblib")
clf2 = IntentClassifier.load("model.joblib")
```

## Thread safety

All mutating methods (`add_intent`, `add_entity`, `remove_*`, `fit`) acquire an internal `RLock`. Concurrent `predict` / `predict_proba` calls are safe after `fit()` and do not take the lock.

---
- Back to [reference index](index.md)
