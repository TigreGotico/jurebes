# Troubleshooting

Common errors when first using jurebes, with diagnostic paths and fixes.

## `ValueError: n_components must be <= n_features`

**Cause.** A reduced-dim baseline (`lsa_*`, `nmf_*`, `lda_*`, `autoencoder_*`) requested more latent components than the TF-IDF vocabulary supplied — typical on very small training sets.

**Fix.**

- Increase the corpus.
- Or build a smaller-dim pipeline by hand:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from jurebes import IntentClassifier
from jurebes.featurizers import lsa

clf = IntentClassifier(Pipeline([("feat", lsa(8)), ("clf", LogisticRegression(max_iter=1000))]))
```

## `ValueError: n_splits=3 cannot be greater than the number of members in each class`

**Cause.** `CalibratedClassifierCV` defaults to `cv=3` and requires at least 3 samples per class.

**Fix.** Add more samples (recommended) or skip calibration if the underlying estimator already exposes `predict_proba`:

```python
from sklearn.linear_model import LogisticRegression
from jurebes import IntentClassifier
clf = IntentClassifier(LogisticRegression(max_iter=1000), calibrate=False)
```

## `AttributeError: 'LinearSVC' object has no attribute 'predict_proba'`

**Cause.** A non-probabilistic estimator was passed with `calibrate=False`.

**Fix.** Use the default `calibrate="if_missing"` (jurebes auto-wraps), or pass `calibrate="always"`:

```python
from sklearn.svm import LinearSVC
from jurebes import IntentClassifier
clf = IntentClassifier(LinearSVC(), calibrate="if_missing")   # default
clf = IntentClassifier(LinearSVC(), calibrate="always")       # also fine
```

`IntentClassifier(LinearSVC(), calibrate=False)` raises immediately on construction with a clear message rather than at `predict` time.

## `ImportError: install jurebes[hf] to use the HF loader`

**Cause.** A canonical loader (`load_snips`, `load_clinc`, etc.) or `load_hf` was called without the `datasets` dependency.

**Fix.**

```bash
pip install jurebes[hf]
```

The first call downloads from HuggingFace and caches under `~/.cache/huggingface/`. Subsequent calls work offline.

```bash
HF_DATASETS_OFFLINE=1 python my_script.py
```

forces offline mode after the cache is warm.

## `KeyError: 'unknown baseline: foo'`

**Cause.** Typo in a baseline name, or attempting to use a baseline that has not been registered.

**Fix.** List available baselines:

```bash
jurebes list-baselines
```

or programmatically:

```python
from jurebes import BASELINES
print(sorted(BASELINES.names()))
```

## `KeyError: 'unknown group: foo'`

**Cause.** Used `@<group>` selector with an unrecognised group name.

**Fix.** List groups:

```python
from jurebes import BASELINES
print(sorted(BASELINES.groups()))
```

Known group names: `naive_bayes`, `linear`, `kernel`, `tree`, `neural`, `ensemble`, `reduced_dim`, `online`, `strategy`, `feature_engineering`, `discriminant`, `categorical`. Plus the special `@all` selector.

## `ValueError: no tagger configured; pass tagger=SklearnIOBTagger(...)`

**Cause.** Called `clf.add_entity(...)` on a classifier built without a tagger.

**Fix.**

```python
from jurebes import IntentClassifier, BASELINES
from jurebes.slots import SklearnIOBTagger

clf = IntentClassifier(BASELINES.build("logreg"), tagger=SklearnIOBTagger())
clf.add_entity("name", ["bob", "alice"])
```

## `RuntimeError: classifier not fitted`

**Cause.** Called `predict()` or `predict_proba()` before `fit()`.

**Fix.** Call `clf.fit()` after adding all intents and entities.

## `ValueError: need at least 2 intent classes to fit`

**Cause.** Only one intent was registered.

**Fix.** Intent classification is multi-class by definition; add at least a second intent before calling `fit()`.

## End-to-end tests skip silently

**Cause.** Optional `e2e` extra not installed.

**Fix.**

```bash
pip install jurebes[e2e]
pytest test/end2end -q
```

The `e2e` extra installs `ovoscope`, `ovos-core`, `ovos-skill-hello-world`, `pytest`, and `pytest-timeout`.

## Tests print `RuntimeWarning: divide by zero` from `f1_score`

**Cause.** A baseline produced zero predictions for some class in a fold; sklearn computes 0/0 for that class's F1.

**Fix.** This is benign noise from the underlying metric on tiny folds. The harness uses `zero_division=0` for the macro/micro reductions, so the reported numbers are well-defined. Suppress via `warnings.filterwarnings("ignore", category=RuntimeWarning)` in your harness script if needed.

## Loaded model raises `ModuleNotFoundError` on `joblib.load`

**Cause.** The model was saved with a scikit-learn version that exposed an internal module renamed in your installed version.

**Fix.** Pin the same scikit-learn major version in production as the one used at training time, or retrain. The `_jurebes_version` tag stored alongside the model helps diagnose drift:

```python
import joblib
print(joblib.load("model.joblib").get("_jurebes_version"))
```

---
- Previous: [Core concepts](04-core-concepts.md)
- Back to [docs index](../index.md)
