# Saving and loading

`IntentClassifier` persists via joblib and round-trips losslessly.

## Saving

```python
clf.save("greet_clf.joblib")
```

The on-disk payload is a single joblib pickle containing:

| key | content |
| --- | --- |
| `_jurebes_version` | the jurebes version string at save time |
| `estimator` | the fitted sklearn pipeline |
| `tagger` | the slot tagger instance (or `None`) |
| `samples` | the per-intent training sample bank |
| `entity_samples` | the per-entity training sample bank |
| `fitted` | `True` after a successful `fit()` |

## Loading

```python
from jurebes import IntentClassifier
clf = IntentClassifier.load("greet_clf.joblib")
clf.predict("hi").intent
```

`load` is a classmethod that constructs an `IntentClassifier` instance without running `__init__` (i.e. without re-wrapping with `CalibratedClassifierCV`). The loaded estimator is used as-is. A `_loaded_from_version` attribute is set on the instance so you can diagnose cross-version drift:

```python
clf._loaded_from_version  # e.g. "0.5.2"
```

## File-size expectations

Order of magnitude per baseline family on a small (~500-sample) corpus, joblib-compressed:

| family | typical size |
| --- | --- |
| `nb_*`, `hashing_sgd_*` | tens of kB |
| `linear_svc`, `logreg` | tens to hundreds of kB |
| `reduced_dim` (LSA/NMF/LDA) | hundreds of kB |
| `rbf_svc`, `knn` | hundreds of kB (stores support vectors / training data) |
| `random_forest`, `extra_trees`, `gradient_boosting` | hundreds of kB to MBs |
| `mlp_shallow` | hundreds of kB |
| `voting_soft`, `stacking` | sum of component sizes |

For tighter sizes, joblib supports compression — passing `compress=3` to `joblib.dump` typically halves the file. `IntentClassifier.save` does not currently expose this knob; serialise directly via joblib if you need it:

```python
import joblib
from jurebes.version import __version__ as _jv
joblib.dump({
    "_jurebes_version": _jv,
    "estimator": clf.estimator,
    "tagger": clf.tagger,
    "samples": clf._samples,
    "entity_samples": clf._entity_samples,
    "fitted": clf._fitted,
}, "model.joblib", compress=3)
```

## Security

joblib uses pickle under the hood. **Only load files you trust.** A maliciously crafted joblib payload can execute arbitrary code on load. In particular:

- Do not load model files received from untrusted sources without sandboxing.
- Pin and audit the dependency tree before deploying loaded models to production.
- Prefer transporting the *training data* and re-fitting locally when crossing trust boundaries.

## sklearn version pinning

scikit-learn does not guarantee pickle compatibility across major versions. A model saved with sklearn 1.4 may not load cleanly under sklearn 1.7 if internal class layouts changed. Pin the same `scikit-learn` version in deployment as in training, or retrain on upgrade.

`_jurebes_version` and (if you keep it alongside) the sklearn version make drift diagnosable:

```python
import sklearn, jurebes
print(sklearn.__version__, jurebes.__version__)
```

## Round-trip test

```python
import tempfile
from jurebes import IntentClassifier, BASELINES

clf = IntentClassifier(BASELINES.build("logreg"))
clf.add_intent("a", ["alpha", "alfa"])
clf.add_intent("b", ["beta", "vita"])
clf.fit()
out_before = clf.predict("alpha").intent

with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
    path = f.name
clf.save(path)
clf2 = IntentClassifier.load(path)
assert clf2.predict("alpha").intent == out_before
```

---
- Back to [docs index](../index.md)
