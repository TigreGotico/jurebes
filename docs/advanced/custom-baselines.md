# Custom baselines

A baseline in jurebes is a *zero-argument factory* returning a fresh sklearn `Pipeline`. Factories are called once per fold, ensuring every CV iteration starts with a clean estimator.

## Registering a baseline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from jurebes.baselines import BASELINES

BASELINES.register(
    "my_logreg",
    lambda: Pipeline([
        ("feat", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf",  LogisticRegression(C=4.0, max_iter=2000)),
    ]),
    group="linear",
)
```

Now `BASELINES.build("my_logreg")` returns a freshly-constructed pipeline and `BASELINES.resolve("@linear")` includes it.

## Calibration wrap pattern

Non-probabilistic estimators (`LinearSVC`, `RidgeClassifier`, hinge-loss `SGDClassifier`, `Perceptron`, `PassiveAggressiveClassifier`) lack `predict_proba`. To match the rest of the registry's behaviour, wrap with `CalibratedClassifierCV(cv=3)`:

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

def _cal(est):
    return CalibratedClassifierCV(est, cv=3)

BASELINES.register(
    "my_svc",
    lambda: Pipeline([
        ("feat", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf",  _cal(LinearSVC())),
    ]),
    group="linear",
)
```

Alternatively, build the baseline without the wrap and let `IntentClassifier(estimator, calibrate="if_missing")` add it at construction time. The registry convention is to wrap at the factory level so `compare()` reports include the calibration cost.

## Adding to multiple groups

```python
BASELINES.register("my_baseline", _factory, group="linear")
BASELINES.add_to_group("ensemble", "my_baseline")
BASELINES.add_to_group("feature_engineering", "my_baseline")
```

The CLI's `@<group>` selector picks up every group membership; `@all` selects every registered baseline regardless of group.

## Inspecting and removing baselines

There is no `unregister` method by design — runs are reproducible. To temporarily exclude a baseline from a comparison, simply omit it from the `compare()` call rather than mutating the registry.

For introspection:

```python
from jurebes.baselines import BASELINES

len(BASELINES)                        # total baseline count
"my_logreg" in BASELINES              # membership test
BASELINES.in_group("my_logreg")       # set of group names this baseline belongs to
BASELINES.groups()["linear"]          # set of baseline names in this group
```

## Reusing the bundled featurizers

`jurebes.featurizers` exports every named featurizer used by the bundled baselines. Mix-and-match without re-implementing:

```python
from jurebes.featurizers import lsa, tfidf_word, text_stats, feature_union
from sklearn.linear_model import LogisticRegression

BASELINES.register(
    "lsa_plus_stats",
    lambda: Pipeline([
        ("feat", feature_union(lsa(50, tfidf_word()), text_stats())),
        ("clf",  LogisticRegression(max_iter=1000)),
    ]),
    group="reduced_dim",
)
```

## Factory determinism

Pin `random_state` for any stochastic step in the factory; otherwise different folds produce different "same baseline" results:

```python
from sklearn.ensemble import RandomForestClassifier

BASELINES.register(
    "my_rf",
    lambda: Pipeline([
        ("feat", tfidf_word()),
        ("clf",  RandomForestClassifier(random_state=0)),
    ]),
    group="tree",
)
```

## Module-import-time registration

To make your baselines available wherever your package is imported, register them in your package `__init__.py`:

```python
# my_jurebes_extensions/__init__.py
from jurebes.baselines import BASELINES
# … register calls …
```

Downstream code that does `import my_jurebes_extensions` will then see them in `BASELINES.names()`.

For OVOS pipeline plugin use, this means importing your extension module before jurebes' pipeline is constructed.

---
- Back to [docs index](../index.md)
