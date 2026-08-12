# Custom featurizers

A featurizer in jurebes is any scikit-learn transformer: an object implementing `fit(X, y=None)`, `transform(X)`, and (by default via `TransformerMixin`) `fit_transform(X, y=None)`.

## Minimal example

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class LengthFeaturizer(BaseEstimator, TransformerMixin):
    """One-feature vectorizer returning utterance length in characters."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray([[len(s)] for s in X], dtype=np.float64)
```

Plug it into a pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from jurebes import IntentClassifier

clf = IntentClassifier(Pipeline([
    ("feat", LengthFeaturizer()),
    ("clf",  LogisticRegression(max_iter=1000)),
]))
```

## Stateful featurizers

If your transformer learns something at fit time (a vocabulary, a centroid, a vectorizer), store it on `self` with a trailing underscore:

```python
class FrequentTokensFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=100):
        self.top_k = top_k

    def fit(self, X, y=None):
        from collections import Counter
        c = Counter()
        for doc in X:
            c.update(doc.lower().split())
        self.vocab_ = [w for w, _ in c.most_common(self.top_k)]
        return self

    def transform(self, X):
        idx = {w: i for i, w in enumerate(self.vocab_)}
        out = np.zeros((len(X), len(self.vocab_)), dtype=np.float64)
        for i, doc in enumerate(X):
            for w in doc.lower().split():
                if w in idx:
                    out[i, idx[w]] += 1.0
        return out
```

The trailing-underscore convention is sklearn's signal that the attribute was learned at fit time.

## Sparse vs dense output

- Sparse output (`scipy.sparse` matrices) is cheaper for high-dimensional bag-of-words. Downstream classifiers like `MultinomialNB`, `LinearSVC`, `LogisticRegression`, `SGDClassifier` accept sparse input directly.
- Dense output is required by `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`, `MLPClassifier`, `HistGradientBoostingClassifier`, `SklearnAutoencoder`. Insert a densifying step:

```python
from sklearn.preprocessing import FunctionTransformer
Pipeline([
    ("feat", tfidf_word()),
    ("dense", FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)),
    ("clf",  MLPClassifier()),
])
```

## Composing via FeatureUnion

`sklearn.pipeline.FeatureUnion` concatenates the outputs of multiple featurizers column-wise. jurebes wraps this in `feature_union(*builders)`:

```python
from jurebes.featurizers import tfidf_word, text_stats, feature_union
union = feature_union(tfidf_word(), text_stats())
```

The returned object is a `FeatureUnion` instance — drop it into any `Pipeline`.

## Registering a custom featurizer-based baseline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from jurebes.baselines import BASELINES

BASELINES.register(
    "length_logreg",
    lambda: Pipeline([
        ("feat", LengthFeaturizer()),
        ("clf",  LogisticRegression(max_iter=1000)),
    ]),
    group="feature_engineering",
)

assert "length_logreg" in BASELINES.names()
```

Registration must happen before any code path that resolves the name; place it in your package's `__init__.py` if you want module-import auto-registration.

## get_feature_names_out (optional)

For sklearn 1.4+ compatibility with `FeatureUnion` and pipeline introspection, implement `get_feature_names_out(input_features=None)`:

```python
class LengthFeaturizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.asarray([[len(s)] for s in X], dtype=np.float64)
    def get_feature_names_out(self, input_features=None):
        return np.array(["length"])
```

Not strictly required but makes downstream debugging easier.

## Testing your featurizer

Round-trip with `clone`:

```python
from sklearn.base import clone
feat = LengthFeaturizer()
feat.fit(["hello", "hi there"])
clone(feat)              # must not raise
```

Plus a smoke test through `Pipeline.fit / predict`. The harness's `cross_validate` will exercise it across folds.

---
- Back to [docs index](../index.md)
