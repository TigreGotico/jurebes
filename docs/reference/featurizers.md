# `jurebes.featurizers`

Every public featurizer constructor and class.

## TF-IDF

### `tfidf_word(min_df=1, max_df=1.0, ngram_range=(1, 1))`

Returns a `TfidfVectorizer` over word tokens.

```python
from jurebes.featurizers import tfidf_word
feat = tfidf_word(ngram_range=(1, 2), min_df=2)
```

### `tfidf_word_sublinear(min_df=1, max_df=1.0, ngram_range=(1, 1))`

As `tfidf_word` but with `sublinear_tf=True` — TF replaced by `1 + log(tf)`.

### `tfidf_char(ngram_range=(3, 5))`

Returns a `TfidfVectorizer` with `analyzer="char_wb"` — character n-grams within word boundaries.

```python
feat = tfidf_char(ngram_range=(2, 5))
```

## Counts

### `count_word(binary=False, ngram_range=(1, 1))`

Returns a `CountVectorizer`. Set `binary=True` for presence/absence vectors.

## Hashing

### `hashing_word(n_features=2 ** 18)`

Returns a `HashingVectorizer` over word tokens with `alternate_sign=False`. Fixed-memory, no fit step.

### `hashing_char(n_features=2 ** 18, ngram_range=(3, 5))`

Char-ngram `HashingVectorizer` with `analyzer="char_wb"`, `alternate_sign=False`.

## Unions

### `char_word_union()`

Returns a `FeatureUnion([("word", tfidf_word()), ("char", tfidf_char())])`. Concatenates word and char TF-IDF columns.

### `feature_union(*builders)`

Generic combinator. Each `builder` is a transformer (or fresh transformer instance). Returns a `FeatureUnion` numbered `f0`, `f1`, ….

```python
from jurebes.featurizers import feature_union, tfidf_word, text_stats
union = feature_union(tfidf_word(), text_stats())
```

## Reduced dimensionality

### `lsa(n_components=200, base=None)`

Truncated-SVD on top of a TF-IDF base. Returns a `Pipeline([("base", base or tfidf_word()), ("svd", TruncatedSVD(n_components=n_components))])`.

### `nmf(n_components=50, base=None)`

NMF on top of a TF-IDF base. `init="nndsvd"`, `max_iter=400`, `random_state=0`.

### `lda_topics(n_topics=20, base=None)`

Latent Dirichlet Allocation on a count-vector base. `learning_method="batch"`, `max_iter=20`, `random_state=0`.

### `autoencoder(hidden_layer_sizes=(64, 16, 64), base=None, **kwargs)`

Neural-bottleneck autoencoder built on top of an optional base featurizer. Returns a `Pipeline` with the autoencoder as the final step.

`**kwargs` flow to `SklearnAutoencoder`'s constructor.

## Composite

### `text_stats()`

Returns a fresh `TextStatsTransformer` — produces a dense 7-column vector per utterance:

| column | meaning |
| --- | --- |
| 0 | `n_chars` |
| 1 | `n_words` |
| 2 | mean word length |
| 3 | digit ratio |
| 4 | upper-case ratio |
| 5 | punctuation ratio |
| 6 | OOV rate vs training vocabulary |

## Categorical

### `categorical(min_frequency=1)`

Returns a fresh `CategoricalVectorizer`. Accepts `list[dict[str, str]]` input — each unique `key=value` pair becomes a one-hot column. Unknown pairs at transform time produce zero columns (no exception).

## Classes

### `TextStatsTransformer`

`BaseEstimator + TransformerMixin`. Captures the training vocabulary in `vocab_` during `fit`; uses it for the OOV-rate column at `transform`. Always returns a dense float64 matrix.

### `SklearnAutoencoder`

`BaseEstimator + TransformerMixin`. Trains an `MLPRegressor` with `y = X`; `transform` runs the encoder layers through the bottleneck; `inverse_transform` continues the forward pass through the decoder layers.

Constructor:

```python
SklearnAutoencoder(
    hidden_layer_sizes=(64, 16, 64),
    bottleneck_index=None,        # default: argmin of hidden_layer_sizes
    activation="relu",            # "relu" | "identity" | "tanh" | "logistic"
    solver="adam",
    alpha=1e-4,
    max_iter=200,
    random_state=0,
    learning_rate_init=1e-3,
    early_stopping=False,
)
```

Extra method `reconstruction_error(X) -> np.ndarray` — per-row MSE between input and reconstruction; useful as an OOD score.

### `CategoricalVectorizer`

Ported from the upstream `guided-categorical-embeddings` project (TigreGotico, Apache-2.0).

Constructor `CategoricalVectorizer(min_frequency=1)`. Methods:

- `fit(X, y=None)` — build the `{f"{key}={value}": column_idx}` vocabulary.
- `transform(X)` — one-hot encode; unknown pairs become zero columns silently.
- `inverse_transform(X)` — recover `list[dict[str, str]]` from the one-hot encoding.
- `save(path)` / `load(path)` — JSON-serialised vocabulary (no pickle).
- `n_features` (property) — vocabulary size.

## Composition examples

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from jurebes import IntentClassifier
from jurebes.featurizers import (
    tfidf_word, tfidf_char, text_stats, feature_union, lsa, autoencoder,
)

# 1. char+word TF-IDF + text_stats, logreg
clf = IntentClassifier(Pipeline([
    ("feat", feature_union(tfidf_word(), tfidf_char(), text_stats())),
    ("clf",  LogisticRegression(max_iter=1000)),
]))

# 2. LSA on a sublinear-TF base, with custom n_components
from jurebes.featurizers import tfidf_word_sublinear
clf = IntentClassifier(Pipeline([
    ("feat", lsa(100, tfidf_word_sublinear(ngram_range=(1, 2)))),
    ("clf",  LogisticRegression(max_iter=1000)),
]))

# 3. autoencoder bottleneck + linear SVC
from sklearn.svm import LinearSVC
clf = IntentClassifier(Pipeline([
    ("feat", autoencoder(hidden_layer_sizes=(128, 32, 128), base=tfidf_word())),
    ("clf",  LinearSVC()),
]))
```

---
- Back to [reference index](index.md)
