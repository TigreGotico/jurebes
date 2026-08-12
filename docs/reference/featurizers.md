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

### `random_projection(n_components=200, base=None, seed=0)`

`SparseRandomProjection` reduced-dim featurizer. Projects the input (or `base` featurizer output) onto a `n_components`-dimensional random subspace; accepts sparse input directly. Returns `Pipeline([("base", base), ("rp", rp)])` or `Pipeline([("rp", rp)])` when `base` is `None`.

## Skip-grams and BM25

### `skipgram_word(n=2, k=2, min_df=1, max_df=1.0)`

`TfidfVectorizer` with a skip-gram analyzer. Emits every `n`-token tuple with up to `k` skipped positions between consecutive members.

### `bm25_word(ngram_range=(1, 1), min_df=1, k1=1.5, b=0.75)`

Okapi BM25 term weighting. Returns `Pipeline([("count", CountVectorizer(...)), ("bm25", BM25Transformer(k1, b))])`.

## Linguistic

Each builder composes a text→text preprocessing transformer with a TF-IDF tail; `**tfidf_kw` flow to the `TfidfVectorizer`. The optional dependency is lazily imported, raising a clear `ImportError` naming the extra when absent.

### `pos_sequence(lang="en", **tfidf_kw)`

TF-IDF over part-of-speech tag sequences via `brill_postagger` (`[postag]` extra). Raises `ValueError` for a language outside brill's 11.

### `word_pos(lang="en", **tfidf_kw)`

TF-IDF over hybrid `token__POS` tokens via `brill_postagger` (`[postag]` extra). Raises `ValueError` for a language outside brill's 11.

### `stemmed_tfidf(lang="en", **tfidf_kw)`

TF-IDF over Snowball-stemmed tokens via `nltk` (`[stem]` extra). Raises `ValueError` for a language outside Snowball's 15.

### `lemmatized_tfidf(lang="en", **tfidf_kw)`

TF-IDF over lemmatised tokens via `simplemma` (`[lemma]` extra).

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

### `BM25Transformer`

`BaseEstimator + TransformerMixin`. Constructor `BM25Transformer(k1=1.5, b=0.75)`. Sits after a `CountVectorizer`. `fit` records the BM25 idf vector (`idf_`) and average document length (`avgdl_`); `transform` applies the Okapi saturation formula over a sparse count matrix using pure numpy and scipy.sparse, so a fitted instance is picklable.

### `PosSequenceTransformer` / `WordPosTransformer`

`BaseEstimator + TransformerMixin`, constructor takes `lang`. `fit` loads a `brill_postagger` pretrained model once (cached on the instance), raising `ValueError` for an unsupported language and `ImportError` when the `[postag]` extra is absent. `transform` maps `list[str] -> list[str]`: a POS-tag sequence string, or hybrid `token__POS` tokens respectively.

### `StemTransformer` / `LemmaTransformer`

`BaseEstimator + TransformerMixin`, constructor takes `lang`. `StemTransformer` loads an `nltk` Snowball stemmer (`[stem]` extra); `LemmaTransformer` uses `simplemma` (`[lemma]` extra). `transform` maps `list[str] -> list[str]` of stemmed / lemmatised tokens.

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
