"""Named featurizer constructors — thin wrappers over sklearn text vectorizers.

`CategoricalVectorizer` is ported from the upstream
`guided-categorical-embeddings` project (TigreGotico, Apache-2.0 licence).
"""

from __future__ import annotations

import itertools
import json
import re
import warnings
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.random_projection import SparseRandomProjection


def tfidf_word(min_df=1, max_df=1.0, ngram_range=(1, 1)):
    return TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)


def tfidf_word_sublinear(min_df=1, max_df=1.0, ngram_range=(1, 1)):
    """TF-IDF with sublinear_tf=True — log(1+tf) instead of raw tf."""
    return TfidfVectorizer(
        min_df=min_df, max_df=max_df, ngram_range=ngram_range, sublinear_tf=True,
    )


def tfidf_char(ngram_range=(3, 5)):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range)


def count_word(binary=False, ngram_range=(1, 1)):
    return CountVectorizer(binary=binary, ngram_range=ngram_range)


def hashing_word(n_features=2 ** 18):
    return HashingVectorizer(n_features=n_features, alternate_sign=False)


def hashing_char(n_features=2 ** 18, ngram_range=(3, 5)):
    """Char-ngram HashingVectorizer — fixed memory, online-friendly."""
    return HashingVectorizer(
        analyzer="char_wb", n_features=n_features,
        ngram_range=ngram_range, alternate_sign=False,
    )


def char_word_union():
    return FeatureUnion([("word", tfidf_word()), ("char", tfidf_char())])


def lsa(n_components: int = 200, base=None):
    """Truncated-SVD on a TF-IDF base (latent semantic analysis)."""
    base_step = base if base is not None else tfidf_word()
    return Pipeline([("base", base_step), ("svd", TruncatedSVD(n_components=n_components))])


def nmf(n_components: int = 50, base=None):
    """NMF on a TF-IDF base — non-negative latent factors."""
    base_step = base if base is not None else tfidf_word()
    return Pipeline([("base", base_step), ("nmf", NMF(n_components=n_components, init="nndsvd", max_iter=400, random_state=0))])


def lda_topics(n_topics: int = 20, base=None):
    """Latent Dirichlet Allocation on a count-vector base."""
    base_step = base if base is not None else count_word()
    return Pipeline([
        ("base", base_step),
        ("lda", LatentDirichletAllocation(n_components=n_topics, learning_method="batch", max_iter=20, random_state=0)),
    ])


def feature_union(*builders):
    """Combine multiple featurizer builders into a single FeatureUnion."""
    return FeatureUnion([(f"f{i}", b) for i, b in enumerate(builders)])


# ── skip-grams ─────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"\w+")


def _skipgram_analyzer(n: int = 2, k: int = 2):
    """Return an analyzer callable that emits ``n``-token skip-grams.

    Each skip-gram is an ``n``-tuple of tokens drawn from the document with up
    to ``k`` token positions skipped between any two consecutive members. The
    returned callable accepts a raw document string — the signature
    `TfidfVectorizer(analyzer=...)` expects — lowercases it, tokenises on
    ``\\w+`` and yields space-joined skip-grams (Guthrie et al. 2006).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if k < 0:
        raise ValueError("k must be >= 0")

    def analyzer(doc: str) -> List[str]:
        tokens = _TOKEN_RE.findall(str(doc).lower())
        m = len(tokens)
        if n == 1:
            return list(tokens)
        out: List[str] = []
        for idx in itertools.combinations(range(m), n):
            # consecutive members may skip at most k positions
            if all(idx[i + 1] - idx[i] - 1 <= k for i in range(n - 1)):
                out.append(" ".join(tokens[i] for i in idx))
        return out

    return analyzer


def skipgram_word(n=2, k=2, min_df=1, max_df=1.0):
    """TF-IDF over ``n``-token skip-grams with up to ``k`` skipped positions."""
    return TfidfVectorizer(analyzer=_skipgram_analyzer(n, k),
                           min_df=min_df, max_df=max_df)


# ── BM25 ───────────────────────────────────────────────────────────────────

class BM25Transformer(BaseEstimator, TransformerMixin):
    """Okapi BM25 term-weighting transformer over a count matrix.

    Sits after a `CountVectorizer`. ``fit`` records per-term document
    frequencies, the average document length and the BM25 idf vector;
    ``transform`` applies the Okapi saturation formula

        ``idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``

    over a sparse count matrix (Robertson & Zaragoza 2009). Implemented with
    pure numpy and scipy.sparse so a fitted instance is picklable.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, X, y=None):
        X = sp.csr_matrix(X)
        n_docs = X.shape[0]
        # document frequency: number of docs each term appears in
        df = np.bincount(X.indices, minlength=X.shape[1]) if n_docs else np.zeros(X.shape[1])
        # for non-binary count matrices, recompute df from the binarised pattern
        binary = X.copy()
        binary.data = np.ones_like(binary.data)
        df = np.asarray(binary.sum(axis=0)).ravel()
        # BM25 idf with the +1 smoothing that keeps weights non-negative
        self.idf_ = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        doc_len = np.asarray(X.sum(axis=1)).ravel()
        self.avgdl_ = float(doc_len.mean()) if n_docs else 0.0
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = sp.csr_matrix(X, dtype=np.float64)
        doc_len = np.asarray(X.sum(axis=1)).ravel()
        avgdl = self.avgdl_ if self.avgdl_ > 0 else 1.0
        # per-row denominator component k1 * (1 - b + b * dl / avgdl)
        denom_norm = self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)
        out = X.tocoo(copy=True)
        tf = out.data
        norm = denom_norm[out.row]
        weighted = tf * (self.k1 + 1.0) / (tf + norm)
        out.data = weighted * self.idf_[out.col]
        return out.tocsr()


def bm25_word(ngram_range=(1, 1), min_df=1, k1=1.5, b=0.75):
    """Okapi BM25 term weighting over a `CountVectorizer` base."""
    return Pipeline([
        ("count", CountVectorizer(ngram_range=ngram_range, min_df=min_df)),
        ("bm25", BM25Transformer(k1=k1, b=b)),
    ])


# ── text_stats ─────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]")


class TextStatsTransformer(BaseEstimator, TransformerMixin):
    """Vectorised text-statistics featurizer.

    Per utterance produces a fixed-size dense feature row:
        [n_chars, n_words, mean_word_len, digit_ratio, upper_ratio,
         punct_ratio, oov_rate]

    The training vocabulary is captured during ``fit`` for the OOV-rate
    feature.
    """

    def fit(self, X, y=None):
        vocab = set()
        for s in X:
            for w in str(s).lower().split():
                vocab.add(w)
        self.vocab_ = vocab
        return self

    def transform(self, X):
        X = list(X)
        n = len(X)
        out = np.zeros((n, 7), dtype=np.float64)
        if n == 0:
            return out
        vocab = getattr(self, "vocab_", set())
        # vectorised-ish: still a small Python loop over rows (unavoidable
        # for the per-utt token operations), but no nested per-token loops
        # for the numeric features — all numpy after extraction.
        arr_chars = np.empty(n, dtype=np.int64)
        arr_words = np.empty(n, dtype=np.int64)
        arr_mean_wlen = np.empty(n, dtype=np.float64)
        arr_digit = np.empty(n, dtype=np.float64)
        arr_upper = np.empty(n, dtype=np.float64)
        arr_punct = np.empty(n, dtype=np.float64)
        arr_oov = np.empty(n, dtype=np.float64)
        for i, raw in enumerate(X):
            s = str(raw)
            chars = len(s)
            words = s.split()
            wc = len(words)
            arr_chars[i] = chars
            arr_words[i] = wc
            if wc:
                # numpy mean over word lengths
                wlens = np.fromiter((len(w) for w in words), dtype=np.int64, count=wc)
                arr_mean_wlen[i] = float(wlens.mean())
            else:
                arr_mean_wlen[i] = 0.0
            if chars:
                digit_count = sum(c.isdigit() for c in s)
                upper_count = sum(c.isupper() for c in s)
                punct_count = len(_PUNCT_RE.findall(s))
                arr_digit[i] = digit_count / chars
                arr_upper[i] = upper_count / chars
                arr_punct[i] = punct_count / chars
            else:
                arr_digit[i] = arr_upper[i] = arr_punct[i] = 0.0
            if wc and vocab:
                oov = sum(1 for w in words if w.lower() not in vocab)
                arr_oov[i] = oov / wc
            else:
                arr_oov[i] = 0.0
        out[:, 0] = arr_chars
        out[:, 1] = arr_words
        out[:, 2] = arr_mean_wlen
        out[:, 3] = arr_digit
        out[:, 4] = arr_upper
        out[:, 5] = arr_punct
        out[:, 6] = arr_oov
        return out


def text_stats():
    """Return a fresh TextStatsTransformer instance."""
    return TextStatsTransformer()


# ── autoencoder ───────────────────────────────────────────────────────────

def _ae_activation(name: str):
    if name == "relu":
        return lambda Z: np.maximum(Z, 0.0)
    if name == "identity":
        return lambda Z: Z
    if name == "tanh":
        return np.tanh
    if name == "logistic":
        return lambda Z: 1.0 / (1.0 + np.exp(-Z))
    raise ValueError(f"unsupported activation: {name}")


def _auto_sizes(n_features: int) -> tuple:
    """Heuristic MLP layer sizes given the input dimension.

    Returns a symmetric ``(wide, narrow, wide)`` triple where the bottleneck
    scales with ``sqrt(n_features)`` so the latent capacity adapts to the
    actual TF-IDF vocab size of the corpus. Shared by ``SklearnAutoencoder``
    (reconstruction bottleneck) and ``LabelGuidedEmbeddingsTransformer``
    (supervised classifier bottleneck).
    """
    wide = max(64, int(2 * np.sqrt(n_features)))
    bottleneck = max(32, int(np.sqrt(n_features)))
    return (wide, bottleneck, wide)


class SklearnAutoencoder(BaseEstimator, TransformerMixin):
    """Neural-bottleneck autoencoder built on `MLPRegressor` fitted to ``y = X``.

    ``transform`` runs the forward pass through the encoder layers up to (and
    including) the bottleneck using ``mlp.coefs_`` / ``mlp.intercepts_`` in
    pure numpy. ``inverse_transform`` continues the forward pass through the
    remaining decoder layers. Sparse input is densified via ``.toarray()``.

    Parameters
    ----------
    hidden_layer_sizes : tuple or "auto"
        Layer widths including the bottleneck. ``"auto"`` (default) scales
        the layers from the input feature count at fit time so the bottleneck
        does not become artificially narrow on high-class-count datasets.
    noise_level : float
        Denoising autoencoder noise standard deviation. When > 0, Gaussian
        noise is added to the input at fit time while the target stays clean;
        the model learns a noise-robust reconstruction (Vincent et al. 2008).
    """

    def __init__(
        self,
        hidden_layer_sizes="auto",
        bottleneck_index: Optional[int] = None,
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 1e-4,
        max_iter: int = 500,
        random_state: Optional[int] = 0,
        learning_rate_init: float = 1e-3,
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
        noise_level: float = 0.0,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.bottleneck_index = bottleneck_index
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.learning_rate_init = learning_rate_init
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.noise_level = noise_level

    def _densify(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64)

    def _resolve_sizes(self, n_features: int) -> tuple:
        if self.hidden_layer_sizes == "auto":
            return _auto_sizes(n_features)
        return tuple(self.hidden_layer_sizes)

    def _resolve_bottleneck(self) -> int:
        sizes = list(self.layer_sizes_)
        if self.bottleneck_index is None:
            return int(np.argmin(sizes))
        return int(self.bottleneck_index)

    def fit(self, X, y=None):
        from sklearn.neural_network import MLPRegressor
        from sklearn.utils import check_random_state

        Xd = self._densify(X)
        self.layer_sizes_ = self._resolve_sizes(Xd.shape[1])
        if self.noise_level > 0:
            rng = check_random_state(self.random_state)
            X_in = Xd + rng.normal(0.0, self.noise_level, size=Xd.shape)
        else:
            X_in = Xd
        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=self.layer_sizes_,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            learning_rate_init=self.learning_rate_init,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
        )
        self.mlp_.fit(X_in, Xd)
        self.bottleneck_index_ = self._resolve_bottleneck()
        self.n_features_in_ = Xd.shape[1]
        return self

    def _forward(self, X: np.ndarray, start: int, stop: int) -> np.ndarray:
        """Run affine + activation chain from layer index ``start`` to ``stop``
        inclusive over ``mlp_.coefs_`` / ``mlp_.intercepts_``."""
        act = _ae_activation(self.activation)
        Z = X
        coefs = self.mlp_.coefs_
        intercepts = self.mlp_.intercepts_
        last = len(coefs) - 1
        for layer in range(start, stop + 1):
            Z = Z @ coefs[layer] + intercepts[layer]
            # Final output layer of an MLPRegressor uses identity activation;
            # all hidden layers use the configured activation.
            if layer != last:
                Z = act(Z)
        return Z

    def transform(self, X):
        Xd = self._densify(X)
        # Encoder runs from layer 0 up to and including the bottleneck layer.
        return self._forward(Xd, 0, self.bottleneck_index_)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        last = len(self.mlp_.coefs_) - 1
        # Decoder runs from layer immediately after the bottleneck to output.
        start = self.bottleneck_index_ + 1
        if start > last:
            return Z
        return self._forward(Z, start, last)

    def reconstruction_error(self, X) -> np.ndarray:
        Xd = self._densify(X)
        Xr = self._forward(Xd, 0, len(self.mlp_.coefs_) - 1)
        return ((Xd - Xr) ** 2).mean(axis=1)


def _toarray(X):
    return X.toarray() if hasattr(X, "toarray") else X


def autoencoder(hidden_layer_sizes="auto", base=None, **kwargs):
    """Build a Pipeline ending in a `SklearnAutoencoder` bottleneck.

    When ``base`` is provided, the pipeline is ``[base, dense, autoencoder]``;
    a `FunctionTransformer` densifies sparse output from the base. Without a
    base, returns a single-step pipeline wrapping the autoencoder directly.

    Default ``hidden_layer_sizes="auto"`` derives reasonable layer widths
    from the input feature dimension at fit time.
    """
    from sklearn.preprocessing import FunctionTransformer
    ae = SklearnAutoencoder(hidden_layer_sizes=hidden_layer_sizes, **kwargs)
    if base is None:
        return Pipeline([("ae", ae)])
    return Pipeline([
        ("base", base),
        ("dense", FunctionTransformer(_toarray, accept_sparse=True)),
        ("ae", ae),
    ])


def denoising_autoencoder(noise_level: float = 0.1, hidden_layer_sizes="auto",
                          base=None, **kwargs):
    """Build a Pipeline ending in a denoising `SklearnAutoencoder`.

    Convenience wrapper around :func:`autoencoder` that sets ``noise_level``
    (Vincent et al. 2008). Gaussian noise of the given standard deviation
    is added to the input at training time only; ``transform`` runs on the
    clean input.
    """
    kwargs.pop("noise_level", None)
    return autoencoder(hidden_layer_sizes=hidden_layer_sizes, base=base,
                       noise_level=noise_level, **kwargs)


class LabelGuidedEmbeddingsTransformer(BaseEstimator, TransformerMixin):
    """Label-guided neural embedding (Apache-2.0 port).

    Trains an `MLPClassifier` end-to-end on ``(X, y)``; ``transform`` returns
    the activations of a chosen hidden layer, optionally densified via PCA
    so the output dimension stays fixed regardless of layer width.

    The training objective is class discrimination, not reconstruction, so
    the bottleneck preserves class-discriminative axes by construction.
    This addresses the "unsupervised reconstruction misaligned with
    classification" failure mode of vanilla autoencoders on high-class-count
    datasets.

    Ported from `guided-categorical-embeddings-sklearn`
    (https://github.com/TigreGotico/guided-categorical-embeddings-sklearn,
    Apache-2.0); adapted to jurebes' sparse-input + auto-sized defaults.
    """

    def __init__(
        self,
        hidden_layer_sizes="auto",
        hidden_layer_index: int = 0,
        embedding_size: Optional[int] = None,
        max_iter: int = 500,
        random_state: int = 0,
        early_stopping: bool = False,
        n_iter_no_change: int = 10,
    ):
        # ``early_stopping`` defaults to False because MLPClassifier's
        # internal validation-score check (np.isnan on y_pred) trips on
        # string class labels. Users who pass integer labels can flip it on.
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_index = hidden_layer_index
        self.embedding_size = embedding_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change

    def _densify(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64)

    def _resolve_sizes(self, n_features: int) -> tuple:
        if self.hidden_layer_sizes == "auto":
            return _auto_sizes(n_features)
        return tuple(self.hidden_layer_sizes)

    def fit(self, X, y):
        from sklearn.decomposition import PCA
        from sklearn.neural_network import MLPClassifier

        Xd = self._densify(X)
        self.layer_sizes_ = self._resolve_sizes(Xd.shape[1])
        self.mlp_ = MLPClassifier(
            hidden_layer_sizes=self.layer_sizes_,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
        )
        self.mlp_.fit(Xd, y)
        self.n_classes_ = len(set(y))
        hidden = self._activations(Xd)
        target = self.embedding_size or 3 * self.n_classes_
        n_components = min(target, hidden.shape[1])
        self.pca_ = PCA(n_components=n_components, random_state=self.random_state)
        self.pca_.fit(hidden)
        self.n_features_in_ = Xd.shape[1]
        return self

    def _activations(self, X: np.ndarray) -> np.ndarray:
        coefs = self.mlp_.coefs_
        intercepts = self.mlp_.intercepts_
        act = X
        for i in range(self.hidden_layer_index + 1):
            act = np.dot(act, coefs[i]) + intercepts[i]
            act = np.maximum(act, 0)  # ReLU
        return act

    def transform(self, X):
        Xd = self._densify(X)
        return self.pca_.transform(self._activations(Xd))

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def label_guided(hidden_layer_sizes="auto", base=None, **kwargs):
    """Build a Pipeline ending in a `LabelGuidedEmbeddingsTransformer`.

    The classification head trained inside the embedder is discarded; only
    the PCA-densified hidden activations flow downstream to the user's
    classifier.
    """
    from sklearn.preprocessing import FunctionTransformer
    lge = LabelGuidedEmbeddingsTransformer(hidden_layer_sizes=hidden_layer_sizes, **kwargs)
    if base is None:
        return Pipeline([("lge", lge)])
    return Pipeline([
        ("base", base),
        ("dense", FunctionTransformer(_toarray, accept_sparse=True)),
        ("lge", lge),
    ])


# ── categorical vectorizer ────────────────────────────────────────────────
# Ported from guided-categorical-embeddings (TigreGotico, Apache-2.0).

class CategoricalVectorizer(BaseEstimator, TransformerMixin):
    """One-hot vectorizer over list-of-dict categorical features.

    Each unique ``key=value`` pair becomes a column. Unknown pairs at
    transform time produce a zero column for that feature without raising.
    Vocabulary is saved/loaded as JSON (no pickle).
    """

    def __init__(self, min_frequency: int = 1) -> None:
        self.vocabulary_: Optional[Dict[str, int]] = None
        self.min_frequency = min_frequency

    @property
    def n_features(self) -> int:
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer has not been fitted yet.")
        return len(self.vocabulary_)

    def fit(self, X: List[Dict[str, str]], y=None) -> "CategoricalVectorizer":
        if not X:
            raise ValueError("Cannot fit on empty data.")
        counts: Counter = Counter()
        for row in X:
            for key, value in row.items():
                counts[f"{key}={value}"] += 1
        if self.min_frequency > 1:
            feature_set = {f for f, c in counts.items() if c >= self.min_frequency}
        else:
            feature_set = set(counts.keys())
        if not feature_set:
            warnings.warn(
                f"All features filtered out by min_frequency={self.min_frequency}. "
                f"Vocabulary is empty.",
                UserWarning,
                stacklevel=2,
            )
        self.vocabulary_ = {feat: idx for idx, feat in enumerate(sorted(feature_set))}
        return self

    def transform(self, X: List[Dict[str, str]]) -> np.ndarray:
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer has not been fitted yet.")
        result = np.zeros((len(X), len(self.vocabulary_)), dtype=np.float32)
        for i, row in enumerate(X):
            for key, value in row.items():
                feat = f"{key}={value}"
                if feat in self.vocabulary_:
                    result[i, self.vocabulary_[feat]] = 1.0
        return result

    def fit_transform(self, X: List[Dict[str, str]], y=None) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> List[Dict[str, str]]:
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer has not been fitted yet.")
        idx_to_feat = {idx: feat for feat, idx in self.vocabulary_.items()}
        results: List[Dict[str, str]] = []
        for row in X:
            d: Dict[str, str] = {}
            for idx in np.nonzero(row)[0]:
                feat = idx_to_feat[int(idx)]
                key, value = feat.split("=", 1)
                d[key] = value
            results.append(d)
        return results

    def save(self, path: str) -> None:
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer has not been fitted yet.")
        data = {"vocabulary": self.vocabulary_, "min_frequency": self.min_frequency}
        with open(path, "w") as f:
            json.dump(data, f, sort_keys=True)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        if "vocabulary" in data and "min_frequency" in data:
            self.vocabulary_ = data["vocabulary"]
            self.min_frequency = data["min_frequency"]
        else:
            warnings.warn(
                "Loaded legacy vocabulary format; min_frequency not preserved.",
                UserWarning,
                stacklevel=2,
            )
            self.vocabulary_ = data


def categorical(min_frequency: int = 1) -> CategoricalVectorizer:
    """Return a fresh `CategoricalVectorizer` for dict-of-string inputs."""
    return CategoricalVectorizer(min_frequency=min_frequency)
