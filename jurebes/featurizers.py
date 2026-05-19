"""Named featurizer constructors — thin wrappers over sklearn text vectorizers."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.pipeline import FeatureUnion, Pipeline


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


class SklearnAutoencoder(BaseEstimator, TransformerMixin):
    """Neural-bottleneck autoencoder built on `MLPRegressor` fitted to ``y = X``.

    ``transform`` runs the forward pass through the encoder layers up to (and
    including) the bottleneck using ``mlp.coefs_`` / ``mlp.intercepts_`` in
    pure numpy. ``inverse_transform`` continues the forward pass through the
    remaining decoder layers. Sparse input is densified via ``.toarray()``.
    """

    def __init__(
        self,
        hidden_layer_sizes=(64, 16, 64),
        bottleneck_index: Optional[int] = None,
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 1e-4,
        max_iter: int = 200,
        random_state: Optional[int] = 0,
        learning_rate_init: float = 1e-3,
        early_stopping: bool = False,
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

    def _densify(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64)

    def _resolve_bottleneck(self) -> int:
        sizes = list(self.hidden_layer_sizes)
        if self.bottleneck_index is None:
            return int(np.argmin(sizes))
        return int(self.bottleneck_index)

    def fit(self, X, y=None):
        from sklearn.neural_network import MLPRegressor

        Xd = self._densify(X)
        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            learning_rate_init=self.learning_rate_init,
            early_stopping=self.early_stopping,
        )
        self.mlp_.fit(Xd, Xd)
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


def autoencoder(hidden_layer_sizes=(64, 16, 64), base=None, **kwargs):
    """Build a Pipeline ending in a `SklearnAutoencoder` bottleneck.

    When ``base`` is provided, the pipeline is ``[base, dense, autoencoder]``;
    a `FunctionTransformer` densifies sparse output from the base. Without a
    base, returns a single-step pipeline wrapping the autoencoder directly.
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
