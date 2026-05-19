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
    return Pipeline([("base", base_step), ("nmf", NMF(n_components=n_components, init="nndsvd", max_iter=400))])


def lda_topics(n_topics: int = 20, base=None):
    """Latent Dirichlet Allocation on a count-vector base."""
    base_step = base if base is not None else count_word()
    return Pipeline([
        ("base", base_step),
        ("lda", LatentDirichletAllocation(n_components=n_topics, learning_method="batch", max_iter=20)),
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
