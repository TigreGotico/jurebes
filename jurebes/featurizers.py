"""Named featurizer constructors — thin wrappers over sklearn text vectorizers."""

from __future__ import annotations

from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.pipeline import FeatureUnion


def tfidf_word(min_df=1, max_df=1.0, ngram_range=(1, 1)):
    return TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)


def tfidf_char(ngram_range=(3, 5)):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range)


def count_word(binary=False, ngram_range=(1, 1)):
    return CountVectorizer(binary=binary, ngram_range=ngram_range)


def hashing_word(n_features=2 ** 18):
    return HashingVectorizer(n_features=n_features, alternate_sign=False)


def char_word_union():
    return FeatureUnion([("word", tfidf_word()), ("char", tfidf_char())])
