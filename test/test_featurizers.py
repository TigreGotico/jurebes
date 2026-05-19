"""Tests for the featurizer builders added in B1."""

from __future__ import annotations

import numpy as np

from jurebes.featurizers import (
    feature_union,
    hashing_char,
    lda_topics,
    lsa,
    nmf,
    text_stats,
    tfidf_word,
    tfidf_word_sublinear,
)


_DOCS = [
    "hello world", "hi there friend", "hey there",
    "tell me a joke", "say a joke", "make me laugh",
    "what is your name", "who are you",
    "weather today in Paris", "is it raining outside",
    "play some music", "stop the music please",
]


def test_tfidf_word_sublinear_fits():
    v = tfidf_word_sublinear()
    X = v.fit_transform(_DOCS)
    assert X.shape[0] == len(_DOCS)


def test_lsa_pipeline_reduces():
    p = lsa(n_components=4)
    X = p.fit_transform(_DOCS)
    assert X.shape == (len(_DOCS), 4)


def test_nmf_pipeline_nonneg():
    p = nmf(n_components=3)
    X = p.fit_transform(_DOCS)
    assert X.shape == (len(_DOCS), 3)
    assert (X >= 0).all()


def test_lda_pipeline():
    p = lda_topics(n_topics=3)
    X = p.fit_transform(_DOCS)
    assert X.shape == (len(_DOCS), 3)


def test_hashing_char_fixed_dim():
    v = hashing_char(n_features=512)
    X = v.fit_transform(_DOCS)
    assert X.shape == (len(_DOCS), 512)


def test_text_stats_shape_and_oov():
    t = text_stats().fit(_DOCS)
    X = t.transform(["hello world", "unseen tokens here"])
    assert X.shape == (2, 7)
    # first row: all in-vocab → oov_rate 0
    assert X[0, 6] == 0.0
    # second row: "unseen tokens here" — none of those words are in _DOCS vocab
    assert X[1, 6] > 0.0


def test_text_stats_features_sane():
    t = text_stats().fit(_DOCS)
    X = t.transform(["AaBb 123!"])
    # chars
    assert X[0, 0] == 9
    # words
    assert X[0, 1] == 2
    # digit ratio > 0
    assert X[0, 3] > 0


def test_text_stats_feature_union_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    docs = _DOCS + ["yet another", "more text here"]
    labels = (["a"] * 7) + (["b"] * 7)
    union = feature_union(tfidf_word(), text_stats())
    pipe = Pipeline([("feat", union), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(docs, labels)
    preds = pipe.predict(docs)
    assert len(preds) == len(docs)


def test_text_stats_empty_input():
    t = text_stats()
    t.fit([""])
    # empty list
    X0 = t.transform([])
    assert X0.shape == (0, 7)
    # empty string row
    X1 = t.transform([""])
    assert X1.shape == (1, 7)
    assert (X1 == 0).all()


def test_lda_topics_seeded_is_deterministic():
    p1 = lda_topics(n_topics=3)
    p2 = lda_topics(n_topics=3)
    X1 = p1.fit_transform(_DOCS)
    X2 = p2.fit_transform(_DOCS)
    np.testing.assert_allclose(X1, X2)


def test_nmf_seeded_is_deterministic():
    p1 = nmf(n_components=3)
    p2 = nmf(n_components=3)
    X1 = p1.fit_transform(_DOCS)
    X2 = p2.fit_transform(_DOCS)
    np.testing.assert_allclose(X1, X2)


def test_feature_union_combines():
    u = feature_union(tfidf_word(), text_stats())
    u.fit(_DOCS)
    X = u.transform(_DOCS[:3])
    assert X.shape[0] == 3
    # union width = vocab size + 7
    assert X.shape[1] > 7
