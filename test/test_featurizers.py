"""Tests for the featurizer builders added in B1."""

from __future__ import annotations

import numpy as np

from jurebes.featurizers import (
    BM25Transformer,
    CategoricalVectorizer,
    SklearnAutoencoder,
    autoencoder,
    bm25_word,
    categorical,
    feature_union,
    hashing_char,
    lda_topics,
    lsa,
    nmf,
    random_projection,
    skipgram_word,
    text_stats,
    tfidf_word,
    tfidf_word_sublinear,
    _SkipgramAnalyzer,
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


def _ae_toy_X(n=40, d=5, seed=0):
    return np.random.RandomState(seed).rand(n, d)


def test_autoencoder_shape():
    X = _ae_toy_X()
    ae = SklearnAutoencoder(hidden_layer_sizes=(8, 3, 8), max_iter=50, random_state=0)
    ae.fit(X)
    Z = ae.transform(X)
    assert Z.shape == (X.shape[0], 3)


def test_autoencoder_deterministic():
    X = _ae_toy_X()
    a1 = SklearnAutoencoder(hidden_layer_sizes=(8, 3, 8), max_iter=50, random_state=42).fit(X)
    a2 = SklearnAutoencoder(hidden_layer_sizes=(8, 3, 8), max_iter=50, random_state=42).fit(X)
    np.testing.assert_allclose(a1.transform(X), a2.transform(X))


def test_autoencoder_inverse_transform_shape():
    X = _ae_toy_X()
    ae = SklearnAutoencoder(hidden_layer_sizes=(8, 3, 8), max_iter=50, random_state=0).fit(X)
    Xr = ae.inverse_transform(ae.transform(X))
    assert Xr.shape == X.shape


def test_autoencoder_reconstruction_error_decreases_with_training():
    X = _ae_toy_X(n=80, d=6, seed=1)
    short = SklearnAutoencoder(hidden_layer_sizes=(8, 3, 8), max_iter=5, random_state=0).fit(X)
    long = SklearnAutoencoder(hidden_layer_sizes=(8, 3, 8), max_iter=200, random_state=0).fit(X)
    assert long.reconstruction_error(X).mean() < short.reconstruction_error(X).mean()


def test_autoencoder_in_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    docs = _DOCS + ["yet another", "more text here"]
    labels = (["a"] * 7) + (["b"] * 7)
    pipe = Pipeline([
        ("feat", autoencoder(hidden_layer_sizes=(16, 4, 16), base=tfidf_word(), max_iter=100, random_state=0)),
        ("clf", LogisticRegression(max_iter=500)),
    ])
    pipe.fit(docs, labels)
    preds = pipe.predict(docs)
    assert len(preds) == len(docs)


def test_categorical_round_trip():
    X = [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}, {"a": "1", "b": "y"}]
    v = categorical()
    v.fit(X)
    Z = v.transform(X)
    assert Z.shape[0] == 3
    assert v.inverse_transform(Z) == X


def test_categorical_save_load_json(tmp_path):
    X = [{"k": "a"}, {"k": "b"}, {"k": "a"}]
    v1 = categorical().fit(X)
    p = tmp_path / "vocab.json"
    v1.save(str(p))
    v2 = CategoricalVectorizer()
    v2.load(str(p))
    assert v2.vocabulary_ == v1.vocabulary_
    np.testing.assert_array_equal(v2.transform(X), v1.transform(X))


def test_categorical_unknown_category_zero_vector():
    v = categorical().fit([{"a": "1"}, {"a": "2"}])
    Z = v.transform([{"a": "3"}])
    # unseen value yields all-zero row across the column group
    assert Z.shape == (1, 2)
    assert (Z == 0).all()


def test_categorical_min_frequency_filter():
    X = [{"k": "a"}, {"k": "a"}, {"k": "b"}]  # 'b' appears once
    v = CategoricalVectorizer(min_frequency=2).fit(X)
    # only 'k=a' survives
    assert set(v.vocabulary_.keys()) == {"k=a"}


def test_feature_union_combines():
    u = feature_union(tfidf_word(), text_stats())
    u.fit(_DOCS)
    X = u.transform(_DOCS[:3])
    assert X.shape[0] == 3
    # union width = vocab size + 7
    assert X.shape[1] > 7


# ── skip-grams ─────────────────────────────────────────────────────────────

def test_skipgram_analyzer_emits_expected():
    a = _SkipgramAnalyzer(n=2, k=1)
    # "a b c d": pairs whose member gap is at most k+1 positions apart
    grams = a("a b c d")
    assert grams == ["a b", "a c", "b c", "b d", "c d"]


def test_skipgram_analyzer_k0_is_contiguous_bigrams():
    a = _SkipgramAnalyzer(n=2, k=0)
    assert a("one two three") == ["one two", "two three"]


def test_skipgram_analyzer_lowercases():
    a = _SkipgramAnalyzer(n=2, k=2)
    assert all(g == g.lower() for g in a("Hello WORLD"))


def test_skipgram_word_fits():
    v = skipgram_word(n=2, k=2)
    X = v.fit_transform(_DOCS)
    assert X.shape[0] == len(_DOCS)
    assert X.shape[1] > 0


# ── BM25 ───────────────────────────────────────────────────────────────────


def test_skipgram_word_pickles():
    """A fitted skipgram vectorizer must joblib-round-trip (closure-free)."""
    import io, joblib
    v = skipgram_word(n=2, k=2)
    v.fit(["play africa now", "set a timer", "what is the weather"])
    buf = io.BytesIO()
    joblib.dump(v, buf)
    buf.seek(0)
    v2 = joblib.load(buf)
    assert v2.transform(["play the song"]).shape[1] == len(v.vocabulary_)


def test_bm25_output_non_negative():
    p = bm25_word()
    X = p.fit_transform(_DOCS)
    assert X.shape[0] == len(_DOCS)
    assert X.min() >= 0.0


def test_bm25_rank_orders_toy_corpus():
    # a rare term must outscore a ubiquitous one in the document containing both
    corpus = ["common common common rare", "common", "common", "common"]
    p = bm25_word()
    X = p.fit_transform(corpus).toarray()
    vocab = p.named_steps["count"].vocabulary_
    rare_w = X[0, vocab["rare"]]
    common_w = X[0, vocab["common"]]
    assert rare_w > common_w > 0.0


def test_bm25_pipeline_round_trips_via_joblib(tmp_path):
    import joblib
    p = bm25_word()
    p.fit(_DOCS)
    path = tmp_path / "bm25.joblib"
    joblib.dump(p, path)
    p2 = joblib.load(path)
    np.testing.assert_allclose(
        p.transform(_DOCS).toarray(), p2.transform(_DOCS).toarray())


def test_bm25_transformer_params():
    t = BM25Transformer(k1=2.0, b=0.5)
    assert t.k1 == 2.0 and t.b == 0.5


# ── random projection ──────────────────────────────────────────────────────

def test_random_projection_yields_requested_components():
    docs = _DOCS * 4
    p = random_projection(n_components=6, base=tfidf_word())
    X = p.fit_transform(docs)
    assert X.shape == (len(docs), 6)


def test_random_projection_seeded_is_deterministic():
    docs = _DOCS * 4
    X1 = random_projection(n_components=6, base=tfidf_word(), seed=1).fit_transform(docs)
    X2 = random_projection(n_components=6, base=tfidf_word(), seed=1).fit_transform(docs)
    assert (X1 != X2).nnz == 0 if hasattr(X1, "nnz") else np.array_equal(X1, X2)
