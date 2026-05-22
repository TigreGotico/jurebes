"""Tests for the optional-dependency linguistic featurizers.

Each test is guarded by ``pytest.importorskip`` so the suite stays green when
the ``[postag]`` / ``[stem]`` / ``[lemma]`` extras are not installed.
"""

from __future__ import annotations

import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from jurebes.featurizers import (
    LemmaTransformer,
    PosSequenceTransformer,
    StemTransformer,
    WordPosTransformer,
    lemmatized_tfidf,
    pos_sequence,
    stemmed_tfidf,
    tfidf_word,
    word_pos,
)


_DOCS = [
    "hello there friend", "hi friend", "good morning",
    "tell me a joke", "say a joke", "make me laugh",
    "what is your name", "who are you", "tell me your name",
    "play some music", "stop the music", "play a song",
]
_LABELS = ["greet"] * 3 + ["joke"] * 3 + ["name"] * 3 + ["music"] * 3


def _fit_score(pipe: Pipeline) -> float:
    pipe.fit(_DOCS, _LABELS)
    preds = pipe.predict(_DOCS)
    return sum(p == t for p, t in zip(preds, _LABELS)) / len(_LABELS)


# ── POS-tag featurizers ────────────────────────────────────────────────────

def test_pos_sequence_transform():
    pytest.importorskip("brill_postaggers")
    t = PosSequenceTransformer("en").fit(_DOCS)
    out = t.transform(["hello there friend"])
    assert len(out) == 1
    # POS sequence has the same token count, all upper-case tags
    assert len(out[0].split()) == 3


def test_word_pos_transform_hybrid_tokens():
    pytest.importorskip("brill_postaggers")
    t = WordPosTransformer("en").fit(_DOCS)
    out = t.transform(["play music"])[0]
    assert "__" in out
    assert all("__" in tok for tok in out.split())


def test_pos_sequence_unsupported_language():
    with pytest.raises(ValueError):
        PosSequenceTransformer("zz").fit(_DOCS)


def test_word_pos_unsupported_language():
    with pytest.raises(ValueError):
        WordPosTransformer("zz").fit(_DOCS)


def test_pos_sequence_pipeline_fits():
    pytest.importorskip("brill_postaggers")
    pipe = Pipeline([("feat", pos_sequence("en")),
                     ("clf", LogisticRegression(max_iter=1000))])
    assert _fit_score(pipe) >= 0.9


def test_word_pos_pipeline_fits():
    pytest.importorskip("brill_postaggers")
    pipe = Pipeline([("feat", word_pos("en")),
                     ("clf", LogisticRegression(max_iter=1000))])
    assert _fit_score(pipe) >= 0.9


# ── stemming ───────────────────────────────────────────────────────────────

def test_stem_transform_collapses_inflections():
    pytest.importorskip("nltk")
    t = StemTransformer("en").fit(_DOCS)
    assert t.transform(["running runs runner"])[0].split()[0] == \
        t.transform(["runs"])[0]


def test_stem_unsupported_language():
    with pytest.raises(ValueError):
        StemTransformer("zz").fit(_DOCS)


def test_stemmed_tfidf_pipeline_fits():
    pytest.importorskip("nltk")
    pipe = Pipeline([("feat", stemmed_tfidf("en")),
                     ("clf", LogisticRegression(max_iter=1000))])
    assert _fit_score(pipe) >= 0.9


# ── lemmatisation ──────────────────────────────────────────────────────────

def test_lemma_transform():
    pytest.importorskip("simplemma")
    t = LemmaTransformer("en").fit(_DOCS)
    out = t.transform(["plays"])[0]
    assert out == "play"


def test_lemmatized_tfidf_pipeline_fits():
    pytest.importorskip("simplemma")
    pipe = Pipeline([("feat", lemmatized_tfidf("en")),
                     ("clf", LogisticRegression(max_iter=1000))])
    assert _fit_score(pipe) >= 0.9


# ── composition with feature_union ─────────────────────────────────────────

def test_pos_sequence_composes_with_feature_union():
    pytest.importorskip("brill_postaggers")
    from jurebes.featurizers import feature_union
    union = feature_union(tfidf_word(), pos_sequence("en"))
    pipe = Pipeline([("feat", union),
                     ("clf", LogisticRegression(max_iter=1000))])
    assert _fit_score(pipe) >= 0.9
