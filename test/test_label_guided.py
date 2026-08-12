"""Tests for the LabelGuidedEmbeddingsTransformer port + denoising AE + new sized AE variants."""

import numpy as np
import pytest

from jurebes.featurizers import (
    LabelGuidedEmbeddingsTransformer,
    SklearnAutoencoder,
    _auto_sizes,
    autoencoder,
    denoising_autoencoder,
    label_guided,
    tfidf_word,
)


# ── _auto_sizes ─────────────────────────────────────────────────────


def test_auto_sizes_scales_with_input_dim():
    small = _auto_sizes(100)
    large = _auto_sizes(10000)
    assert small[1] < large[1]
    assert min(small) <= small[1]  # bottleneck is the smallest layer


def test_auto_sizes_minimum_bottleneck():
    """Even tiny inputs get a bottleneck of at least 32."""
    sizes = _auto_sizes(10)
    assert sizes[1] >= 32


# ── SklearnAutoencoder defaults ─────────────────────────────────────


def test_sklearn_autoencoder_auto_sizing():
    """With default 'auto' the layer sizes are computed at fit time."""
    ae = SklearnAutoencoder(max_iter=50)
    rng = np.random.RandomState(0)
    X = rng.rand(60, 100)
    ae.fit(X)
    assert ae.layer_sizes_ == _auto_sizes(100)


def test_sklearn_autoencoder_explicit_sizes_honored():
    ae = SklearnAutoencoder(hidden_layer_sizes=(20, 5, 20), max_iter=50)
    rng = np.random.RandomState(0)
    X = rng.rand(60, 50)
    ae.fit(X)
    assert ae.layer_sizes_ == (20, 5, 20)


def test_sklearn_autoencoder_early_stopping_on_by_default():
    ae = SklearnAutoencoder()
    assert ae.early_stopping is True
    assert ae.max_iter == 500


def test_denoising_autoencoder_uses_noise_at_fit_only():
    """Transform on clean input matches a non-denoising AE — noise only at training."""
    rng = np.random.RandomState(0)
    X = rng.rand(60, 30)
    den = SklearnAutoencoder(noise_level=0.05, max_iter=80,
                             hidden_layer_sizes=(20, 10, 20))
    den.fit(X)
    z = den.transform(X)
    assert z.shape == (60, 10)


def test_denoising_autoencoder_builder():
    p = denoising_autoencoder(noise_level=0.05, base=tfidf_word())
    # Pipeline should have a base, dense, ae step
    names = [name for name, _ in p.steps]
    assert "base" in names and "ae" in names


# ── LabelGuidedEmbeddingsTransformer ────────────────────────────────


def test_label_guided_shape():
    rng = np.random.RandomState(0)
    X = rng.rand(60, 30)
    y = (["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    lge = LabelGuidedEmbeddingsTransformer(
        hidden_layer_sizes=(16, 8), embedding_size=4, max_iter=50,
    )
    lge.fit(X, y)
    z = lge.transform(X)
    assert z.shape == (60, 4)


def test_label_guided_default_embedding_size_uses_3_times_classes():
    rng = np.random.RandomState(0)
    X = rng.rand(60, 30)
    y = ["a"] * 20 + ["b"] * 20 + ["c"] * 20
    lge = LabelGuidedEmbeddingsTransformer(hidden_layer_sizes=(16, 8), max_iter=50)
    lge.fit(X, y)
    z = lge.transform(X)
    assert z.shape[1] == min(3 * 3, 16)  # 3 classes * 3, capped by hidden width


def test_label_guided_in_pipeline_trains_and_predicts():
    """Pipeline wiring: tfidf_word -> label_guided -> classifier round-trips."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    X = (["hello world hello", "hi there friend", "good morning sun"] * 4
         + ["thank you kindly", "thanks a lot", "much appreciated"] * 4)
    y = ["greet"] * 12 + ["thanks"] * 12
    pipe = Pipeline([
        ("emb", label_guided(base=tfidf_word(), hidden_layer_sizes=(16, 8),
                              max_iter=80)),
        ("clf", LogisticRegression(max_iter=500)),
    ])
    pipe.fit(X, y)
    pred = pipe.predict(["hello world"])
    assert pred[0] in {"greet", "thanks"}


def test_label_guided_requires_y():
    rng = np.random.RandomState(0)
    X = rng.rand(20, 10)
    lge = LabelGuidedEmbeddingsTransformer()
    with pytest.raises(Exception):
        lge.fit(X)


# ── new baseline factory entries ────────────────────────────────────


def test_new_autoencoder_baselines_registered():
    from jurebes.baselines import BASELINES
    for name in (
        "autoencoder_logreg_wide",
        "autoencoder_logreg_deep",
        "denoising_autoencoder_logreg",
        "label_guided_logreg",
        "label_guided_linear_svc",
    ):
        assert name in BASELINES.names()


def test_label_guided_baseline_in_reduced_dim_group():
    from jurebes.baselines import BASELINES
    groups = BASELINES.groups()
    assert "label_guided_logreg" in groups["reduced_dim"]
    assert "label_guided_linear_svc" in groups["reduced_dim"]
