"""Smoke test: Jurebes pipeline plugin loads inside MiniCroft."""

from __future__ import annotations

import pytest

ovoscope = pytest.importorskip("ovoscope")


def test_jurebes_pipeline_registered(empty_minicroft, jurebes_pipeline_ids):
    """Pipeline plugin loads at high/medium/low confidence stages."""
    croft = empty_minicroft
    assert croft is not None
    # IntentService keeps a registry of loaded pipeline matchers
    intent_service = croft.intents
    pipeline_plugins = getattr(intent_service, "pipeline_plugins", {})
    loaded_keys = list(pipeline_plugins.keys()) if isinstance(pipeline_plugins, dict) else []
    assert any("jurebes" in str(k).lower() for k in loaded_keys), (
        f"jurebes pipeline plugin not loaded, got: {loaded_keys}"
    )
