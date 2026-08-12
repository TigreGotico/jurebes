"""Shared fixtures for jurebes end-to-end tests.

These tests are skipped if `ovoscope` is not installed. Install with::

    pip install jurebes[e2e]
"""

from __future__ import annotations

import pytest

ovoscope = pytest.importorskip("ovoscope")


JUREBES_PIPELINE = [
    "ovos-jurebes-pipeline-plugin-high",
    "ovos-jurebes-pipeline-plugin-medium",
    "ovos-jurebes-pipeline-plugin-low",
]


def _jurebes_pipeline_config(**overrides):
    cfg = {
        "baseline": "logreg",
        "enable_slots": False,
        "conf_high": 0.8,
        "conf_med": 0.5,
        "conf_low": 0.2,
    }
    cfg.update(overrides)
    return {"ovos-jurebes-pipeline-plugin": cfg}


@pytest.fixture
def jurebes_pipeline_ids():
    return list(JUREBES_PIPELINE)


@pytest.fixture
def jurebes_pipeline_config():
    return _jurebes_pipeline_config()


@pytest.fixture
def empty_minicroft(jurebes_pipeline_ids, jurebes_pipeline_config):
    """A MiniCroft with no skills and the Jurebes pipeline wired in."""
    croft = ovoscope.get_minicroft(
        skill_ids=[],
        default_pipeline=jurebes_pipeline_ids,
        pipeline_config=jurebes_pipeline_config,
    )
    yield croft
    try:
        croft.stop()
    except Exception:
        pass


@pytest.fixture
def hello_minicroft(jurebes_pipeline_ids, jurebes_pipeline_config):
    """A MiniCroft with the hello-world skill loaded behind the Jurebes pipeline."""
    croft = ovoscope.get_minicroft(
        skill_ids=["ovos-skill-hello-world.openvoiceos"],
        default_pipeline=jurebes_pipeline_ids,
        pipeline_config=jurebes_pipeline_config,
        secondary_langs=["pt-PT"],
    )
    yield croft
    try:
        croft.stop()
    except Exception:
        pass
