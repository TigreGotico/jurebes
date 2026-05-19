"""detach_skill clears the skill's intents from the Jurebes container."""

from __future__ import annotations

import pytest

ovoscope = pytest.importorskip("ovoscope")

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from jurebes.opm import JurebesPipeline


def test_pipeline_drops_skill_intents_on_detach():
    bus = FakeBus()
    pipe = JurebesPipeline(bus=bus, config={"baseline": "logreg", "enable_slots": False})

    skill_id = "ovos-skill-hello-world.openvoiceos"
    bus.emit(Message("padatious:register_intent", {
        "name": f"{skill_id}:hello",
        "skill_id": skill_id,
        "lang": "en-US",
        "samples": ["hello", "hi", "hey there", "hello friend"],
    }))
    bus.emit(Message("padatious:register_intent", {
        "name": "skill.other:other",
        "lang": "en-US",
        "samples": ["tell me a joke", "say a joke", "make me laugh"],
    }))

    ctx = Message("recognizer_loop:utterance", {"utterances": ["hello"]})
    first = pipe.match_low(["hello"], "en-US", ctx)
    assert first is not None
    assert first.skill_id == skill_id
    assert f"{skill_id}:hello" in pipe.registered_intents

    # detach the skill — drops the skill's intents from the registry
    bus.emit(Message("detach_skill", {"skill_id": skill_id}))

    assert f"{skill_id}:hello" not in pipe.registered_intents
    # other skill intents are still present
    assert "skill.other:other" in pipe.registered_intents
