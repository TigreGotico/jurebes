"""Session-level controls applied to Jurebes pipeline matches."""

from __future__ import annotations

import pytest

ovoscope = pytest.importorskip("ovoscope")

from ovos_bus_client.message import Message

from jurebes.opm import JurebesPipeline


def test_pipeline_handles_session_blacklist():
    """A blacklisted intent must not be matched by Jurebes."""
    from ovos_utils.fakebus import FakeBus

    bus = FakeBus()
    pipe = JurebesPipeline(bus=bus, config={"baseline": "logreg", "enable_slots": False})

    bus.emit(Message("padatious:register_intent", {
        "name": "skill.hello:hello",
        "lang": "en-US",
        "samples": ["hello", "hi", "hey there", "hello friend"],
    }))
    bus.emit(Message("padatious:register_intent", {
        "name": "skill.bye:bye",
        "lang": "en-US",
        "samples": ["goodbye", "see you", "bye now"],
    }))

    # baseline match works
    ctx_msg = Message("recognizer_loop:utterance", {"utterances": ["hello"]})
    assert pipe.match_low(["hello"], "en-US", ctx_msg) is not None

    # now blacklist the intent in the session
    from ovos_bus_client.session import Session

    session = Session("blocked")
    session.blacklisted_intents = ["skill.hello:hello"]
    blocked_msg = Message(
        "recognizer_loop:utterance",
        {"utterances": ["hello"]},
        {"session": session.serialize()},
    )
    assert pipe.match_low(["hello"], "en-US", blocked_msg) is None
