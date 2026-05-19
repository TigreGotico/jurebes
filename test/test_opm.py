import pytest

pytest.importorskip("ovos_plugin_manager")

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from jurebes.opm import JurebesPipeline


def test_opm_register_and_match():
    bus = FakeBus()
    pipe = JurebesPipeline(bus=bus, config={"baseline": "logreg", "enable_slots": False})

    bus.emit(Message("padatious:register_intent", {
        "name": "skill.hello:hello",
        "lang": "en-US",
        "samples": ["hello", "hi", "hey there", "hello friend", "hi friend"],
    }))
    bus.emit(Message("padatious:register_intent", {
        "name": "skill.joke:joke",
        "lang": "en-US",
        "samples": ["tell me a joke", "say a joke", "make me laugh", "tell joke"],
    }))
    bus.emit(Message("padatious:register_intent", {
        "name": "skill.name:name",
        "lang": "en-US",
        "samples": ["what is your name", "who are you", "tell me your name"],
    }))

    match = pipe.match_low(["tell me a joke please"], "en-US", Message("test"))
    assert match is not None
    assert match.match_type == "skill.joke:joke"


def test_opm_exact_match():
    bus = FakeBus()
    pipe = JurebesPipeline(bus=bus, config={"baseline": "logreg", "enable_slots": False})
    bus.emit(Message("padatious:register_intent", {
        "name": "skill.hello:hello",
        "lang": "en-US",
        "samples": ["hello", "hi"],
    }))
    bus.emit(Message("padatious:register_intent", {
        "name": "skill.joke:joke",
        "lang": "en-US",
        "samples": ["tell joke", "say joke"],
    }))
    match = pipe.match_high(["hello"], "en-US", Message("test"))
    assert match is not None
    assert match.match_type == "skill.hello:hello"
