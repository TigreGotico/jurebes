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


def test_opm_skill_id_from_message_data():
    bus = FakeBus()
    pipe = JurebesPipeline(bus=bus, config={"baseline": "logreg", "enable_slots": False})
    bus.emit(Message("padatious:register_intent", {
        "name": "some_intent_no_colon",
        "skill_id": "skill.custom",
        "lang": "en-US",
        "samples": ["hello", "hi", "hey there", "hello friend"],
    }))
    bus.emit(Message("padatious:register_intent", {
        "name": "skill.joke:joke",
        "lang": "en-US",
        "samples": ["tell me a joke", "say a joke"],
    }))
    match = pipe.match_low(["hello there"], "en-US", Message("test"))
    assert match is not None
    assert match.skill_id == "skill.custom"


def test_opm_unicode_samples_file(tmp_path):
    bus = FakeBus()
    pipe = JurebesPipeline(bus=bus, config={"baseline": "logreg", "enable_slots": False})
    f1 = tmp_path / "greet.intent"
    f1.write_text("olá\nbão dia\ncomo está você\n", encoding="utf-8")
    bus.emit(Message("padatious:register_intent", {
        "name": "skill.greet:greet",
        "lang": "en-US",
        "file_name": str(f1),
    }))
    bus.emit(Message("padatious:register_intent", {
        "name": "skill.joke:joke",
        "lang": "en-US",
        "samples": ["tell me a joke", "say a joke"],
    }))
    assert "skill.greet:greet" in pipe.registered_intents
    match = pipe.match_low(["olá"], "en-US", Message("test"))
    assert match is not None


def test_opm_detach_intent_after_match():
    bus = FakeBus()
    pipe = JurebesPipeline(bus=bus, config={"baseline": "logreg", "enable_slots": False})
    bus.emit(Message("padatious:register_intent", {
        "name": "skill.hello:hello",
        "lang": "en-US",
        "samples": ["hello", "hi", "hey there", "hello friend"],
    }))
    bus.emit(Message("padatious:register_intent", {
        "name": "skill.joke:joke",
        "lang": "en-US",
        "samples": ["tell me a joke", "say a joke", "make me laugh"],
    }))
    m = pipe.match_low(["hello there"], "en-US", Message("test"))
    assert m is not None and m.match_type == "skill.hello:hello"
    bus.emit(Message("detach_intent", {"intent_name": "skill.hello:hello"}))
    assert "skill.hello:hello" not in pipe.registered_intents
    # _exact cache should have dropped any keys pointing to the detached intent
    assert "skill.hello:hello" not in pipe._exact.values()


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
