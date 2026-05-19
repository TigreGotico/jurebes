"""Smoke tests for JurebesIntentContainer and the OPM pipeline wrapper."""

import pytest

from jurebes import JurebesIntentContainer
from jurebes.opm import JurebesPipeline


@pytest.fixture
def trained_container():
    engine = JurebesIntentContainer()
    engine.add_intent("hello", ["hello", "hi", "hey there"])
    engine.add_intent("bye", ["goodbye", "see you", "bye"])
    engine.add_intent("joke", ["tell me a joke", "say a joke"])
    engine.train()
    return engine


def test_calc_intent_returns_match(trained_container):
    match = trained_container.calc_intent("hello there")
    assert match is not None
    assert match.intent_name in {"hello", "bye", "joke"}
    assert 0.0 <= match.confidence <= 1.0


def test_calc_intents_yields_iterable(trained_container):
    intents = list(trained_container.calc_intents("tell me a joke"))
    assert len(intents) >= 1
    assert any(i.intent_name == "joke" for i in intents)


def test_entity_extraction():
    engine = JurebesIntentContainer()
    engine.add_entity("name", ["bob", "alice", "jarbas"])
    engine.add_intent("name", ["my name is {name}", "call me {name}"])
    engine.add_intent("hello", ["hello", "hi"])
    engine.train()
    match = engine.calc_intent("my name is bob")
    assert match is not None
    assert match.intent_name == "name"


def test_pipeline_instantiates():
    """Pipeline should construct against a FakeBus without errors."""
    from ovos_utils.fakebus import FakeBus
    pipe = JurebesPipeline(bus=FakeBus(), config={})
    assert pipe.lang
    assert pipe.containers
    pipe.shutdown()
