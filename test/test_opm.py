from unittest.mock import patch

import pytest

pytest.importorskip("ovos_plugin_manager")

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

import jurebes.opm as opm_mod
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


def _register_hello_and_joke(pipe, bus):
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


def test_opm_exact_match_default_skips_classifier():
    # default behavior (exact_match unset -> True): classifier must NOT be
    # consulted when a registered training utterance has an exact hit.
    bus = FakeBus()
    pipe = JurebesPipeline(bus=bus, config={"baseline": "logreg", "enable_slots": False})
    _register_hello_and_joke(pipe, bus)

    with patch.object(opm_mod, "_calc_jurebes", wraps=opm_mod._calc_jurebes) as spy:
        match = pipe.match_high(["hello"], "en-US", Message("test"))
        assert match is not None
        assert match.match_type == "skill.hello:hello"
        spy.assert_not_called()


def test_opm_exact_match_false_always_consults_classifier():
    # with exact_match=False, the classifier must be consulted even for a
    # registered training utterance that would otherwise be an exact hit.
    bus = FakeBus()
    pipe = JurebesPipeline(bus=bus, config={"baseline": "logreg", "enable_slots": False,
                                             "exact_match": False})
    _register_hello_and_joke(pipe, bus)
    assert pipe.exact_match is False

    with patch.object(opm_mod, "_calc_jurebes", wraps=opm_mod._calc_jurebes) as spy:
        match = pipe.match_low(["hello"], "en-US", Message("test"))
        assert match is not None
        assert match.match_type == "skill.hello:hello"
        spy.assert_called()


def test_opm_initial_train_failure_logs_once_and_falls_back_to_no_match():
    # Regression test for the arena-observed bug: when the underlying
    # classifier raises during initial train (e.g. NuSVC "specified nu is
    # infeasible" on imbalanced ca-ES data), the pipeline must not keep
    # retrying the fit and logging "classifier not fitted" per-inference.
    # It must log the failure exactly once, then serve a clean no-match
    # for anything that needs the classifier — WITHOUT raising, since a
    # raise here would also break exact matching for this lang (verified:
    # exact matches must keep working even though the classifier itself is
    # unusable).
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
        "samples": ["tell me a joke", "say a joke"],
    }))

    clf = pipe.containers["en-US"]
    with patch.object(clf, "fit", side_effect=ValueError("specified nu is infeasible")) as mock_fit, \
            patch.object(opm_mod.LOG, "error") as mock_log_error:
        pipe.handle_initial_train(Message("mycroft.ready"))
        assert mock_fit.call_count == 1
        assert mock_log_error.call_count == 1
        assert "en-US" in pipe._train_error
        assert pipe._fitted["en-US"] is False

        # A non-exact utterance needs the (unusable) classifier: clean
        # no-match, no raise.
        assert pipe.calc_intent(["completely unrelated gibberish utterance"], "en-US", Message("test")) is None

        # An exact registered utterance must still match at full confidence
        # even though the classifier for this lang failed to train — exact
        # matching does not need a fitted classifier.
        match = pipe.calc_intent(["hello"], "en-US", Message("test"))
        assert match is not None
        assert match.intent == "skill.hello:hello"
        assert match.confidence == 1.0

        # Neither call may retry the deterministic training failure or add
        # further error-log spam beyond the single initial-train failure log.
        assert mock_fit.call_count == 1
        assert mock_log_error.call_count == 1


def test_opm_not_enough_intents_is_not_a_training_failure():
    # "need at least 2 intent classes to fit" is the normal not-ready state
    # at startup (fires in nearly every real install before a second skill
    # has registered its intents yet) — it must NOT be recorded as a
    # training failure, and must not be logged at ERROR level.
    bus = FakeBus()
    pipe = JurebesPipeline(bus=bus, config={"baseline": "logreg", "enable_slots": False})
    bus.emit(Message("padatious:register_intent", {
        "name": "skill.hello:hello",
        "lang": "en-US",
        "samples": ["hello", "hi", "hey there", "hello friend"],
    }))

    with patch.object(opm_mod.LOG, "error") as mock_log_error:
        pipe.handle_initial_train(Message("mycroft.ready"))
        assert pipe._fitted["en-US"] is False
        assert "en-US" not in pipe._train_error
        mock_log_error.assert_not_called()
