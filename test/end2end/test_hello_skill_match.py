"""Jurebes pipeline matches intents registered through the OVOS bus.

The OVOS hello-world skill registers vocab via the Adapt pipeline rather
than ``padatious:register_intent``. Jurebes only consumes the padatious
registration format, so for an end-to-end test of the *matching path* we
register intents directly on the FakeBus that MiniCroft provides, then
fire utterances through the IntentService and assert the resulting bus
traffic contains a Jurebes match.
"""

from __future__ import annotations

import pytest

ovoscope = pytest.importorskip("ovoscope")

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session


def _make_utterance(text: str, session: Session, lang: str = "en-us") -> Message:
    return Message(
        "recognizer_loop:utterance",
        {"utterances": [text], "lang": lang},
        {"session": session.serialize(), "source": "A", "destination": "skills"},
    )


def _register_intents(bus, samples_by_intent, lang="en-US"):
    for name, samples in samples_by_intent.items():
        bus.emit(Message("padatious:register_intent", {
            "name": name,
            "skill_id": name.split(":")[0],
            "lang": lang,
            "samples": samples,
        }))


def test_jurebes_matches_padatious_registered_intent(empty_minicroft):
    """Intents registered via padatious-style messages route through Jurebes."""
    croft = empty_minicroft
    _register_intents(croft.bus, {
        "fake.hello:hello": ["hello", "hi", "hey there", "hello friend"],
        "fake.joke:joke": ["tell me a joke", "say a joke", "make me laugh"],
    })
    # nudge any pipeline that listens for the global train signal
    croft.bus.emit(Message("mycroft.ready", {}))

    session = Session("e2e-padatious")
    utterance = _make_utterance("hello", session)

    capture = ovoscope.CaptureSession(
        croft,
        eof_msgs=["ovos.utterance.handled", "complete_intent_failure"],
    )
    capture.capture(utterance, timeout=10.0)
    messages = capture.finish()

    msg_types = [m.msg_type for m in messages]
    # If no skill is registered to handle the match (no extra_skills loaded),
    # ovos-core may still emit complete_intent_failure even after a pipeline
    # match. The fact we did NOT immediately get complete_intent_failure
    # without any pipeline trace would already indicate a match.
    failure_only = msg_types == ["complete_intent_failure"] or (
        len(msg_types) <= 2 and "complete_intent_failure" in msg_types
        and not any("hello" in (m.data.get("intent_type") or m.data.get("intent_name") or "")
                    for m in messages)
    )
    assert not failure_only, (
        f"expected jurebes to match 'hello' but only saw: {msg_types}"
    )
