"""End-to-end checks that OVOS template syntax is expanded when
intents are added via :class:`jurebes.core.IntentClassifier`.

* ``(a|b)``    — alternatives
* ``[opt]``    — optional segments
* ``{slot}``   — placeholders preserved verbatim
"""

from jurebes.baselines import BASELINES
from jurebes.core import IntentClassifier


def _stored(clf: IntentClassifier, label: str):
    # IntentClassifier stores per-label samples in ``_samples``.
    return list(clf._samples[label])


def test_alternatives_expanded_in_add_intent():
    clf = IntentClassifier(BASELINES.build("logreg"))
    clf.add_intent("greet", ["(hello|hi|hey) there"])
    samples = _stored(clf, "greet")
    for variant in ("hello there", "hi there", "hey there"):
        assert variant in samples, (variant, samples)


def test_optionals_expanded_in_add_intent():
    clf = IntentClassifier(BASELINES.build("logreg"))
    clf.add_intent("lights_on", ["turn [the] lights on"])
    samples = _stored(clf, "lights_on")
    assert "turn the lights on" in samples
    assert "turn lights on" in samples
    # whitespace is collapsed
    assert "turn  lights on" not in samples


def test_slot_placeholders_preserved():
    clf = IntentClassifier(BASELINES.build("logreg"))
    clf.add_intent("introduce", ["(my name is|call me) {name}"])
    samples = _stored(clf, "introduce")
    assert "my name is {name}" in samples
    assert "call me {name}" in samples


def test_combined_alternatives_optionals_and_slots():
    clf = IntentClassifier(BASELINES.build("logreg"))
    clf.add_intent("play", ["(play|put on) [the song] {title}"])
    samples = _stored(clf, "play")
    for variant in (
        "play the song {title}",
        "play {title}",
        "put on the song {title}",
        "put on {title}",
    ):
        assert variant in samples, (variant, samples)


def test_plain_samples_pass_through():
    clf = IntentClassifier(BASELINES.build("logreg"))
    clf.add_intent("bye", ["see you later", "goodbye"])
    samples = _stored(clf, "bye")
    assert samples == ["see you later", "goodbye"]
