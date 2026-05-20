"""Tests for jurebes.datasets.expansion."""

from jurebes.datasets import expand_slots, expand_template


def test_expand_template_no_brackets():
    assert expand_template("hello world") == ["hello world"]


def test_expand_template_alternation():
    out = expand_template("(hello|hi) there")
    assert set(out) == {"hello there", "hi there"}


def test_expand_template_optional():
    out = expand_template("hello [there]")
    assert set(out) == {"hello", "hello there"}


def test_expand_template_combined():
    out = expand_template("(hello|hi) [there] friend")
    assert set(out) == {
        "hello friend", "hi friend",
        "hello there friend", "hi there friend",
    }


def test_expand_template_nested_alternations():
    out = expand_template("(hi|hello) (friend|world)")
    assert set(out) == {"hi friend", "hi world", "hello friend", "hello world"}


def test_expand_slots_basic():
    out = expand_slots("play {song}", {"song": ["a", "b"]})
    assert set(out) == {"play a", "play b"}


def test_expand_slots_cartesian():
    out = expand_slots("play {song} by {artist}",
                       {"song": ["a", "b"], "artist": ["x", "y"]})
    assert set(out) == {"play a by x", "play a by y", "play b by x", "play b by y"}


def test_expand_slots_unknown_slot_kept():
    out = expand_slots("play {song}", {})
    assert out == ["play {song}"]


def test_expand_slots_with_alternation():
    out = expand_slots("(play|put on) {song}", {"song": ["a"]})
    assert set(out) == {"play a", "put on a"}


def test_expand_slots_with_optional_and_alternation():
    out = expand_slots(
        "[please] (play|put on) {song}",
        {"song": ["a", "b"]},
    )
    assert "play a" in out
    assert "please put on b" in out
    assert "put on a" in out
