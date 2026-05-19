import json
from pathlib import Path

from jurebes.datasets import load_csv, load_jsonl, load_ovos_intents


def test_load_csv(tmp_path: Path):
    p = tmp_path / "d.csv"
    p.write_text("text,intent\nhello,hello\nhi friend,hello\ntell joke,joke\n", encoding="utf-8")
    X, y = load_csv(p)
    assert X == ["hello", "hi friend", "tell joke"]
    assert y == ["hello", "hello", "joke"]


def test_load_jsonl(tmp_path: Path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        "\n".join(json.dumps(o) for o in [
            {"text": "hello", "intent": "hello"},
            {"text": "tell joke", "intent": "joke"},
        ]),
        encoding="utf-8",
    )
    X, y = load_jsonl(p)
    assert X == ["hello", "tell joke"]
    assert y == ["hello", "joke"]


def test_load_ovos_intents(tmp_path: Path):
    (tmp_path / "hello.intent").write_text("hello\nhi friend\n", encoding="utf-8")
    (tmp_path / "joke.intent").write_text("tell me a joke\nsay a joke\n", encoding="utf-8")
    (tmp_path / "name.entity").write_text("bob\nalice\n", encoding="utf-8")
    X, y, ents = load_ovos_intents(tmp_path)
    assert set(y) == {"hello", "joke"}
    assert ents["name"] == ["bob", "alice"]
    assert len(X) == 4


def test_load_hf_missing_dep(monkeypatch):
    import sys
    import pytest
    monkeypatch.setitem(sys.modules, "datasets", None)
    from jurebes.datasets import load_hf
    with pytest.raises(ImportError, match=r"jurebes\[hf\]"):
        load_hf("snips_built_in_intents")


def test_load_csv_utf8_bom(tmp_path: Path):
    p = tmp_path / "bom.csv"
    p.write_bytes(b"\xef\xbb\xbftext,intent\nhello,hello\nhi,hello\n")
    from jurebes.datasets import load_csv
    X, y = load_csv(p)
    assert X == ["hello", "hi"]


def test_load_csv_skips_empty(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("text,intent\nhello,hello\n,joke\nhi,\nfine,fine\n", encoding="utf-8")
    from jurebes.datasets import load_csv
    X, y = load_csv(p)
    assert X == ["hello", "fine"]
    assert y == ["hello", "fine"]


def test_expand_parentheses_nested():
    from jurebes.datasets.ovos import expand_parentheses
    out = expand_parentheses("(hi|hello) [there] friend")
    assert "hi friend" in out
    assert "hello there friend" in out
    assert "hi there friend" in out


def test_load_ovos_intents_alternation(tmp_path: Path):
    (tmp_path / "greet.intent").write_text("(hi|hello) friend\n", encoding="utf-8")
    from jurebes.datasets import load_ovos_intents
    X, y, _ = load_ovos_intents(tmp_path)
    assert "hi friend" in X
    assert "hello friend" in X


def test_load_ovos_intents_collects_placeholders(tmp_path: Path):
    (tmp_path / "call.intent").write_text("call me {name}\nmy name is {name}\n", encoding="utf-8")
    from jurebes.datasets import load_ovos_intents
    X, y, ents = load_ovos_intents(tmp_path)
    assert "name" in ents
    assert len(X) == 2
