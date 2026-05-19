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
