from jurebes.slots import SklearnIOBTagger, tokenize


def test_tokenize():
    assert tokenize("hello, world!") == ["hello", ",", "world", "!"]


def test_iob_learns_entity():
    t = SklearnIOBTagger()
    t.add_entity("name", ["bob", "alice", "tom", "jarbas"])
    t.fit({
        "name": [
            "my name is {name}",
            "call me {name}",
            "I am {name}",
            "the name is {name}",
        ],
        "hello": ["hello there", "hi friend", "hey", "hello"],
    })
    assert t.fitted
    out = t.predict("my name is bob")
    assert out.get("name") == "bob"


def test_iob_no_entity_in_plain_text():
    t = SklearnIOBTagger()
    t.add_entity("name", ["bob", "alice"])
    t.fit({
        "name": ["my name is {name}", "call me {name}"],
        "hello": ["hello there", "hi friend"],
    })
    out = t.predict("hello there")
    assert "name" not in out
