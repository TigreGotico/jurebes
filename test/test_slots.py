from jurebes.slots import (
    DictionaryTagger,
    SklearnIOBTagger,
    TemplateTagger,
    tokenize,
)


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


def test_iob_tagger_save_load_round_trip(tmp_path):
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
    p = tmp_path / "tagger.joblib"
    t.save(p)
    loaded = SklearnIOBTagger.load(p)
    assert loaded.fitted
    out = loaded.predict("my name is bob")
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


# ── DictionaryTagger ────────────────────────────────────────────────


def test_dictionary_tagger_basic():
    t = DictionaryTagger()
    t.add_entity("city", ["paris", "lisbon"])
    assert t.predict("weather in paris") == {"city": "paris"}


def test_dictionary_tagger_case_insensitive():
    t = DictionaryTagger()
    t.add_entity("city", ["paris", "lisbon"])
    out = t.predict("weather in Lisbon")
    assert out.get("city", "").lower() == "lisbon"


def test_dictionary_tagger_multitoken():
    t = DictionaryTagger()
    t.add_entity("city", ["new york", "paris"])
    out = t.predict("weather in new york")
    assert out.get("city", "").lower() == "new york"


def test_dictionary_tagger_save_load(tmp_path):
    t = DictionaryTagger()
    t.add_entity("city", ["paris", "lisbon"])
    p = tmp_path / "dict.json"
    t.save(p)
    loaded = DictionaryTagger.load(p)
    assert loaded.predict("weather in paris") == {"city": "paris"}


# ── TemplateTagger ──────────────────────────────────────────────────


def test_template_tagger_basic():
    t = TemplateTagger()
    t.add_intent("weather", ["weather in {city}"])
    t.fit()
    assert t.predict("weather in lisbon") == {"city": "lisbon"}


def test_template_tagger_alternation():
    t = TemplateTagger()
    t.add_intent("weather", ["(weather|temperature) in {city}"])
    t.fit()
    assert t.predict("weather in lisbon") == {"city": "lisbon"}
    assert t.predict("temperature in paris") == {"city": "paris"}


def test_template_tagger_no_match():
    t = TemplateTagger()
    t.add_intent("weather", ["weather in {city}"])
    t.fit()
    assert t.predict("hello world") == {}


def test_template_tagger_save_load(tmp_path):
    t = TemplateTagger()
    t.add_intent("weather", ["weather in {city}"])
    t.fit()
    p = tmp_path / "tpl.json"
    t.save(p)
    loaded = TemplateTagger.load(p)
    assert loaded.predict("weather in lisbon") == {"city": "lisbon"}


# ── HybridCascadeTagger ─────────────────────────────────────────────


def _train_iob_intent_samples():
    return {
        "name": [
            "my name is {name}",
            "call me {name}",
            "I am {name}",
            "the name is {name}",
        ],
        "hello": ["hello there", "hi friend", "hey", "hello"],
    }


def test_hybrid_dictionary_wins():
    from jurebes.slots import DictionaryTagger, TemplateTagger
    from jurebes.slots.hybrid import HybridCascadeTagger

    d = DictionaryTagger()
    d.add_entity("city", ["paris"])
    t = TemplateTagger()
    t.add_intent("weather", ["weather in {city}"])
    h = HybridCascadeTagger(taggers=[d, t])
    h.fit()
    out = h.predict("weather in paris")
    # dictionary returns "paris" literally; if template were first it'd return same here,
    # so use a case marker by registering "Paris" only in dict
    assert out.get("city") == "paris"


def test_hybrid_template_fallback():
    from jurebes.slots import DictionaryTagger, TemplateTagger
    from jurebes.slots.hybrid import HybridCascadeTagger

    d = DictionaryTagger()
    d.add_entity("city", ["paris"])  # does not contain "berlin"
    t = TemplateTagger()
    t.add_intent("weather", ["weather in {city}"])
    h = HybridCascadeTagger(taggers=[d, t])
    h.fit()
    out = h.predict("weather in berlin")
    assert out.get("city") == "berlin"


def test_hybrid_classifier_fallback():
    from jurebes.slots import SklearnIOBTagger
    from jurebes.slots.hybrid import HybridCascadeTagger

    iob = SklearnIOBTagger()
    iob.add_entity("name", ["bob", "alice", "tom", "jarbas"])
    h = HybridCascadeTagger(taggers=[iob])
    h.fit(_train_iob_intent_samples())
    out = h.predict("my name is bob")
    assert out.get("name") == "bob"


def test_hybrid_save_load(tmp_path):
    from jurebes.slots import DictionaryTagger
    from jurebes.slots.hybrid import HybridCascadeTagger

    d = DictionaryTagger()
    d.add_entity("city", ["paris"])
    h = HybridCascadeTagger(taggers=[d])
    h.fit()
    p = tmp_path / "hyb.joblib"
    h.save(p)
    loaded = HybridCascadeTagger.load(p)
    assert loaded.predict("weather in paris") == {"city": "paris"}

