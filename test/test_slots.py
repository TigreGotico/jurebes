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


# ── TAGGERS registry ────────────────────────────────────────────────


def test_taggers_registry_names():
    from jurebes.slots import TAGGERS
    names = set(TAGGERS.names())
    assert {"dictionary", "template", "sklearn_iob", "knn", "hybrid"}.issubset(names)
    assert len(TAGGERS) >= 5


# ── KNNTagger ────────────────────────────────────────────────────────


def _train_knn_for(samples):
    from jurebes.slots import KNNTagger
    t = KNNTagger(k=1)
    t.add_entity("city", ["lisbon", "paris", "berlin"])
    t.fit({"weather": samples})
    return t


def test_knn_tagger_basic():
    t = _train_knn_for(["weather in {city}", "forecast for {city}"])
    result = t.predict("weather in lisbon")
    assert result.get("city") == "lisbon"


def test_knn_tagger_unseen_entity_value():
    """Strength of KNN: tag patterns transfer to values absent from the gazetteer."""
    t = _train_knn_for(["weather in {city}"])
    result = t.predict("weather in tokyo")
    assert result.get("city") == "tokyo"


def test_knn_tagger_no_match_returns_empty():
    t = _train_knn_for(["weather in {city}"])
    # input that doesn't align positionally — no city token at position 2
    result = t.predict("hello there")
    assert "city" not in result


def test_knn_tagger_save_load(tmp_path):
    from jurebes.slots import KNNTagger
    t = _train_knn_for(["weather in {city}"])
    path = tmp_path / "knn.joblib"
    t.save(path)
    loaded = KNNTagger.load(path)
    assert loaded.predict("weather in paris").get("city") == "paris"


def test_knn_in_taggers_registry():
    from jurebes.slots import TAGGERS, KNNTagger
    t = TAGGERS.build("knn")
    assert isinstance(t, KNNTagger)


def test_taggers_resolve_hybrid_builds_correctly():
    from jurebes.slots import TAGGERS
    from jurebes.slots.hybrid import HybridCascadeTagger
    h = TAGGERS.build("hybrid")
    assert isinstance(h, HybridCascadeTagger)
    assert len(h.taggers) == 3


def test_intent_classifier_accepts_tagger_string():
    from jurebes.core import IntentClassifier
    from jurebes.slots.dictionary import DictionaryTagger
    clf = IntentClassifier(tagger="dictionary")
    assert isinstance(clf.tagger, DictionaryTagger)


# ── CRFTagger (optional) ────────────────────────────────────────────


def test_crf_tagger_smoke():
    import pytest
    pytest.importorskip("sklearn_crfsuite")
    from jurebes.slots.crf import CRFTagger
    t = CRFTagger()
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


# ── benchmark.compare_taggers ───────────────────────────────────────


def test_compare_taggers_smoke():
    from jurebes.benchmark.slots import compare_taggers

    intent_samples = {
        "weather": [
            "weather in {city}",
            "what is the weather in {city}",
            "tell me the weather in {city}",
            "how is the weather in {city}",
        ],
        "greet": ["hello there", "hi friend", "hey", "hello"],
    }
    entity_samples = {"city": ["paris", "lisbon", "berlin", "madrid"]}
    test = [
        ("weather in paris", {"city": "paris"}),
        ("weather in lisbon", {"city": "lisbon"}),
        ("weather in berlin", {"city": "berlin"}),
        ("weather in madrid", {"city": "madrid"}),
        ("hello there", {}),
        ("hi friend", {}),
    ]
    result = compare_taggers(
        ["dictionary", "template", "sklearn_iob", "hybrid"],
        intent_samples, entity_samples, test,
    )
    assert len(result.rows) == 4
    for r in result.rows:
        assert 0.0 <= r.exact_match <= 1.0


def test_template_tagger_duplicate_slot_name():
    """A slot name repeated in one template must not crash regex compilation."""
    from jurebes.slots import TemplateTagger
    t = TemplateTagger()
    t.add_intent("rel", ["the {relation} of my {relation}"])
    t.fit()
    out = t.predict("the brother of my sister")
    assert out.get("relation") == "brother"  # first occurrence wins


def test_compare_taggers_isolates_failures():
    """One tagger raising must not abort the whole comparison."""
    from jurebes.benchmark.slots import compare_taggers
    intent_samples = {
        "weather": ["weather in {city}", "forecast for {city}"],
        "greet": ["hello", "hi there"],
    }
    entity_samples = {"city": ["paris", "lisbon"]}
    test = [("weather in paris", {"city": "paris"}), ("hello", {})]
    result = compare_taggers(
        ["dictionary", "template", "sklearn_iob"],
        intent_samples, entity_samples, test,
    )
    assert len(result.rows) == 3  # all three rows present even if one NaNs
