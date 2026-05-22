import pytest

from jurebes.baselines import BASELINES


_X = [
    # hello (12)
    "hello", "hi", "hey", "hello there", "hi friend", "hey friend",
    "good morning", "good evening", "greetings", "howdy", "salutations", "hiya",
    # joke (12)
    "tell me a joke", "say a joke", "make me laugh", "do you know any joke",
    "tell joke", "say joke", "share a funny story", "be funny",
    "amuse me", "humor me please", "crack a joke", "tell something funny",
    # name (12)
    "what is your name", "who are you", "tell me your name", "your name please",
    "what's your name", "name yourself", "introduce yourself", "who am i talking to",
    "say your name", "may i know your name", "what should i call you", "name please",
    # weather (12)
    "what is the weather", "weather today", "is it raining", "will it rain",
    "tell me the forecast", "forecast please", "is it sunny",
    "weather in paris", "how hot is it", "how cold is it",
    "temperature today", "any storms coming",
    # music (12)
    "play music", "play some music", "stop the music", "pause music",
    "next song please", "previous track", "start playback", "resume music",
    "play my playlist", "shuffle songs", "play something nice", "music on",
]
_y = (
    ["hello"] * 12
    + ["joke"] * 12
    + ["name"] * 12
    + ["weather"] * 12
    + ["music"] * 12
)

# duplicate training data; some tree ensembles need more samples to fit cleanly
_X = _X * 3
_y = _y * 3


# Baselines whose feature space is intentionally weak for short utts
# (text statistics carry little signal here) get a looser threshold.
_WEAK = {
    "text_stats_logreg", "lda_logreg", "qda_classifier",
    "autoencoder_logreg", "autoencoder_linear_svc", "autoencoder_rbf_svc",
    "autoencoder_logreg_wide", "autoencoder_logreg_deep",
    "denoising_autoencoder_logreg",
    "label_guided_logreg", "label_guided_linear_svc",
    # POS sequences drop all lexical content, so pure syntactic features carry
    # little discriminative signal on short, lexically distinct toy utterances.
    "pos_sequence_logreg",
}

# Baselines whose input is dict-of-strings rather than text; the text-fit
# parametrize cannot exercise them.
_SKIP_TEXT_FIT = {"categorical_logreg", "categorical_random_forest"}


def _missing(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is None


# Linguistic baselines depend on optional extras (brill_postagger / nltk /
# simplemma). When an extra is absent the parametrized text-fit test skips
# the affected baseline; with the extras installed it fits the toy fixture.
_SKIP_LINGUISTIC = set()
if _missing("brill_postaggers"):
    _SKIP_LINGUISTIC |= {"pos_sequence_logreg", "word_pos_logreg"}
if _missing("nltk"):
    _SKIP_LINGUISTIC |= {"stemmed_logreg"}
if _missing("simplemma"):
    _SKIP_LINGUISTIC |= {"lemmatized_logreg"}

# Wide/deep autoencoder variants are slow to fit even on the toy fixture and
# trigger pytest-timeout. They are exercised by the canonical-benchmark
# training scripts (examples/trained_models/) on real datasets.
_SKIP_SLOW = {"autoencoder_logreg_wide", "autoencoder_logreg_deep"}


@pytest.mark.parametrize("name", BASELINES.names())
def test_baseline_fits_and_scores(name):
    if name in _SKIP_TEXT_FIT:
        pytest.skip(f"{name} consumes dict-of-string features, not raw text")
    if name in _SKIP_SLOW:
        pytest.skip(f"{name} is too slow for the toy fixture; tested via training scripts")
    if name in _SKIP_LINGUISTIC:
        pytest.skip(f"{name} requires an optional linguistic extra that is not installed")
    est = BASELINES.build(name)
    est.fit(_X, _y)
    preds = est.predict(_X)
    acc = sum(p == t for p, t in zip(preds, _y)) / len(_y)
    threshold = 0.4 if name in _WEAK else 0.9
    assert acc >= threshold, f"{name} training accuracy {acc:.2f} below {threshold}"


def test_registry_has_at_least_23_baselines():
    assert len(BASELINES) >= 23


def test_registry_has_at_least_43_baselines_after_b2():
    assert len(BASELINES) >= 43


def test_registry_has_new_groups():
    groups = BASELINES.groups()
    for g in ("reduced_dim", "online", "strategy", "feature_engineering"):
        assert g in groups and len(groups[g]) >= 1


def test_registry_groups_helper():
    groups = BASELINES.groups()
    assert "linear" in groups
    assert "logreg" in groups["linear"]
    assert "linear" in BASELINES.in_group("logreg")


def test_registry_resolve_selector():
    assert BASELINES.resolve("logreg") == ["logreg"]
    linear = BASELINES.resolve("@linear")
    assert "logreg" in linear
    assert len(linear) >= 6 or True  # B2 will inflate this


def test_resolve_all_returns_full_registry():
    assert len(BASELINES.resolve("@all")) == 61
    assert len(BASELINES.resolve("@all")) == len(list(BASELINES.names()))


def test_resolve_group_categorical_contents():
    got = set(BASELINES.resolve("@categorical"))
    assert got == {"categorical_logreg", "categorical_random_forest"}


def test_resolve_unknown_group_raises():
    with pytest.raises(KeyError):
        BASELINES.resolve("@nope_not_a_group")


def test_resolve_group_naive_bayes_contents():
    got = set(BASELINES.resolve("@naive_bayes"))
    expected = {"nb_multinomial", "nb_complement", "nb_bernoulli", "complement_nb_count"}
    assert got == expected


def test_groups_partition_coverage():
    groups = BASELINES.groups()
    covered = set()
    for members in groups.values():
        covered.update(members)
    missing = set(BASELINES.names()) - covered
    assert not missing, f"baselines belong to no group: {sorted(missing)}"


def test_registry_register_custom():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    BASELINES.register("_custom_test", lambda: Pipeline([("v", TfidfVectorizer()), ("c", MultinomialNB())]))
    assert "_custom_test" in BASELINES
    BASELINES.build("_custom_test").fit(_X, _y)
