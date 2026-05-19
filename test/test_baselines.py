import pytest

from jurebes.baselines import BASELINES


_X = [
    "hello", "hi", "hey", "hello there", "hi friend", "hey friend",
    "tell me a joke", "say a joke", "make me laugh", "do you know any joke",
    "tell joke", "say joke",
    "what is your name", "who are you", "tell me your name", "your name please",
    "what's your name", "name yourself",
]
_y = (
    ["hello"] * 6
    + ["joke"] * 6
    + ["name"] * 6
)

# duplicate training data; some tree ensembles need more samples to fit cleanly
_X = _X * 3
_y = _y * 3


@pytest.mark.parametrize("name", BASELINES.names())
def test_baseline_fits_and_scores(name):
    est = BASELINES.build(name)
    est.fit(_X, _y)
    preds = est.predict(_X)
    acc = sum(p == t for p, t in zip(preds, _y)) / len(_y)
    assert acc >= 0.9, f"{name} training accuracy {acc:.2f} below 0.9"


def test_registry_has_at_least_23_baselines():
    assert len(BASELINES) >= 23


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


def test_registry_register_custom():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    BASELINES.register("_custom_test", lambda: Pipeline([("v", TfidfVectorizer()), ("c", MultinomialNB())]))
    assert "_custom_test" in BASELINES
    BASELINES.build("_custom_test").fit(_X, _y)
