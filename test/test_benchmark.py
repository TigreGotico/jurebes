from jurebes.benchmark import compare, cross_validate, to_json, to_markdown, train_test


_X = (
    ["hello", "hi", "hey", "hello there", "hi friend", "hey friend"] * 3
    + ["tell joke", "say joke", "make me laugh", "do you know a joke", "tell me a joke", "say me a joke"] * 3
    + ["what is your name", "who are you", "tell me your name", "your name please", "name yourself", "what's your name"] * 3
)
_y = (["hello"] * 18) + (["joke"] * 18) + (["name"] * 18)


def test_train_test():
    r = train_test("logreg", _X, _y, test_size=0.3, seed=0)
    assert r.name == "logreg"
    assert 0.0 <= r.accuracy <= 1.0


def test_cross_validate():
    r = cross_validate("nb_multinomial", _X, _y, k=3)
    assert r.accuracy > 0.5
    assert set(r.per_class_f1) == {"hello", "joke", "name"}


def test_compare_markdown_json():
    cmp = compare(["logreg", "nb_multinomial"], _X, _y, k=2)
    assert len(cmp.rows) == 2
    md = to_markdown(cmp)
    assert "logreg" in md and "nb_multinomial" in md
    js = to_json(cmp)
    assert "logreg" in js
