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


def test_compare_with_multi_scoring():
    cmp = compare(
        ["logreg", "nb_multinomial"], _X, _y, k=2,
        scoring=("accuracy", "f1_macro", "log_loss"),
    )
    md = to_markdown(cmp)
    # log_loss should appear as an extra column in the markdown table
    assert "log_loss" in md
    for r in cmp.rows:
        assert "log_loss" in r.extra_scores


def test_pooled_percentiles_helper():
    from jurebes.benchmark import pooled_percentiles
    out = pooled_percentiles([1.0, 2.0, 3.0, 4.0, 100.0])
    assert out["p50"] == 3.0
    assert out["p95"] > 50.0
    assert out["mean"] > 0


def test_run_result_has_pooled_fields():
    r = cross_validate("nb_multinomial", _X, _y, k=2)
    assert r.predict_ms_p95_pooled >= 0.0
    assert r.predict_ms_mean >= 0.0


def test_to_markdown_group_column():
    cmp = compare(["logreg"], _X, _y, k=2)
    md = to_markdown(cmp)
    # group column header + logreg's group ("linear") should appear
    assert "group" in md
    assert "linear" in md


def test_to_markdown_sort_by_accuracy():
    cmp = compare(["logreg", "nb_multinomial"], _X, _y, k=2)
    md = to_markdown(cmp, sort_by="accuracy")
    lines = md.splitlines()
    # ensure both rows are present
    assert any("logreg" in ln for ln in lines)


def test_plot_optional_dep(monkeypatch):
    import sys, pytest
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    sys.modules.pop("jurebes.benchmark.plot", None)
    from jurebes.benchmark.plot import plot_comparison
    cmp = compare(["logreg"], _X, _y, k=2)
    with pytest.raises(ImportError, match=r"jurebes\[bench-plot\]"):
        plot_comparison(cmp)
