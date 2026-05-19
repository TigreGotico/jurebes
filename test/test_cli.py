"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from jurebes.cli import main


_CSV = (
    "text,intent\n"
    "hello,hello\nhi,hello\nhey,hello\nhello friend,hello\n"
    "tell joke,joke\nsay joke,joke\nmake me laugh,joke\ndo you know a joke,joke\n"
    "what is your name,name\nwho are you,name\nname yourself,name\ntell me your name,name\n"
)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    p = tmp_path / "toy.csv"
    p.write_text(_CSV, encoding="utf-8")
    return p


def test_cli_list_baselines(capsys):
    rc = main(["list-baselines"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "nb_multinomial" in out


def test_cli_train_and_predict(tmp_path: Path, dataset: Path, capsys):
    model = tmp_path / "model.joblib"
    rc = main([
        "train", "--dataset", str(dataset),
        "--baseline", "nb_multinomial", "--out", str(model),
    ])
    assert rc == 0
    assert model.is_file()
    rc = main(["predict", "--model", str(model), "--text", "hello there"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "hello" in out


def test_cli_list_baselines_group_filter(capsys):
    rc = main(["list-baselines", "--group", "naive_bayes"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "nb_multinomial" in out
    assert "logreg" not in out


def test_cli_train_with_tagger(tmp_path: Path, dataset: Path):
    model = tmp_path / "m.joblib"
    rc = main([
        "train", "--dataset", str(dataset),
        "--baseline", "nb_multinomial",
        "--tagger", "--out", str(model),
    ])
    assert rc == 0
    from jurebes import IntentClassifier
    clf = IntentClassifier.load(model)
    assert clf.tagger is not None


def test_cli_search(tmp_path: Path, dataset: Path, capsys):
    model = tmp_path / "best.joblib"
    rc = main([
        "search", "--dataset", str(dataset),
        "--baseline", "nb_multinomial", "--backend", "random",
        "--cv", "3", "--n-iter", "3",
        "--space", "{'clf__alpha': [0.1, 0.5, 1.0]}",
        "--out", str(model),
    ])
    assert rc == 0
    assert model.is_file()
    out = capsys.readouterr().out
    assert "best_score" in out


def test_cli_benchmark_group_selector(tmp_path: Path, dataset: Path, capsys):
    rc = main([
        "benchmark", "--dataset", str(dataset),
        "--baselines", "@naive_bayes", "--cv", "3",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "nb_multinomial" in out


def test_cli_benchmark(tmp_path: Path, dataset: Path, capsys):
    rc = main([
        "benchmark", "--dataset", str(dataset),
        "--baselines", "nb_multinomial", "--cv", "3",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "nb_multinomial" in out
    assert "accuracy" in out
