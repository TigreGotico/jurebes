"""Baselines registry — stub; populated in next commit."""

from __future__ import annotations

from typing import Callable, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


class _Registry:
    def __init__(self):
        self._items: Dict[str, Callable[[], Pipeline]] = {}

    def register(self, name: str, factory: Callable[[], Pipeline]) -> None:
        self._items[name] = factory

    def build(self, name: str) -> Pipeline:
        if name not in self._items:
            raise KeyError(f"unknown baseline: {name}")
        return self._items[name]()

    def names(self):
        return list(self._items.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._items


BASELINES = _Registry()


def _linear_svc():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", CalibratedClassifierCV(LinearSVC(), cv=3)),
    ])


BASELINES.register("linear_svc", _linear_svc)


def default() -> Pipeline:
    return BASELINES.build("linear_svc")
