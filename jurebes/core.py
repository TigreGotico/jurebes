"""Core IntentClassifier — pure-sklearn intent classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Union

import joblib
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class IntentResult:
    intent: Optional[str]
    confidence: float
    entities: Dict[str, str] = field(default_factory=dict)
    utterance: str = ""


def _has_proba(estimator) -> bool:
    final = estimator
    if hasattr(estimator, "steps"):
        final = estimator.steps[-1][1]
    return hasattr(final, "predict_proba")


class IntentClassifier:
    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        *,
        tagger: Optional[Any] = None,
        calibrate: bool = True,
    ):
        if estimator is None:
            from jurebes.baselines import BASELINES
            estimator = BASELINES.build("linear_svc")
        if calibrate and not _has_proba(estimator):
            estimator = CalibratedClassifierCV(estimator, cv=3)
        self.estimator = estimator
        self.tagger = tagger
        self._samples: Dict[str, List[str]] = {}
        self._entity_samples: Dict[str, List[str]] = {}
        self._fitted = False
        self._lock = RLock()

    def add_intent(self, name: str, samples: List[str]) -> None:
        with self._lock:
            self._samples.setdefault(name, []).extend(samples)

    def add_entity(self, name: str, samples: List[str]) -> None:
        if self.tagger is None:
            raise ValueError("no tagger configured; pass tagger=SklearnIOBTagger(...)")
        with self._lock:
            self._entity_samples.setdefault(name, []).extend(samples)
            self.tagger.add_entity(name, samples)

    def remove_intent(self, name: str) -> None:
        with self._lock:
            self._samples.pop(name, None)

    def remove_entity(self, name: str) -> None:
        with self._lock:
            self._entity_samples.pop(name, None)
            if self.tagger is not None:
                self.tagger.remove_entity(name)

    def _build_xy(self):
        X: List[str] = []
        y: List[str] = []
        for label, samples in self._samples.items():
            for s in samples:
                X.append(s)
                y.append(label)
        return X, y

    def fit(self) -> "IntentClassifier":
        with self._lock:
            X, y = self._build_xy()
            if len(set(y)) < 2:
                raise ValueError("need at least 2 intent classes to fit")
            self.estimator.fit(X, y)
            if self.tagger is not None:
                self.tagger.fit(self._samples)
            self._fitted = True
        return self

    def _extract_entities(self, utt: str) -> Dict[str, str]:
        if self.tagger is None or not getattr(self.tagger, "fitted", False):
            return {}
        try:
            return self.tagger.predict(utt)
        except Exception:
            return {}

    def predict(self, utterance: str) -> IntentResult:
        if not self._fitted:
            raise RuntimeError("classifier not fitted")
        ranked = self.predict_proba(utterance)
        return ranked[0] if ranked else IntentResult(None, 0.0, {}, utterance)

    def predict_proba(self, utterance: str) -> List[IntentResult]:
        if not self._fitted:
            raise RuntimeError("classifier not fitted")
        probs = self.estimator.predict_proba([utterance])[0]
        classes = self.estimator.classes_
        ents = self._extract_entities(utterance)
        results = [
            IntentResult(intent=c, confidence=float(p), entities=dict(ents), utterance=utterance)
            for c, p in zip(classes, probs)
        ]
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def predict_batch(self, utterances: List[str]) -> List[IntentResult]:
        return [self.predict(u) for u in utterances]

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump(
            {
                "estimator": self.estimator,
                "tagger": self.tagger,
                "samples": self._samples,
                "entity_samples": self._entity_samples,
                "fitted": self._fitted,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IntentClassifier":
        blob = joblib.load(str(path))
        inst = cls.__new__(cls)
        inst.estimator = blob["estimator"]
        inst.tagger = blob["tagger"]
        inst._samples = blob["samples"]
        inst._entity_samples = blob["entity_samples"]
        inst._fitted = blob["fitted"]
        inst._lock = RLock()
        return inst
