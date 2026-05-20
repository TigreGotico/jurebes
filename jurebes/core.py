"""Core IntentClassifier — pure-sklearn intent classification."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Literal, Optional, Union

import joblib
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

from jurebes.datasets.expansion import expand_template

_WS_RE = re.compile(r"\s+")


def _expand_samples(samples: List[str]) -> List[str]:
    """Expand OVOS ``(a|b)``/``[opt]`` syntax in ``samples``.

    ``{slot}`` placeholders are preserved so slot taggers still learn
    them. Whitespace from removed optionals is collapsed and duplicates
    are dropped while preserving insertion order. Samples without
    template metacharacters are passed through untouched.
    """
    out: List[str] = []
    seen: set = set()
    for s in samples:
        variants = expand_template(s) if ("(" in s or "[" in s) else [s]
        for v in variants:
            v = _WS_RE.sub(" ", v).strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
    return out


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
        calibrate: Union[Literal["if_missing", "always"], bool] = "if_missing",
    ):
        """Initialise the classifier.

        Args:
            estimator: any sklearn-compatible classifier (typically a Pipeline).
            tagger: optional slot tagger with ``add_entity``/``fit``/``predict``.
            calibrate: probability-calibration mode.

                - ``"if_missing"`` (default, also ``True``): wrap with
                  CalibratedClassifierCV only when the estimator lacks
                  ``predict_proba``. Recommended for most pipelines.
                - ``"always"``: always wrap, even when ``predict_proba``
                  exists. Useful for tree/forest/kNN classifiers whose
                  native probabilities are unreliable.
                - ``False``: never wrap. If the estimator has no
                  ``predict_proba`` this raises ``ValueError`` immediately.

        Raises:
            ValueError: if ``calibrate=False`` and the estimator has no
                ``predict_proba``.
        """
        if estimator is None:
            from jurebes.baselines import BASELINES
            estimator = BASELINES.build("linear_svc")
        # Normalise: True → "if_missing", False stays False
        mode: Union[str, bool] = "if_missing" if calibrate is True else calibrate
        has_proba = _has_proba(estimator)
        if mode == "always":
            estimator = CalibratedClassifierCV(estimator, cv=3)
        elif mode == "if_missing":
            if not has_proba:
                estimator = CalibratedClassifierCV(estimator, cv=3)
        elif mode is False:
            if not has_proba:
                raise ValueError(
                    "estimator has no predict_proba and calibrate=False; "
                    "pass calibrate='if_missing' or 'always' to enable wrapping"
                )
        else:
            raise ValueError(f"unknown calibrate mode: {calibrate!r}")
        self.estimator = estimator
        if isinstance(tagger, str):
            from jurebes.slots import TAGGERS
            tagger = TAGGERS.build(tagger)
        self.tagger = tagger
        self._samples: Dict[str, List[str]] = {}
        self._entity_samples: Dict[str, List[str]] = {}
        self._fitted = False
        self._lock = RLock()

    def add_intent(self, name: str, samples: List[str]) -> None:
        with self._lock:
            self._samples.setdefault(name, []).extend(_expand_samples(samples))

    def add_entity(self, name: str, samples: List[str]) -> None:
        if self.tagger is None:
            raise ValueError("no tagger configured; pass tagger=SklearnIOBTagger(...)")
        with self._lock:
            samples = _expand_samples(samples)
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
        try:
            ranked = self.predict_proba(utterance)
            return ranked[0] if ranked else IntentResult(None, 0.0, {}, utterance)
        except (AttributeError, NotImplementedError):
            pred = self.estimator.predict([utterance])[0]
            ents = self._extract_entities(utterance)
            return IntentResult(
                intent=str(pred), confidence=1.0,
                entities=dict(ents), utterance=utterance,
            )

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
        if not self._fitted:
            raise RuntimeError("classifier not fitted")
        if not utterances:
            return []
        # If estimator has predict_proba, get top-1 confidence per row in one shot.
        try:
            probs = self.estimator.predict_proba(list(utterances))
            classes = list(self.estimator.classes_)
            results: List[IntentResult] = []
            for utt, row in zip(utterances, probs):
                idx = int(max(range(len(row)), key=lambda i: row[i]))
                ents = self._extract_entities(utt)
                results.append(IntentResult(
                    intent=classes[idx], confidence=float(row[idx]),
                    entities=dict(ents), utterance=utt,
                ))
            return results
        except (AttributeError, NotImplementedError):
            preds = self.estimator.predict(list(utterances))
            results = []
            for utt, pred in zip(utterances, preds):
                ents = self._extract_entities(utt)
                results.append(IntentResult(
                    intent=str(pred), confidence=1.0,
                    entities=dict(ents), utterance=utt,
                ))
            return results

    def save(self, path: Union[str, Path]) -> None:
        from jurebes.version import __version__ as _jv
        joblib.dump(
            {
                "_jurebes_version": _jv,
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
        inst._loaded_from_version = blob.get("_jurebes_version")
        return inst
