"""HybridCascadeTagger — fan-out + merge across multiple constituent taggers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib

from jurebes.slots.dictionary import DictionaryTagger
from jurebes.slots.iob import SklearnIOBTagger
from jurebes.slots.template import TemplateTagger


class HybridCascadeTagger:
    """Cascade taggers in order. Earlier taggers win on key collisions."""

    def __init__(self, taggers: Optional[List] = None):
        self.taggers = taggers if taggers is not None else [
            DictionaryTagger(),
            TemplateTagger(),
            SklearnIOBTagger(),
        ]
        self.fitted = False

    def _fanout(self, method: str, *args, **kwargs) -> None:
        for tg in self.taggers:
            fn = getattr(tg, method, None)
            if callable(fn):
                try:
                    fn(*args, **kwargs)
                except Exception:
                    continue

    def add_entity(self, name: str, samples: List[str]) -> None:
        self._fanout("add_entity", name, samples)

    def remove_entity(self, name: str) -> None:
        self._fanout("remove_entity", name)

    def add_intent(self, name: str, samples: List[str]) -> None:
        self._fanout("add_intent", name, samples)

    def fit(self, intent_samples: Optional[Dict[str, List[str]]] = None) -> "HybridCascadeTagger":
        for tg in self.taggers:
            try:
                tg.fit(intent_samples) if intent_samples is not None else tg.fit()
            except TypeError:
                # tagger expects a mandatory positional arg
                try:
                    tg.fit({})
                except Exception:
                    pass
            except Exception:
                continue
        self.fitted = True
        return self

    def predict(self, utterance: str) -> Dict[str, str]:
        merged: Dict[str, str] = {}
        for tg in self.taggers:
            try:
                out = tg.predict(utterance) or {}
            except Exception:
                out = {}
            for k, v in out.items():
                if k not in merged:
                    merged[k] = v
        return merged

    def tag(self, text: str) -> List[Tuple[str, str]]:
        for tg in self.taggers:
            try:
                tagged = tg.tag(text)
                if tagged and any(t != "O" for _, t in tagged):
                    return tagged
            except Exception:
                continue
        return []

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump({"taggers": self.taggers, "fitted": self.fitted}, str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HybridCascadeTagger":
        blob = joblib.load(str(path))
        inst = cls(taggers=blob["taggers"])
        inst.fitted = blob.get("fitted", False)
        return inst
