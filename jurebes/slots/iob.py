"""SklearnIOBTagger — per-token sklearn classifier for IOB slot tagging."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from jurebes.slots.features import token_features

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


def _build_iob(sample: str, entity_samples: Dict[str, List[str]]) -> List[Tuple[List[str], List[str]]]:
    out: List[Tuple[List[str], List[str]]] = []
    if "{" not in sample:
        toks = tokenize(sample)
        out.append((toks, ["O"] * len(toks)))
        return out
    placeholders = re.findall(r"\{(\w+)\}", sample)
    if not placeholders or not all(p in entity_samples for p in placeholders):
        toks = tokenize(sample)
        out.append((toks, ["O"] * len(toks)))
        return out

    def expand(template: str, remaining: List[str]):
        if not remaining:
            tokens: List[str] = []
            tags: List[str] = []
            parts = re.split(r"(__SLOT_\w+__)", template)
            for part in parts:
                m = re.match(r"__SLOT_(\w+)__", part)
                if m:
                    ent = m.group(1)
                    slot_val = current[ent]
                    slot_toks = tokenize(slot_val)
                    if not slot_toks:
                        continue
                    tokens.extend(slot_toks)
                    tags.append(f"B-{ent}")
                    tags.extend([f"I-{ent}"] * (len(slot_toks) - 1))
                else:
                    part_toks = tokenize(part)
                    tokens.extend(part_toks)
                    tags.extend(["O"] * len(part_toks))
            if tokens:
                out.append((tokens, tags))
            return
        ent = remaining[0]
        for val in entity_samples[ent]:
            current[ent] = val
            expand(template, remaining[1:])

    marked = sample
    for p in placeholders:
        marked = marked.replace("{" + p + "}", f"__SLOT_{p}__")
    current: Dict[str, str] = {}
    expand(marked, list(dict.fromkeys(placeholders)))
    return out


class SklearnIOBTagger:
    def __init__(self, estimator=None):
        self.estimator = estimator or Pipeline(
            [("vec", DictVectorizer(sparse=True)), ("clf", LogisticRegression(max_iter=1000))]
        )
        self._entity_samples: Dict[str, List[str]] = {}
        self._intent_samples: Dict[str, List[str]] = {}
        self.fitted = False

    def add_entity(self, name: str, samples: List[str]) -> None:
        self._entity_samples.setdefault(name, []).extend(samples)

    def remove_entity(self, name: str) -> None:
        self._entity_samples.pop(name, None)

    def fit(self, intent_samples: Dict[str, List[str]]) -> "SklearnIOBTagger":
        self._intent_samples = intent_samples
        X: List[dict] = []
        y: List[str] = []
        for samples in intent_samples.values():
            for s in samples:
                for tokens, tags in _build_iob(s, self._entity_samples):
                    for i, tag in enumerate(tags):
                        X.append(token_features(tokens, i))
                        y.append(tag)
        if not X or len(set(y)) < 2:
            self.fitted = False
            return self
        self.estimator.fit(X, y)
        self.fitted = True
        return self

    def tag(self, text: str) -> List[Tuple[str, str]]:
        if not self.fitted:
            return []
        tokens = tokenize(text)
        if not tokens:
            return []
        feats = [token_features(tokens, i) for i in range(len(tokens))]
        tags = self.estimator.predict(feats)
        return list(zip(tokens, tags))

    def predict(self, text: str) -> Dict[str, str]:
        tagged = self.tag(text)
        entities: Dict[str, str] = {}
        current_ent: Optional[str] = None
        current_words: List[str] = []
        for word, tag in tagged:
            if tag == "O":
                if current_ent is not None:
                    entities[current_ent] = " ".join(current_words)
                    current_ent = None
                    current_words = []
                continue
            prefix, _, ent = tag.partition("-")
            if prefix == "B" or current_ent != ent:
                if current_ent is not None:
                    entities[current_ent] = " ".join(current_words)
                current_ent = ent
                current_words = [word]
            else:
                current_words.append(word)
        if current_ent is not None:
            entities[current_ent] = " ".join(current_words)
        return entities

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump(
            {
                "estimator": self.estimator,
                "entity_samples": self._entity_samples,
                "fitted": self.fitted,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SklearnIOBTagger":
        blob = joblib.load(str(path))
        inst = cls(estimator=blob["estimator"])
        inst._entity_samples = blob["entity_samples"]
        inst.fitted = blob["fitted"]
        return inst
