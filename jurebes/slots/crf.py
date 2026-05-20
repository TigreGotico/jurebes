"""CRFTagger — optional CRF slot tagger via sklearn-crfsuite."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib

from jurebes.slots.features import token_features
from jurebes.slots.iob import _build_iob, tokenize


_INSTALL_HINT = "install jurebes[slots-crf] to use the CRF tagger"


class CRFTagger:
    """CRF-based IOB tagger. Requires ``sklearn-crfsuite``."""

    def __init__(self):
        self.model = None
        self._entity_samples: Dict[str, List[str]] = {}
        self._intent_samples: Dict[str, List[str]] = {}
        self.fitted = False

    def add_entity(self, name: str, samples: List[str]) -> None:
        self._entity_samples.setdefault(name, []).extend(samples)

    def remove_entity(self, name: str) -> None:
        self._entity_samples.pop(name, None)

    def fit(self, intent_samples: Dict[str, List[str]]) -> "CRFTagger":
        try:
            import sklearn_crfsuite  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_INSTALL_HINT) from exc
        from sklearn_crfsuite import CRF

        self._intent_samples = intent_samples
        X_seqs: List[List[dict]] = []
        y_seqs: List[List[str]] = []
        for samples in intent_samples.values():
            for s in samples:
                for tokens, tags in _build_iob(s, self._entity_samples):
                    X_seqs.append([token_features(tokens, i) for i in range(len(tokens))])
                    y_seqs.append(list(tags))
        if not X_seqs:
            self.fitted = False
            return self
        self.model = CRF(algorithm="lbfgs", max_iterations=100, all_possible_transitions=True)
        self.model.fit(X_seqs, y_seqs)
        self.fitted = True
        return self

    def tag(self, text: str) -> List[Tuple[str, str]]:
        if not self.fitted or self.model is None:
            return []
        tokens = tokenize(text)
        if not tokens:
            return []
        feats = [token_features(tokens, i) for i in range(len(tokens))]
        tags = self.model.predict_single(feats)
        return list(zip(tokens, tags))

    def predict(self, text: str) -> Dict[str, str]:
        tagged = self.tag(text)
        entities: Dict[str, str] = {}
        cur_ent: Optional[str] = None
        cur_words: List[str] = []
        for word, tag in tagged:
            if tag == "O":
                if cur_ent is not None:
                    entities[cur_ent] = " ".join(cur_words)
                    cur_ent, cur_words = None, []
                continue
            prefix, _, ent = tag.partition("-")
            if prefix == "B" or cur_ent != ent:
                if cur_ent is not None:
                    entities[cur_ent] = " ".join(cur_words)
                cur_ent, cur_words = ent, [word]
            else:
                cur_words.append(word)
        if cur_ent is not None:
            entities[cur_ent] = " ".join(cur_words)
        return entities

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump(
            {
                "model": self.model,
                "entity_samples": self._entity_samples,
                "fitted": self.fitted,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CRFTagger":
        blob = joblib.load(str(path))
        inst = cls()
        inst.model = blob["model"]
        inst._entity_samples = blob["entity_samples"]
        inst.fitted = blob["fitted"]
        return inst
