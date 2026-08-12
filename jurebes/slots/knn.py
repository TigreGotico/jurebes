"""KNNTagger — nearest-utterance slot transfer.

At predict time, the input utterance is TF-IDF-vectorised and matched
against the training utterances via :class:`sklearn.neighbors.NearestNeighbors`.
The IOB tag sequences of the ``k`` nearest neighbours are aligned by
token position and majority-voted onto the input, then decoded into a
slot dictionary.

Handles unseen entity values via pattern transfer: a training sample
``"weather in lisbon"`` (tagged ``O O B-city``) at neighbour distance
will project the same ``B-city`` tag onto the corresponding position of
``"weather in paris"`` even though ``paris`` was never in the gazetteer.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from jurebes.slots.iob import _build_iob, tokenize


class KNNTagger:
    """k-nearest-neighbour slot tagger over training utterances."""

    def __init__(
        self,
        k: int = 3,
        vectorizer: Optional[TfidfVectorizer] = None,
        metric: str = "cosine",
    ):
        self.k = k
        self.vectorizer = vectorizer or TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.metric = metric
        self._entity_samples: Dict[str, List[str]] = {}
        self._train_tokens: List[List[str]] = []
        self._train_tags: List[List[str]] = []
        self._train_text: List[str] = []
        self._nn: Optional[NearestNeighbors] = None
        self.fitted: bool = False

    def add_entity(self, name: str, samples: List[str]) -> None:
        self._entity_samples[name] = list(samples)

    def remove_entity(self, name: str) -> None:
        self._entity_samples.pop(name, None)

    def fit(self, intent_samples: Dict[str, List[str]]) -> "KNNTagger":
        self._train_tokens = []
        self._train_tags = []
        self._train_text = []
        for samples in intent_samples.values():
            for sample in samples:
                for tokens, tags in _build_iob(sample, self._entity_samples):
                    self._train_tokens.append(tokens)
                    self._train_tags.append(tags)
                    self._train_text.append(" ".join(tokens))
        if not self._train_text:
            self.fitted = False
            return self
        X = self.vectorizer.fit_transform(self._train_text)
        self._nn = NearestNeighbors(
            n_neighbors=min(self.k, len(self._train_text)),
            metric=self.metric,
        )
        self._nn.fit(X)
        self.fitted = True
        return self

    def tag(self, text: str) -> List[Tuple[str, str]]:
        if not self.fitted or self._nn is None:
            tokens = tokenize(text)
            return [(t, "O") for t in tokens]
        tokens = tokenize(text)
        vec = self.vectorizer.transform([" ".join(tokens)])
        _, indices = self._nn.kneighbors(vec, n_neighbors=min(self.k, len(self._train_text)))
        labels: List[str] = []
        for i in range(len(tokens)):
            votes = Counter()
            for nb_idx in indices[0]:
                nb_tags = self._train_tags[nb_idx]
                if i < len(nb_tags):
                    votes[nb_tags[i]] += 1
            label = votes.most_common(1)[0][0] if votes else "O"
            labels.append(label)
        return list(zip(tokens, labels))

    def predict(self, text: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        current_entity: Optional[str] = None
        current_value: List[str] = []
        for token, tag in self.tag(text):
            if tag.startswith("B-"):
                if current_entity is not None:
                    out[current_entity] = " ".join(current_value)
                current_entity = tag[2:]
                current_value = [token]
            elif tag.startswith("I-") and current_entity == tag[2:]:
                current_value.append(token)
            else:
                if current_entity is not None:
                    out[current_entity] = " ".join(current_value)
                    current_entity = None
                    current_value = []
        if current_entity is not None:
            out[current_entity] = " ".join(current_value)
        return out

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        payload = {
            "k": self.k,
            "metric": self.metric,
            "entity_samples": self._entity_samples,
            "train_tokens": self._train_tokens,
            "train_tags": self._train_tags,
            "train_text": self._train_text,
            "fitted": self.fitted,
            "vectorizer": self.vectorizer,
            "nn": self._nn,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "KNNTagger":
        payload = joblib.load(Path(path))
        obj = cls(k=payload["k"], vectorizer=payload["vectorizer"], metric=payload["metric"])
        obj._entity_samples = payload["entity_samples"]
        obj._train_tokens = payload["train_tokens"]
        obj._train_tags = payload["train_tags"]
        obj._train_text = payload["train_text"]
        obj._nn = payload["nn"]
        obj.fitted = payload["fitted"]
        return obj
