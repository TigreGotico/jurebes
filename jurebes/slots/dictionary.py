"""DictionaryTagger — gazetteer-based slot tagging via compiled regex alternations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from jurebes.slots.iob import tokenize


class DictionaryTagger:
    """Gazetteer tagger: registers entity values and matches by regex.

    Training-free. Multi-token entity values are matched as exact substrings.
    Single-word values are anchored at word boundaries to avoid mid-word matches.
    """

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
        self._entity_samples: Dict[str, List[str]] = {}
        self._patterns: Dict[str, re.Pattern] = {}
        self.fitted = True  # always "fitted" — no training needed

    # ── data registration ───────────────────────────────────────────
    def add_entity(self, name: str, samples: List[str]) -> None:
        self._entity_samples.setdefault(name, []).extend(samples)
        self._compile(name)

    def remove_entity(self, name: str) -> None:
        self._entity_samples.pop(name, None)
        self._patterns.pop(name, None)

    def _compile(self, name: str) -> None:
        vals = sorted(self._entity_samples.get(name, []), key=len, reverse=True)
        if not vals:
            self._patterns.pop(name, None)
            return
        parts = []
        for v in vals:
            esc = re.escape(v)
            # word-boundary anchor for single-word atomic values
            if re.fullmatch(r"\w+", v):
                parts.append(rf"\b{esc}\b")
            else:
                parts.append(esc)
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self._patterns[name] = re.compile("(?:" + "|".join(parts) + ")", flags)

    # ── training interface (no-op for symmetry) ─────────────────────
    def fit(self, intent_samples: Optional[Dict[str, List[str]]] = None) -> "DictionaryTagger":
        # rebuild all patterns in case entities were added without compile
        for name in list(self._entity_samples.keys()):
            self._compile(name)
        self.fitted = True
        return self

    # ── prediction ──────────────────────────────────────────────────
    def predict(self, utterance: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        # earliest match per entity wins
        for name, pat in self._patterns.items():
            m = pat.search(utterance)
            if m:
                out[name] = m.group(0)
        return out

    def tag(self, text: str) -> List[Tuple[str, str]]:
        """Reconstruct BIO-labelled token sequence from regex hits."""
        tokens = tokenize(text)
        if not tokens:
            return []
        tags = ["O"] * len(tokens)
        # token char spans
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for tok in tokens:
            idx = text.find(tok, cursor) if self.case_sensitive else text.lower().find(tok.lower(), cursor)
            if idx < 0:
                idx = cursor
            spans.append((idx, idx + len(tok)))
            cursor = idx + len(tok)
        for name, pat in self._patterns.items():
            for m in pat.finditer(text):
                a, b = m.start(), m.end()
                hit = [i for i, (s, e) in enumerate(spans) if s < b and e > a]
                if not hit:
                    continue
                tags[hit[0]] = f"B-{name}"
                for j in hit[1:]:
                    tags[j] = f"I-{name}"
        return list(zip(tokens, tags))

    # ── persistence ─────────────────────────────────────────────────
    def save(self, path: Union[str, Path]) -> None:
        blob = {
            "case_sensitive": self.case_sensitive,
            "entity_samples": self._entity_samples,
        }
        Path(path).write_text(json.dumps(blob), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DictionaryTagger":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        inst = cls(case_sensitive=blob.get("case_sensitive", False))
        for name, vals in blob.get("entity_samples", {}).items():
            inst.add_entity(name, list(vals))
        return inst
