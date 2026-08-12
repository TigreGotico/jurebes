"""TemplateTagger — regex template matching with named slot groups."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


_SLOT_RE = re.compile(r"\{(\w+)\}")
_ALT_RE = re.compile(r"\(([^()]*\|[^()]*)\)")


def _compile_template(template: str) -> re.Pattern:
    """Compile a template like ``"(weather|temperature) in {city}"`` into regex.

    A slot name repeated within one template (e.g. MASSIVE's
    ``"{relation} of my {relation}"``) cannot reuse the same Python named
    group. The second and later occurrences get a suffixed group name
    (``relation__2``); :meth:`TemplateTagger.predict` collapses the suffix
    back to the base slot name.
    """
    out: List[str] = []
    seen: Dict[str, int] = {}
    i = 0
    n = len(template)
    while i < n:
        m_slot = _SLOT_RE.match(template, i)
        m_alt = _ALT_RE.match(template, i)
        if m_slot:
            name = m_slot.group(1)
            seen[name] = seen.get(name, 0) + 1
            group = name if seen[name] == 1 else f"{name}__{seen[name]}"
            out.append(rf"(?P<{group}>.+?)")
            i = m_slot.end()
        elif m_alt:
            parts = [re.escape(p) for p in m_alt.group(1).split("|")]
            out.append("(?:" + "|".join(parts) + ")")
            i = m_alt.end()
        else:
            out.append(re.escape(template[i]))
            i += 1
    pattern = "^" + "".join(out) + "$"
    return re.compile(pattern, re.IGNORECASE)


class TemplateTagger:
    """Template-driven tagger: stores `{slot}` templates and matches utterances."""

    def __init__(self):
        self._intent_samples: Dict[str, List[str]] = {}
        self._compiled: List[Tuple[str, re.Pattern]] = []
        self.fitted = False

    # ── data registration ───────────────────────────────────────────
    def add_intent(self, name: str, samples: List[str]) -> None:
        self._intent_samples.setdefault(name, []).extend(samples)

    def add_entity(self, name: str, samples: List[str]) -> None:
        # templates don't need entity gazetteers; accepted for interface symmetry
        return

    def remove_entity(self, name: str) -> None:
        return

    # ── training ────────────────────────────────────────────────────
    def fit(self, intent_samples: Optional[Dict[str, List[str]]] = None) -> "TemplateTagger":
        if intent_samples is not None:
            for name, samples in intent_samples.items():
                self._intent_samples.setdefault(name, []).extend(samples)
        templates: List[str] = []
        for samples in self._intent_samples.values():
            for s in samples:
                if "{" in s:
                    templates.append(s)
        # determinism: longest template first
        templates.sort(key=len, reverse=True)
        self._compiled = [(t, _compile_template(t)) for t in templates]
        self.fitted = True
        return self

    # ── prediction ──────────────────────────────────────────────────
    def predict(self, utterance: str) -> Dict[str, str]:
        for _t, pat in self._compiled:
            m = pat.match(utterance.strip())
            if m:
                out: Dict[str, str] = {}
                for k, v in m.groupdict().items():
                    if v is None:
                        continue
                    # Collapse a repeated-slot suffix (relation__2 -> relation);
                    # the first non-empty occurrence wins.
                    base = k.split("__")[0]
                    out.setdefault(base, v.strip())
                return out
        return {}

    def tag(self, text: str) -> List[Tuple[str, str]]:
        from jurebes.slots.iob import tokenize
        tokens = tokenize(text)
        if not tokens:
            return []
        slots = self.predict(text)
        if not slots:
            return [(t, "O") for t in tokens]
        tags = ["O"] * len(tokens)
        lower = [t.lower() for t in tokens]
        for name, value in slots.items():
            val_toks = [t.lower() for t in re.findall(r"\w+|[^\w\s]", value)]
            if not val_toks:
                continue
            for i in range(len(tokens) - len(val_toks) + 1):
                if lower[i:i + len(val_toks)] == val_toks:
                    tags[i] = f"B-{name}"
                    for j in range(1, len(val_toks)):
                        tags[i + j] = f"I-{name}"
                    break
        return list(zip(tokens, tags))

    # ── persistence ─────────────────────────────────────────────────
    def save(self, path: Union[str, Path]) -> None:
        blob = {"intent_samples": self._intent_samples, "fitted": self.fitted}
        Path(path).write_text(json.dumps(blob), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TemplateTagger":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        inst = cls()
        inst._intent_samples = {k: list(v) for k, v in blob.get("intent_samples", {}).items()}
        if blob.get("fitted"):
            inst.fit()
        return inst
