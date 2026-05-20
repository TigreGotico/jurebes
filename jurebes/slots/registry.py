"""TAGGERS registry — pluggable slot-tagging strategies.

Built-in entries:
    - ``dictionary``   : :class:`DictionaryTagger` (gazetteer regex)
    - ``template``     : :class:`TemplateTagger`   (template regex with `{slot}`)
    - ``sklearn_iob``  : :class:`SklearnIOBTagger` (per-token sklearn classifier)
    - ``hybrid``       : :class:`HybridCascadeTagger` (dict → template → IOB cascade)
    - ``crf``          : :class:`CRFTagger` (lazy — requires ``sklearn-crfsuite``;
                         install via ``pip install jurebes[slots-crf]``)

The ``whole_span`` strategy (exhaustive n-gram binary classifier) is
deliberately not implemented yet — it is reserved for a future iteration.
"""

from __future__ import annotations

from typing import Callable, Dict, Set

from jurebes.slots.dictionary import DictionaryTagger
from jurebes.slots.hybrid import HybridCascadeTagger
from jurebes.slots.iob import SklearnIOBTagger
from jurebes.slots.template import TemplateTagger


class _TaggerRegistry:
    def __init__(self):
        self._items: Dict[str, Callable[[], object]] = {}
        self._groups: Dict[str, Set[str]] = {}

    def register(self, name: str, factory: Callable[[], object], *, group: str = None) -> None:
        self._items[name] = factory
        if group is not None:
            self._groups.setdefault(group, set()).add(name)

    def add_to_group(self, group: str, name: str) -> None:
        self._groups.setdefault(group, set()).add(name)

    def build(self, name: str):
        if name not in self._items:
            raise KeyError(f"unknown tagger: {name}")
        return self._items[name]()

    def names(self):
        return list(self._items.keys())

    def groups(self) -> Dict[str, Set[str]]:
        return {g: {n for n in members if n in self._items} for g, members in self._groups.items()}

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


def _build_crf():
    try:
        from jurebes.slots.crf import CRFTagger
    except ImportError as exc:  # pragma: no cover - exercised via tests indirectly
        raise ImportError(
            "install jurebes[slots-crf] to use the CRF tagger"
        ) from exc
    return CRFTagger()


TAGGERS = _TaggerRegistry()
TAGGERS.register("dictionary", lambda: DictionaryTagger(), group="rule_based")
TAGGERS.register("template", lambda: TemplateTagger(), group="rule_based")
TAGGERS.register("sklearn_iob", lambda: SklearnIOBTagger(), group="ml")
TAGGERS.register("hybrid", lambda: HybridCascadeTagger(), group="hybrid")
TAGGERS.register("crf", _build_crf, group="ml")
